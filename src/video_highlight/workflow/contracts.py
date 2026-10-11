"""Small, atomic run records shared by CLI and future user interfaces."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import os


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def run_lock(root):
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "workflow.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another workflow owns this run directory") from error
        yield


class RunState:
    def __init__(self, root, plan):
        self.root = Path(root)
        self.state = {"schema_version": "workflow-status/v1", "version": plan["version"],
                      "plan_sha256": fingerprint(plan), "pid": os.getpid(),
                      "status": "running", "stage": "initializing", "started_utc": stamp(),
                      "requested_ids": plan["config"]["ids"], "completed_ids": []}

    def update(self, **fields):
        self.state.update(fields, updated_utc=stamp())
        write_json(self.root / "status.json", self.state)

    def event(self, kind, **fields):
        record = {"schema_version": "workflow-event/v1", "at_utc": stamp(), "event": kind, **fields}
        path = self.root / "events.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def failure(self, error):
        record = {"stage": self.state["stage"], "error_type": type(error).__name__,
                  "error": str(error), "at_utc": stamp(), "safe_resume": "same frozen inputs required",
                  "video_id": self.state.get("current_video"), "segment": self.state.get("segment"),
                  "log": self.state.get("log")}
        path = self.root / "errors/failed_items.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=record)
        self.event("failed", **record)
