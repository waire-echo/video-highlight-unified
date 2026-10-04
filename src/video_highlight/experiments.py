"""Small file-backed experiment registry, with reproducible source snapshots."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

from .prompts import bundle_hashes

REPO = Path(__file__).resolve().parents[2]
FIELDS = ["experiment_id", "date", "author", "pipeline", "model", "dataset", "prompt_version",
          "config", "git_commit", "status", "score", "metric_type", "notes"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_path(experiment_id: str):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,95}", experiment_id):
        raise ValueError("experiment_id must be 3-96 lowercase letters, digits, '_' or '-'")
    return REPO / "experiments/records" / experiment_id


def current_commit(*, require_clean=True):
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    state = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all"], text=True)
    dirty = [line for line in state.splitlines() if not line[3:].startswith("experiments/")]
    if require_clean and dirty:
        raise ValueError("Commit source/config/prompt changes before a formal experiment: " + "; ".join(dirty[:5]))
    return commit


def source_fingerprint():
    entries = subprocess.check_output(["git", "-C", str(REPO), "ls-files", "--stage", "-z"]).decode().split('\0')
    files = {}
    for entry in filter(None, entries):
        metadata, name = entry.split('\t', 1)
        if not name.startswith('experiments/'):
            files[name] = metadata.split()[1]
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def registry():
    rows = []
    for file in sorted((REPO / "experiments/records").glob("*/experiment.json")):
        record = read_json(file)
        metrics = read_json(file.parent / "metrics.json") if (file.parent / "metrics.json").exists() else {}
        rows.append({**{field: record.get(field, "") for field in FIELDS},
                     "model": record.get("model", {}).get("id", "unknown") if isinstance(record.get("model"), dict) else record.get("model"),
                     "dataset": record.get("dataset", {}).get("id", "unknown") if isinstance(record.get("dataset"), dict) else record.get("dataset"),
                     "prompt_version": ";".join(record.get("prompts", {}).values()),
                     "score": metrics.get("score", ""), "metric_type": metrics.get("metric_type", "")})
    path = REPO / "experiments/registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return rows


def create(experiment_id, author, config, *, notes=""):
    if not author.strip():
        raise ValueError("Author is required")
    model = config.get('model', {})
    if model.get('id') not in ('none', None) and model.get('revision') in (None, '', 'unknown', 'local-snapshot-unresolved'):
        raise ValueError('Set the exact model revision before registering a formal experiment')
    target = experiment_path(experiment_id)
    commit = current_commit()
    hashes = bundle_hashes(config.get("prompts", {}))
    serialized = json.dumps(config, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    record = {"schema_version": "experiment/v1", "experiment_id": experiment_id,
              "date": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(), "author": author,
              "pipeline": config["pipeline"], "model": config.get("model", {}), "dataset": config["dataset"],
              "prompts": config.get("prompts", {}), "prompt_hashes": hashes,
              "config": f"experiments/records/{experiment_id}/config.json",
              "config_sha256": hashlib.sha256(serialized).hexdigest(),
              "git_commit": commit, "source_fingerprint": source_fingerprint(), "status": "created", "notes": notes}
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / "config.json", config)
    write_json(target / "experiment.json", record)
    registry()
    return record


def verify(experiment_id):
    path = experiment_path(experiment_id)
    record = read_json(path / "experiment.json")
    config = read_json(path / "config.json")
    digest = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if record["config_sha256"] != digest or record["prompt_hashes"] != bundle_hashes(config.get("prompts", {})):
        raise ValueError("Experiment configuration or prompt assets changed; create a new experiment")
    current_commit()
    if record["source_fingerprint"] != source_fingerprint():
        raise ValueError("Experiment belongs to another code commit; check out that commit or create a new experiment")
    return record, config


def record_metrics(experiment_id, metrics):
    if not isinstance(metrics, dict) or not metrics.get("metric_type"):
        raise ValueError("Metrics require metric_type (leaderboard, temporal_proxy, engineering, ...)")
    if "score" in metrics and (type(metrics["score"]) not in (int, float) or not math.isfinite(metrics["score"])):
        raise ValueError("Score must be finite")
    path = experiment_path(experiment_id)
    if not (path / "experiment.json").exists():
        raise ValueError("Unknown experiment")
    write_json(path / "metrics.json", metrics)
    registry()
