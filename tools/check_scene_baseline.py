#!/usr/bin/env python3
"""Enforce the frozen scene contract without requiring a full Git checkout."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONTRACT = "configs/workflows/frozen_contracts.json"
RELEASES = "configs/workflows/versions"


def check(repo=REPO):
    repo = Path(repo).resolve()
    contract = json.loads((repo / CONTRACT).read_text())
    errors = []
    for name, expected in contract["sha256"].items():
        file = (repo / name).resolve()
        if (not file.is_relative_to(repo) or not file.is_file() or
                hashlib.sha256(file.read_bytes()).hexdigest() != expected):
            errors.append("Frozen file changed or removed: " + name)
    for file in sorted((repo / RELEASES).glob("*.json")):
        descriptor = json.loads(file.read_text())
        if descriptor.get("schema_version") != "workflow-release/v1" or descriptor.get("id") != file.stem:
            errors.append("Invalid release identity: " + str(file))
            continue
        source = (repo / descriptor["source_root"]).resolve()
        if not source.is_relative_to(repo):
            errors.append("Source outside repository: " + file.stem)
            continue
        sealed = {**descriptor["source_files"], **descriptor["provenance_files"]}
        for key in ("entrypoint", "preflight_entrypoint"):
            if descriptor.get(key) not in descriptor["source_files"]:
                errors.append("Unsealed entry point: " + file.stem + "/" + key)
        for relative, expected in sealed.items():
            target = (source / relative).resolve()
            if (not target.is_relative_to(source) or not target.is_file() or
                    hashlib.sha256(target.read_bytes()).hexdigest() != expected):
                errors.append("Source fingerprint mismatch: " + file.stem + "/" + relative)
    if errors:
        raise ValueError("\n".join(errors))
    print(f"Frozen scene baseline passed: {len(contract['sha256'])} immutable files; all release fingerprints checked")


if __name__ == "__main__":
    check()
