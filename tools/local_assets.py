#!/usr/bin/env python3
"""Find moved local assets and verify their retained file fingerprints."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from video_highlight.experiments import read_json, sha256

AUDIT = REPO / "artifacts/workspace_cleanup/20261004/operations.local.json"
INDEX = REPO / "references/assets.local.json"


def redirect(path, audit):
    moves = sorted(audit["path_map"], key=lambda row: len(row["source"]), reverse=True)
    duplicates = {Path(row["path"]): Path(row["canonical_path"])
                  for row in audit["deleted"] if row.get("canonical_path")}
    for _ in range(len(moves) + len(duplicates) + 1):
        previous = path
        for row in moves:
            source = Path(row["source"])
            if path.is_relative_to(source):
                path = Path(row["destination"]) / path.relative_to(source)
                break
        path = duplicates.get(path, path)
        if path == previous:
            return path
    raise ValueError("Cycle in the local asset path map")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--kind")
    finding = sub.add_parser("find")
    finding.add_argument("--path", required=True, help="Original path relative to the outer workspace, or current repo path")
    sub.add_parser("verify")
    args = parser.parse_args()
    if not INDEX.exists() or not AUDIT.exists():
        parser.error("Local asset metadata is not installed on this machine. See references/README.md.")
    assets = read_json(INDEX)["files"]
    audit = read_json(AUDIT)
    if args.action == "list":
        for row in assets:
            if args.kind is None or row["kind"] == args.kind:
                print(json.dumps({key: row[key] for key in ("path", "kind", "sha256")}, ensure_ascii=False))
        return
    if args.action == "find":
        path = Path(args.path)
        if not path.is_absolute():
            current = REPO / path
            path = current if current.exists() else REPO.parent / path
        path = redirect(path.resolve(), audit)
        matches = [row for row in assets if (REPO / row["path"]) == path]
        expected = matches[0]["sha256"] if matches else None
        exists = path.is_file()
        print(json.dumps({"path": str(path), "exists": exists, "sha256": expected,
                          "hash_matches": sha256(path) == expected if exists and expected else None}, ensure_ascii=False))
        if not exists:
            raise SystemExit(1)
        return
    failures = []
    for row in assets:
        path = REPO / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            failures.append(row["path"])
    duplicate_count = 0
    for row in audit["deleted"]:
        if row.get("canonical_path"):
            path = redirect(Path(row["canonical_path"]), audit)
            duplicate_count += 1
            if not path.is_file() or sha256(path) != row["sha256"]:
                failures.append("Duplicate keeper: " + str(path))
    print(json.dumps({"retained_files": len(assets), "duplicate_keepers": duplicate_count,
                      "failures": failures}, ensure_ascii=False))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
