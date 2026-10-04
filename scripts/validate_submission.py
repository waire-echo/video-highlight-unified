#!/usr/bin/env python3
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from video_highlight.validation import validate_submission


def main():
    parser = argparse.ArgumentParser(description="Validate AIC submit.jsonl format.")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--video-dir", default="")
    args = parser.parse_args()
    ok, errors = validate_submission(args.submission, args.index, args.video_dir)
    if ok:
        print("OK")
        return
    print("FAILED")
    for error in errors:
        print("-", error)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
