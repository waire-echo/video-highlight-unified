#!/usr/bin/env python3
"""Retry only unsupported transfer metadata; preserve source pixels/timestamps.

Install as TASK_ROOT/env/bin/ffmpeg (a symlink to this file). The actual binary
remains TASK_ROOT/media/bin/ffmpeg. Every exceptional retry is journaled.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parent.parent
    binary = root / 'media' / 'bin' / 'ffmpeg'
    arguments = sys.argv[1:]
    result = subprocess.run([str(binary), *arguments], stderr=subprocess.PIPE)
    message = result.stderr.decode('utf-8', errors='replace')
    if (result.returncode and 'Unsupported input' in message
            and 'trc:log316' in message and '-vf' in arguments):
        retry = list(arguments)
        index = retry.index('-vf') + 1
        retry[index] = 'setparams=color_trc=unknown,' + retry[index]
        journal = root / 'run_v5' / 'media_compatibility.jsonl'
        journal.parent.mkdir(parents=True, exist_ok=True)
        entry = {'time_unix': time.time(), 'original_arguments': arguments,
                 'retry_arguments': retry, 'original_stderr': message,
                 'action': 'Treat unsupported log316 transfer metadata as unknown before scaling; no source file is changed.',
                 'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        result = subprocess.run([str(binary), *retry], stderr=subprocess.PIPE)
        entry['retry_returncode'] = result.returncode
        with journal.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    sys.stderr.buffer.write(result.stderr)
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
