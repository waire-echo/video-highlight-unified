"""Load the sealed algorithm once; apply only the selected release's I/O policy."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from video_highlight.workflow.contracts import write_json
from video_highlight.workflow.io_optimized import CheckpointExporter, SequentialFrameReader


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--entrypoint", default="experiments/scene_general_v3/temporal.py")
    parser.add_argument("--io-profile", choices=("original", "sequential-checkpoints"), required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    path = (args.source_root / args.entrypoint).resolve()
    if not path.is_relative_to(args.source_root.resolve()):
        raise ValueError("Entry point must be inside sealed sources")
    spec = importlib.util.spec_from_file_location("sealed_scene_v3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reader = exporter = None
    if args.io_profile == "sequential-checkpoints":
        import common
        reader = SequentialFrameReader()
        exporter = CheckpointExporter(common, args.checkpoint_interval)
        module.read_frames = reader
        module.export = exporter
    sys.argv = [str(path), "--ids", args.ids, "--video-dir", args.video_dir, "--output", args.output]
    try:
        module.main()
    finally:
        if reader is not None:
            reader.close()
            write_json(Path(args.output) / "performance.json",
                       {"io_profile": args.io_profile, "reader": reader.metrics, "export": exporter.metrics,
                        "model_request_order_changed": False, "gpu_equivalence_verified": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
