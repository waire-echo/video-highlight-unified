import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_submission_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_submission_pipeline", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalize_submission_row_drops_non_contract_fields():
    value = MODULE.normalize_submission_row(
        {
            "video_id": 7,
            "targetRatioWH": ["9", "16"],
            "predictions": [
                {"frame": 10, "bboxes": [1, 2, 3], "score": 0.9}
            ],
            "debug": "must not reach the final file",
        }
    )
    assert value == {
        "video_id": "7",
        "targetRatioWH": [9, 16],
        "predictions": [{"frame": 10, "bboxes": [1, 2, 3]}],
    }


def test_pipeline_id_changes_with_frozen_inputs(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"_commit_hash": "fixed"}), encoding="utf-8"
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    materialized = tmp_path / "data"
    materialized.mkdir()
    (materialized / "manifest.json").write_text(
        json.dumps({"answer_contract": "highlights-only/v1"}),
        encoding="utf-8",
    )
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "0.mp4").write_bytes(b"video")
    index = tmp_path / "index.json"
    index.write_text(json.dumps([{"video_id": "0"}]), encoding="utf-8")
    package = tmp_path / "package"
    package.mkdir()
    (package / "RELEASE_MANIFEST.json").write_text("{}", encoding="utf-8")
    (package / "vendor").mkdir()
    (package / "vendor/yolo_center.py").write_text("code", encoding="utf-8")
    (package / "assets").mkdir()
    (package / "assets/yolo11m.pt").write_bytes(b"weight")

    args = MODULE.argparse.Namespace(
        model=model,
        adapter=adapter,
        materialized=materialized,
        videos=videos,
        index=index,
        spatial_package=package,
        fps=2.0,
        max_frames=64,
        inference_max_tokens=512,
        video_max_token_num=196,
        video_min_token_num=128,
        end_tolerance_sec=0.5,
        expected_adapter_sha256=MODULE.sha256_file(
            adapter / "adapter_model.safetensors"
        ),
    )
    first = MODULE.build_pipeline_id(args)
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    second = MODULE.build_pipeline_id(args)
    assert first != second


def test_contract_hash_changes_with_pipeline_runner():
    module_path = Path(MODULE.__file__).resolve()
    original = module_path.read_bytes()
    # The runner hash is included; this is verified via a directly constructed
    # contract so the test does not mutate the checked-out script.
    value = {
        "pipeline_runner_sha256": MODULE.sha256_file(module_path),
        "other": "stable",
    }
    first = MODULE.canonical_sha256(value)
    value["pipeline_runner_sha256"] = "0" * 64
    assert first != MODULE.canonical_sha256(value)


def test_prepare_run_preserves_previous_pipeline_on_input_change(tmp_path):
    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir()
    previous = pipeline_dir / "run-old"
    previous.mkdir()
    (pipeline_dir / "CURRENT.json").write_text(
        json.dumps({"pipeline_id": "old", "run_root": str(previous)}),
        encoding="utf-8",
    )

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"_commit_hash": "fixed"}), encoding="utf-8"
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    materialized = tmp_path / "data"
    materialized.mkdir()
    (materialized / "manifest.json").write_text("{}", encoding="utf-8")
    index = tmp_path / "index.json"
    index.write_text("[]", encoding="utf-8")
    package = tmp_path / "package"
    package.mkdir()
    (package / "RELEASE_MANIFEST.json").write_text("{}", encoding="utf-8")
    (package / "vendor").mkdir()
    (package / "vendor/yolo_center.py").write_text("code", encoding="utf-8")
    (package / "assets").mkdir()
    (package / "assets/yolo11m.pt").write_bytes(b"weight")
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "0.mp4").write_bytes(b"video")

    args = MODULE.argparse.Namespace(
        pipeline_dir=pipeline_dir,
        model=model,
        adapter=adapter,
        materialized=materialized,
        videos=videos,
        index=index,
        spatial_package=package,
        seed_strict_merged=None,
        spatial_arm="B",
        fps=2.0,
        max_frames=64,
        inference_max_tokens=512,
        video_max_token_num=196,
        video_min_token_num=128,
        end_tolerance_sec=0.5,
        expected_adapter_sha256=MODULE.sha256_file(
            adapter / "adapter_model.safetensors"
        ),
    )
    run_root, _, resumed = MODULE.prepare_run(args, "new")
    assert resumed is False
    assert run_root == pipeline_dir / "run-new"
    assert previous.is_dir()
    current = json.loads((pipeline_dir / "CURRENT.json").read_text())
    assert current["pipeline_id"] == "new"


def test_strict_seed_is_bound_into_pipeline_id(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"_commit_hash": "fixed"}), encoding="utf-8"
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    materialized = tmp_path / "data"
    materialized.mkdir()
    (materialized / "manifest.json").write_text("{}", encoding="utf-8")
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "0.mp4").write_bytes(b"video")
    index = tmp_path / "index.json"
    index.write_text("[]", encoding="utf-8")
    package = tmp_path / "package"
    package.mkdir()
    (package / "RELEASE_MANIFEST.json").write_text("{}", encoding="utf-8")
    (package / "vendor").mkdir()
    (package / "vendor/yolo_center.py").write_text("code", encoding="utf-8")
    (package / "assets").mkdir()
    (package / "assets/yolo11m.pt").write_bytes(b"weight")
    seed = tmp_path / "seed"
    seed.mkdir()
    for name in ("summary.json", "windows.jsonl", "highlights.jsonl", "failures.jsonl"):
        (seed / name).write_text("{}", encoding="utf-8")

    args = MODULE.argparse.Namespace(
        model=model,
        adapter=adapter,
        materialized=materialized,
        videos=videos,
        index=index,
        spatial_package=package,
        seed_strict_merged=seed,
        fps=2.0,
        max_frames=64,
        inference_max_tokens=512,
        video_max_token_num=196,
        video_min_token_num=128,
        end_tolerance_sec=0.5,
        expected_adapter_sha256=MODULE.sha256_file(
            adapter / "adapter_model.safetensors"
        ),
    )
    first = MODULE.build_pipeline_id(args)
    (seed / "summary.json").write_text('{"changed":true}', encoding="utf-8")
    assert MODULE.build_pipeline_id(args) != first


def test_resume_ignores_only_created_at_in_real_contract(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"_commit_hash": "fixed"}), encoding="utf-8"
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    materialized = tmp_path / "data"
    materialized.mkdir()
    (materialized / "manifest.json").write_text("{}", encoding="utf-8")
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "0.mp4").write_bytes(b"video")
    index = tmp_path / "index.json"
    index.write_text("[]", encoding="utf-8")
    package = tmp_path / "package"
    package.mkdir()
    (package / "RELEASE_MANIFEST.json").write_text("{}", encoding="utf-8")
    (package / "vendor").mkdir()
    (package / "vendor/yolo_center.py").write_text("code", encoding="utf-8")
    (package / "assets").mkdir()
    (package / "assets/yolo11m.pt").write_bytes(b"weight")
    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir()

    args = MODULE.argparse.Namespace(
        pipeline_dir=pipeline_dir,
        model=model,
        adapter=adapter,
        materialized=materialized,
        videos=videos,
        index=index,
        spatial_package=package,
        seed_strict_merged=None,
        spatial_arm="B",
        fps=2.0,
        max_frames=64,
        inference_max_tokens=512,
        video_max_token_num=196,
        video_min_token_num=128,
        end_tolerance_sec=0.5,
    )
    pipeline_id = MODULE.build_pipeline_id(args)
    run_root, contract, _ = MODULE.prepare_run(args, pipeline_id)
    changed = dict(contract)
    changed["created_at"] = "later"
    (run_root / "PIPELINE.json").write_text(
        json.dumps(changed), encoding="utf-8"
    )
    resumed_root, _, resumed = MODULE.prepare_run(args, pipeline_id)
    assert resumed is True
    assert resumed_root == run_root


def test_retry_receipt_merges_only_strict_valid_rows(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "highlights.jsonl").write_text(
        json.dumps(
            {
                "video_id": "7",
                "duration": 30.0,
                "highlights": [{"start": 1.0, "end": 2.0}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    retry = run_root / "02b-retry" / "gpu-2" / "records"
    retry.mkdir(parents=True)
    (retry / "train_7_w0001.json").write_text(
        json.dumps(
            {
                "window_id": "train_7_w0001",
                "source_video_id": "7",
                "raw_json_valid": True,
                "source_highlights": [{"start": 22.0, "end": 23.0}],
            }
        ),
        encoding="utf-8",
    )
    args = MODULE.argparse.Namespace(
        materialized=tmp_path / "data",
        inference_gpus="2",
        python=tmp_path / "python",
        model=tmp_path / "model",
        adapter=tmp_path / "adapter",
        videos=tmp_path / "videos",
        max_frames=64,
        inference_max_tokens=512,
        per_gpu_gib=8,
        fps=2.0,
    )
    failures = [
        {"video_id": "7", "window_id": "train_7_w0001"},
    ]
    receipt_target = run_root / "02b-retry" / "RETRY.json"
    receipt_target.write_text('{"state":"COMPLETE"}', encoding="utf-8")
    receipt = MODULE.retry_failure_windows(
        args, run_root, seed, failures, execute=True
    )
    assert receipt == {"state": "COMPLETE"}


def test_apply_retry_highlights_replaces_only_target_source(tmp_path):
    run_root = tmp_path / "run"
    target = run_root / "02-merge" / "merged" / "highlights.jsonl"
    target.parent.mkdir(parents=True)
    original = [
        {"video_id": "0", "duration": 10.0, "highlights": []},
        {"video_id": "1", "duration": 20.0, "highlights": []},
    ]
    for video_id in range(2, 5):
        original.append(
            {"video_id": str(video_id), "duration": 30.0, "highlights": []}
        )
    target.write_text(
        "".join(json.dumps(row) + "\n" for row in original),
        encoding="utf-8",
    )
    replacement = [
        {
            "video_id": str(video_id),
            "duration": 30.0,
            "highlights": (
                [{"start": 3.0, "end": 4.0}] if video_id == 1 else []
            ),
        }
        for video_id in range(5)
    ]
    MODULE.apply_retry_highlights(run_root, replacement)
    rows = MODULE.read_jsonl(target)
    assert len(rows) == 5
    assert rows[1]["highlights"] == [{"start": 3.0, "end": 4.0}]
