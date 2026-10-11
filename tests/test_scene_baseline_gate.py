"""Keep baseline and native regressions in the regular automatic CPU suite."""
import subprocess
import sys

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'scene_baseline_gate', Path(__file__).resolve().parents[1] / 'tools/check_scene_baseline.py'
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
REPO, check = gate.REPO, gate.check


def test_frozen_scene_core_prompts_and_registered_versions():
    check()


def test_original_scene_v3_regression_suite():
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s",
         str(REPO / "pipelines/ora_spatial/experiments/scene_general_v3"),
         "-p", "test_*.py", "-v"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Ran 10 tests" in result.stderr
