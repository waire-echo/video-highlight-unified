"""The normal CI suite enforces the maintained route and entry-document contract."""
import importlib.util
from pathlib import Path


def test_project_defaults_configs_and_documentation_agree():
    path = Path(__file__).resolve().parents[1] / "tools/check_project_policy.py"
    spec = importlib.util.spec_from_file_location("project_policy_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check()
