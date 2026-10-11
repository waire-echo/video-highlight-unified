#!/usr/bin/env python3
"""Check the maintained workflow, default version and public documentation links."""
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]


def check(repo=REPO):
    repo = Path(repo).resolve()
    policy = json.loads((repo / "configs/project_policy.json").read_text())
    errors = []
    if policy.get("schema_version") != "video-highlight-project/v1" or policy.get("maintained_branch") != "main":
        errors.append("Project must declare main as its maintained branch")
    if policy.get("maintained_workflow") != "scene_general_v3":
        errors.append("Unexpected maintained workflow")
    if policy.get("default_version") != "scene-general-v3-compat-20261010":
        errors.append("Default must retain the frozen compatible baseline")
    for name, expected in policy["workflow_configs"].items():
        config = json.loads((repo / name).read_text())
        for key in ("version", "input_mode"):
            if config.get(key) != expected[key]:
                errors.append(f"Workflow config disagrees with policy: {name}/{key}")
        release_path = repo / "configs/workflows/versions" / (expected["version"] + ".json")
        descriptor = json.loads(release_path.read_text())
        if expected["input_mode"] not in descriptor["input_modes"]:
            errors.append("Unsupported input mode: " + name)
    default = json.loads((repo / policy["default_config"]).read_text())
    if (default["version"] != policy["default_version"] or default["input_mode"] != "baseline" or
            policy["workflow_configs"][policy["default_config"]]["role"] != "default-compat"):
        errors.append("Default configuration must use the frozen baseline")
    descriptor = json.loads((repo / "configs/workflows/versions" / (policy["default_version"] + ".json")).read_text())
    if descriptor["mode"] != "compat" or descriptor["io_profile"] != "original":
        errors.append("Default must preserve original I/O")
    if "pending" in descriptor["gpu_validation"].lower() and policy["gpu_validation"] != "pending":
        errors.append("Project cannot claim GPU validation before its default release")
    if not (repo / policy["frozen_contracts"]).is_file():
        errors.append("Missing independent frozen contracts")
    for name in policy["entry_documents"]:
        file = repo / name
        if not file.is_file():
            errors.append("Missing entry document: " + name)
            continue
        text = file.read_text()
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
            target = target.split("#", 1)[0].strip("<>")
            if not target or re.match(r"[a-zA-Z]+:", target):
                continue
            resolved = (file.parent / target).resolve()
            if not resolved.is_relative_to(repo) or not resolved.exists():
                errors.append(f"Broken documentation link: {name} -> {target}")
    if errors:
        raise ValueError("\n".join(errors))
    print(f"Project policy passed: main, compatible default, {len(policy['workflow_configs'])} configurations and {len(policy['entry_documents'])} entry documents")


if __name__ == "__main__":
    check()
