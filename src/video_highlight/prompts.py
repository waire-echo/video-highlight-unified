"""Load immutable, versioned prompt assets shared by every pipeline."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] / "prompts"


def bundle_path(bundle: str) -> Path:
    overrides = json.loads(os.environ.get("VIDEO_HIGHLIGHT_PROMPT_OVERRIDES", "{}"))
    bundle = overrides.get(bundle, bundle)
    root = Path(os.environ.get("VIDEO_HIGHLIGHT_PROMPT_ROOT", ROOT)).resolve()
    path = (root / bundle).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError("Unknown prompt bundle: " + bundle)
    return path


def load_text(bundle: str, name: str) -> str:
    root = bundle_path(bundle)
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Prompt path leaves its bundle")
    return path.read_text(encoding="utf-8")


def call_builder(bundle: str, name: str, *args, **kwargs):
    path = bundle_path(bundle) / "builders.py"
    identity = "_vh_prompt_" + hashlib.sha256(path.read_bytes()).hexdigest()
    if identity not in sys.modules:
        spec = importlib.util.spec_from_file_location(identity, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[identity] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(identity, None)
            raise
    return getattr(sys.modules[identity], name)(*args, **kwargs)


def bundle_hashes(bundles: dict[str, str]) -> dict:
    result = {}
    for alias, bundle in bundles.items():
        root = ROOT / bundle
        if not root.is_dir():
            raise ValueError("Unknown prompt version: " + bundle)
        result[alias] = {"version": bundle, "files": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts}}
    return result
