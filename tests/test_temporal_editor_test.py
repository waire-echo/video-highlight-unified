"""Run isolated pilot contracts without importing its generic module names into pytest."""
import subprocess
import sys
from pathlib import Path


def test_temporal_pilot_contracts():
    folder=Path(__file__).resolve().parents[1]/'pipelines/ora_temporal_editor_test'
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(folder),'-p','test_editor.py'],check=True)
