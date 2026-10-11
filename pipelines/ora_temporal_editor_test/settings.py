"""Small test configuration; machine paths are explicit CLI arguments."""
import subprocess
from pathlib import Path
from engine import read

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / 'configs/experiments/ora-temporal-editor-test-v3.json'


def data_arguments(parser):
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Existing inputs/manifest.json, inputs/timestamps.npz and output/parameter_ledger.json')
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)


def git_commit():
    # Standalone server snapshots carry the same commit in a deployment receipt.
    receipt = REPO / 'deployment.local.json'
    if receipt.exists():
        return read(receipt)['git_commit']
    return subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
