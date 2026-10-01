"""Call an original Stage-2 CLI with only ordered frame prefetch added."""
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'spatial'))
from stage2_prefetch import install

script = sys.argv.pop(1)
module = importlib.import_module(Path(script).stem)
install(module)
module.generate(module.parse_args())
