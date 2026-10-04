"""Route shortcut; experiment logic stays in the shared runner."""
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
from video_highlight.runner import main
if __name__ == "__main__":
    raise SystemExit(main(["run", "--config", str(root / "configs/pipelines/reze_yesno_v1.json"), *sys.argv[1:]]))
