from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent.parent

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
LOGS_DIR = ROOT_DIR / "logs"
RESULTS_DIR = ROOT_DIR / "results"