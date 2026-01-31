# configs/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "wikitext-2-raw-v1"
PROCESSED_DATA_DIR = DATA_DIR / "wikitext-2-v1"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# create dirs if not exist
for d in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    RESULTS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
