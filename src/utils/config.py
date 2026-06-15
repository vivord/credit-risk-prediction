import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_RAW = BASE_DIR / "data/raw"
ARTIFACTS = BASE_DIR / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

TARGET = "Risk"  # Good=1 (low risk), Bad=2 (high risk) in German dataset
RANDOM_STATE = 42
TEST_SIZE = 0.2