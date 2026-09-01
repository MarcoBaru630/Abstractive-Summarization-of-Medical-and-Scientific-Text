from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
# Create paths like this BASE_DIR / 'subdir'

MODEL_NAME = "facebook/bart-base"
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 256
SEED = 42
DEBUG = False
CACHED = True
