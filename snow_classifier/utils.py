import os
from pathlib import Path

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./data"))
IMAGE_DIR = OUTPUT_DIR / "images"
TRAIN_DIR = OUTPUT_DIR / "train"
