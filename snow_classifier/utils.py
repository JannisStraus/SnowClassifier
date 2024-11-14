import os
from pathlib import Path

OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
IMAGE_DIR = OUTPUT_DIR / "images"
TRAIN_DIR = OUTPUT_DIR / "train"
