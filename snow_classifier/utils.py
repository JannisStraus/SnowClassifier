import os
from io import BytesIO
from pathlib import Path

from PIL import Image

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./data"))
IMAGE_DIR = OUTPUT_DIR / "images"
TRAIN_DIR = OUTPUT_DIR / "train"


def image2buffer(image: Image.Image) -> BytesIO:
    io_buf = BytesIO()
    image.save(io_buf, format="PNG")
    io_buf.seek(0)

    return io_buf
