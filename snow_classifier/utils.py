import os
from io import BytesIO
from pathlib import Path

import cv2
from cv2.typing import MatLike

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./data"))
IMAGE_DIR = OUTPUT_DIR / "images"
TRAIN_DIR = OUTPUT_DIR / "train"


def cv2_to_buffer(image: MatLike) -> BytesIO:
    _, buffer = cv2.imencode(".png", image)
    io_buf = BytesIO(buffer)
    io_buf.seek(0)

    return io_buf
