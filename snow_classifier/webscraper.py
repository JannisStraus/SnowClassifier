import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from tqdm import tqdm

from snow_classifier.utils import IMAGE_DIR

logger = logging.getLogger(__name__)
rng = random.Random(42)
cam_id = 1996
prefix = f"https://api.panomax.com/1.0/cams/{cam_id}"


def download_latest() -> Path | None:
    date = datetime.now().strftime("%Y-%m-%d")
    timestamps = get_timestamps(date)
    if len(timestamps) == 0:
        return None
    selection = sorted(timestamps)[-1]
    server_idx = 1

    image_path = download_image(
        date, selection, server_idx, image_dir="./", small=False
    )
    while image_path is None:
        if server_idx == 15:
            raise TimeoutError("Maximum download attempts reached!")
        server_idx += 1
        logger.info(f"Retrying with {server_idx = }")
        image_path = download_image(
            date, selection, server_idx, image_dir="./", small=False
        )
    return image_path


def download_images(date_from: str, date_to: str) -> dict[str, str]:
    IMAGE_DIR.mkdir(exist_ok=True)
    url = f"{prefix}/days?from={date_from}&to={date_to}"
    server_idx = 1

    days_json: list[dict[str, Any]] = fetch_data(url)
    download_dict: dict[str, str] = {}
    for days in tqdm(days_json):
        date = str(days["date"])
        timestamps = get_timestamps(date)
        selection = rng.choice(timestamps)
        download_dict[date] = selection

        image_path = download_image(date, selection, server_idx)
        while image_path is None:
            if server_idx == 15:
                raise TimeoutError("Maximum download attempts reached!")
            server_idx += 1
            logger.info(f"Retrying with {server_idx = }")
            image_path = download_image(date, selection, server_idx)

    return download_dict


def download_image(
    date: str,
    timestamp: str,
    server_idx: int,
    image_dir: str | Path = IMAGE_DIR,
    small: bool = True,
) -> Path | None:
    image_dir = Path(image_dir)
    d = date.split("-")
    suffix = "small" if small else "default"
    url = f"https://panodata{server_idx}.panomax.com/cams/{cam_id}/{d[0]}/{d[1]}/{d[2]}/{timestamp}_{suffix}.jpg"

    img = fetch_data(url)
    if img is None:
        return None

    output_dir = image_dir / f"{date}_{timestamp}.jpg"
    cv2.imwrite(str(output_dir), img)
    return output_dir


def get_timestamps(date: str) -> list[str]:
    day_url = f"{prefix}/images/day/{date}"
    day_json = fetch_data(day_url)
    timestamps = []
    if day_json is not None:
        for day in day_json["images"]:
            timestamp = str(day["time"])
            splitted = timestamp.split(":")
            first, second = int(splitted[0]), int(splitted[1])
            if first >= 8 and (first < 16 or (first == 16 and second <= 30)):
                timestamps.append(timestamp.replace(":", "-"))
    return timestamps


def fetch_data(url: str) -> Any:
    response = requests.get(url)

    if response.status_code != 200:
        logger.error(f"Failed to fetch data: {response.status_code}: {url}")
        return None

    content_type = response.headers.get("Content-Type", "")

    if content_type == "application/json":
        return response.json()
    elif "image" in content_type:
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        return image
