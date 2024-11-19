import concurrent.futures
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from tqdm import tqdm

from snow_classifier.utils import IMAGE_DIR

logger = logging.getLogger("snow_classifier")
cam_id = 1996
prefix = f"https://api.panomax.com/1.0/cams/{cam_id}"


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
        return cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    else:
        logger.error(f"Unknown content type {content_type} for url {url}")
        return None


def get_timestamps(date: str) -> list[str]:
    day_url = f"{prefix}/images/day/{date}"
    day_json = fetch_data(day_url)
    timestamps = []
    if day_json is not None:
        for image_info in day_json["images"]:
            timestamp = str(image_info["time"])
            hour, minute, _ = map(int, timestamp.split(":"))
            if 8 <= hour < 16 or (hour == 16 and minute <= 30):
                timestamps.append(timestamp.replace(":", "-"))
    return timestamps


def _download_image(
    date: str,
    timestamp: str,
    server_idx: int,
    image_dir: Path = IMAGE_DIR,
    small: bool = True,
) -> Path | None:
    year, month, day = date.split("-")
    suffix = "small" if small else "default"
    url = f"https://panodata{server_idx}.panomax.com/cams/{cam_id}/{year}/{month}/{day}/{timestamp}_{suffix}.jpg"
    img = fetch_data(url)
    if img is None:
        return None
    output_path = image_dir / f"{date}_{timestamp}_{suffix}.jpg"
    cv2.imwrite(str(output_path), img)
    return output_path


def try_download_image(
    date: str, timestamp: str, image_dir: Path = IMAGE_DIR, small: bool = True
) -> Path:
    for server_idx in range(1, 16):
        image_path = _download_image(date, timestamp, server_idx, image_dir, small)
        if image_path is not None:
            return image_path
        logger.info(f"Retrying with server_idx={server_idx + 1}")
    raise TimeoutError("Maximum download attempts reached!")


def download_latest() -> Path | None:
    date = datetime.now().strftime("%Y-%m-%d")
    timestamps = get_timestamps(date)
    for day in range(15):
        date = (datetime.now() - timedelta(days=day)).strftime("%Y-%m-%d")
        timestamps = get_timestamps(date)
        if timestamps:
            break
    if not timestamps:
        return None
    selection = max(timestamps)
    return try_download_image(date, selection, image_dir=Path("."), small=False)


def process_day(date: str | None, random_time: bool = True) -> tuple[str, str] | None:
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    timestamps = get_timestamps(date)
    if not timestamps:
        logger.warning(f"No timestamps found for date {date}")
        return None
    if random_time:
        seed = ",".join(timestamps)
        rng = random.Random(seed)
        selection = rng.choice(timestamps)
    else:
        selection = max(timestamps)
    try:
        try_download_image(date, selection)
        return (date, selection)
    except TimeoutError:
        logger.error(f"Failed to download image for date {date}, timestamp {selection}")
        return None


def download_images(
    date_from: str, date_to: str, max_workers: int | None = None
) -> dict[str, str]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{prefix}/days?from={date_from}&to={date_to}"
    days_json: list[dict[str, Any]] = fetch_data(url)
    download_dict: dict[str, str] = {}

    # Use ProcessPoolExecutor for multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit parallel tasks for each day
        futures = {
            executor.submit(process_day, str(day_info["date"])): str(day_info["date"])
            for day_info in days_json
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(days_json),
            desc="Downloading images",
        ):
            result = future.result()
            if result is not None:
                date, selection = result
                download_dict[date] = selection
    return download_dict
