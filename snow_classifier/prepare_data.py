import random
from datetime import datetime
from pathlib import Path

from snow_classifier.utils import IMAGE_DIR, TRAIN_DIR

rng = random.Random(42)


def filter_images_by_date(
    images: list[Path], date_ranges: list[tuple[str, str]]
) -> list[Path]:
    # Convert date ranges to datetime objects for comparison
    parsed_ranges = [
        (datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d"))
        for start, end in date_ranges
    ]

    filtered_images = []
    for img in images:
        # Extract date from filename
        img_date_str = img.stem.split("_")[0]  # Get YYYY-MM-DD from YYYY-MM-DD_*.jpg
        img_date = datetime.strptime(img_date_str, "%Y-%m-%d")

        # Check if the image date falls within any of the specified ranges
        if any(start <= img_date <= end for start, end in parsed_ranges):
            filtered_images.append(img)

    return filtered_images


def prepare_images(
    snow_dates: list[tuple[str, str]], grass_dates: list[tuple[str, str]]
) -> None:
    TRAIN_DIR.mkdir(exist_ok=True)
    all_images = sorted(IMAGE_DIR.glob("*.jpg"))

    # Filter images for snow and grass date ranges
    snow_images = filter_images_by_date(all_images, snow_dates)
    grass_images = filter_images_by_date(all_images, grass_dates)

    for images, category in [(snow_images, "snow"), (grass_images, "grass")]:
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        train_dir = IMAGE_DIR / "train" / category
        val_dir = IMAGE_DIR / "val" / category
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for image in train_images:
            image.rename(train_dir / image.name)

        for image in val_images:
            image.rename(val_dir / image.name)

    for image in IMAGE_DIR.glob("*.jpg"):
        image.unlink()
