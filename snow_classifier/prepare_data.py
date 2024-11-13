from datetime import datetime
from pathlib import Path

from snow_classifier.utils import IMAGE_DIR


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


def prepare() -> None:
    snow_dates = [
        ("2023-11-03", "2023-11-08"),
        ("2023-11-11", "2023-11-13"),
        ("2023-11-17", "2024-02-17"),
    ]
    grass_dates = [("2024-04-25", "2024-07-07")]
    snow_dir = IMAGE_DIR / "snow"
    grass_dir = IMAGE_DIR / "grass"
    snow_dir.mkdir(exist_ok=True)
    grass_dir.mkdir(exist_ok=True)

    all_images = sorted(IMAGE_DIR.glob("*.jpg"))

    # Filter images for snow and grass date ranges
    snow_images = filter_images_by_date(all_images, snow_dates)
    grass_images = filter_images_by_date(all_images, grass_dates)

    for image in snow_images:
        image.rename(snow_dir / image.name)

    for image in grass_images:
        image.rename(grass_dir / image.name)

    for image in IMAGE_DIR.glob("*.jpg"):
        image.unlink()


if __name__ == "__main__":
    prepare()
