from typing import Any

from snow_classifier.fastvit import infer, train
from snow_classifier.prepare_data import prepare_images
from snow_classifier.webscraper import download_images, download_latest


def train_model() -> None:
    snow_dates = [
        ("2022-11-18", "2023-03-21"),
        ("2023-11-03", "2024-04-05"),
    ]
    grass_dates = [
        ("2022-11-14", "2022-11-17"),
        ("2023-04-17", "2023-11-02"),
        ("2024-04-26", "2024-11-13"),
        ("2024-11-14", "2024-11-16"),
    ]
    date_dict = {
        "snow": snow_dates,
        "grass": grass_dates,
    }

    download_images("2022-11-14", "2024-11-16")
    prepare_images(date_dict)
    train(5)


def run_model() -> dict[str, Any]:
    image_dir = download_latest()

    if image_dir is None:
        raise ValueError()

    predicion = infer(image_dir)[image_dir.name]
    datetime = image_dir.name[:16]
    date, time = datetime[:10], datetime[11:16].replace("-", ":")
    predicion["date"], predicion["time"] = date, time

    image_dir.unlink()

    return predicion


if __name__ == "__main__":
    train_model()
    run_model()
