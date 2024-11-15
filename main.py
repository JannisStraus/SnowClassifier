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
        ("2023-04-15", "2023-11-02"),
        ("2024-04-26", "2024-11-13"),
    ]

    download_images("2022-11-14", "2024-11-14")
    prepare_images(snow_dates, grass_dates)
    train(10)


def run_model() -> str:
    image_dir = download_latest()

    if image_dir is None:
        raise ValueError()

    predicion = infer(image_dir)
    image_dir.unlink()
    return predicion


if __name__ == "__main__":
    train_model()
    run_model()
