from snow_classifier.prepare_data import prepare
from snow_classifier.webscraper import download_images


def run() -> None:
    download_images("2023-11-01", "2024-11-12")
    prepare()


if __name__ == "__main__":
    run()
