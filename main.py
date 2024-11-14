from snow_classifier.prepare_data import prepare_images
from snow_classifier.webscraper import download_images, download_latest
from snow_classifier.yolo import inference, train


def train_model() -> None:
    download_images("2023-11-01", "2024-11-14")
    prepare_images()
    train(10)


def run_model() -> str:
    image_dir = download_latest()

    if image_dir is None:
        raise ValueError()

    predicion = inference(image_dir)
    image_dir.unlink()
    return predicion


if __name__ == "__main__":
    # train_model()
    run_model()
