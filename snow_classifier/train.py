from ultralytics import YOLO

from snow_classifier.utils import IMAGE_DIR, TRAIN_DIR


def train(epochs: int) -> None:
    # Initialize the model
    # https://docs.ultralytics.com/de/tasks
    model = YOLO("yolo11n-cls.pt")

    # Train the model
    # https://docs.ultralytics.com/de/modes/train/#train-settings
    model.train(
        data=IMAGE_DIR,
        epochs=epochs,
        imgsz=687,
        project=TRAIN_DIR,
        name="SnowClassifier",
        seed=42,
    )

    # Validate the model
    # https://docs.ultralytics.com/de/modes/val/#arguments-for-yolo-model-validation
    # metrics = model.val()


if __name__ == "__main__":
    train(25)
