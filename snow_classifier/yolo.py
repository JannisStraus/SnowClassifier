from pathlib import Path

import cv2
from ultralytics import YOLO

from snow_classifier.utils import IMAGE_DIR, TRAIN_DIR

# v8n performs better than v11n
model_name = "SnowClassifier1v8"


def inference(file: str | Path, save_img: bool = True) -> str:
    model_path = TRAIN_DIR / model_name / "weights" / "best.pt"
    model = YOLO(model=model_path, task="classify", verbose=False)
    labels = dict(enumerate(["grass", "snow"]))

    image = cv2.imread(str(file))
    results = model([image], imgsz=384, device="cpu", verbose=False)

    assert len(results) == 1
    result = results[0]

    output_dict = {
        label: {"confidence": float(result.probs.data[i])}
        for i, label in labels.items()
    }

    max_label = max(
        output_dict.items(),
        key=lambda item: item[1]["confidence"],
    )[0]

    if save_img:
        result.save(filename="./output.png")

    return max_label


def train(epochs: int) -> None:
    # Initialize the model
    # https://docs.ultralytics.com/de/tasks
    model = YOLO("yolov8n-cls.pt")  # yolo11n-cls.pt

    # Train the model
    # https://docs.ultralytics.com/de/modes/train/#train-settings
    model.train(
        data=IMAGE_DIR,
        epochs=epochs,
        imgsz=384,
        project=TRAIN_DIR,
        name=model_name,
        seed=42,
    )

    # Validate the model
    # https://docs.ultralytics.com/de/modes/val/#arguments-for-yolo-model-validation
    # metrics = model.val()

    # Remove .pt files
    for m in Path("./").glob("*.pt"):
        m.unlink()
