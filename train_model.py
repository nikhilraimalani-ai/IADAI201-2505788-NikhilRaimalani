"""Fine-tune YOLOv8n on the prepared bay dataset, then score it on the held-out test split.

Usage:
    python train_model.py                      # uses ./bay_data/data.yaml
    python train_model.py --data path/to/data.yaml --epochs 35

A GPU is strongly recommended (Colab T4 works: upload the project, `pip install ultralytics`, run).
On success the best weights are copied to ./bay_detector.pt, which app.py loads.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def absolute_yaml(yaml_path: Path) -> Path:
    """Ultralytics resolves 'path: .' oddly, so write a copy that points at the real folder."""
    fixed = yaml_path.with_name("data_abs.yaml")
    fixed.write_text(yaml_path.read_text().replace("path: .", f"path: {yaml_path.parent.resolve()}"))
    return fixed


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--data", type=Path, default=HERE / "bay_data" / "data.yaml")
    cli.add_argument("--epochs", type=int, default=35)
    cli.add_argument("--patience", type=int, default=8)
    cli.add_argument("--batch", type=int, default=16)
    cli.add_argument("--imgsz", type=int, default=640)
    opts = cli.parse_args()

    data_yaml = absolute_yaml(opts.data)

    # 1. transfer learning from COCO-pretrained weights
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=opts.epochs, patience=opts.patience,
        imgsz=opts.imgsz, batch=opts.batch,
        degrees=8, fliplr=0.5, hsv_v=0.35, hsv_s=0.5,          # augmentation
        project=str(HERE / "runs"), name="bay_detector_run", exist_ok=True,
    )

    # 2. evaluate the best checkpoint on the unseen test split
    best = HERE / "runs" / "bay_detector_run" / "weights" / "best.pt"
    tuned = YOLO(str(best))
    metrics = tuned.val(data=str(data_yaml), split="test")
    rows = [{"class": "ALL", "precision": metrics.box.mp, "recall": metrics.box.mr,
             "mAP50": metrics.box.map50, "mAP50-95": metrics.box.map}]
    for idx, name in tuned.names.items():
        p, r, m50, m = metrics.box.class_result(idx)
        rows.append({"class": name, "precision": p, "recall": r, "mAP50": m50, "mAP50-95": m})
    print("\nTest-set results:")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # 3. publish the weights where the app expects them
    shutil.copy(best, HERE / "bay_detector.pt")
    print(f"\nWeights saved to {HERE / 'bay_detector.pt'}")
    print(f"Confusion matrix and curves: {HERE / 'runs' / 'bay_detector_run'}")


if __name__ == "__main__":
    main()
