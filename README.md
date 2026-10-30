# CurbSense Lite

CurbSense Lite is a Streamlit application that uses a trained YOLOv8 parking-bay detector to analyse uploaded parking-lot images and video clips. It classifies detected bays as **free** or **occupied**, displays occupancy insights, and exports annotated media and CSV reports.

## Features

- **Image analysis:** upload JPG, JPEG, PNG or WEBP images; view the original and annotated result side by side.
- **Video analysis:** upload MP4, MOV, AVI, MKV or M4V clips; sample frames, annotate the sampled results, and review occupancy estimates over time.
- **Occupancy dashboard:** detected-bay totals, available and occupied counts, occupancy percentage, and a crowding rating.
- **Detection controls:** adjust confidence, overlap/NMS threshold, and inference image size.
- **Exports:** download annotated images, processed videos, image-level detection CSVs, and video frame-summary CSVs.
- **Session history:** review the analyses completed in the current Streamlit session.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface, image/video inference, analytics and exports |
| `bay_detector.pt` | Trained parking-bay detection weights used by the app |
| `yolov8n.pt` | Base YOLOv8n weights used for training |
| `prepare_data.py` | Converts the PKLot Roboflow COCO export into a YOLO dataset |
| `train_model.py` | Training and held-out test evaluation script |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System packages used by Streamlit Community Cloud |
| `runs/` | Training plots, logs, and saved checkpoints |
| `.streamlit/config.toml` | Dark theme and upload-size configuration |

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app looks for `bay_detector.pt` beside `app.py`. If that file is absent, it also checks `runs/bay_detector_run/weights/best.pt`.

## Deploy with Streamlit Community Cloud

1. Create your own GitHub copy of this project and include `app.py`, `requirements.txt`, `.streamlit/config.toml`, and the trained `bay_detector.pt` weights.
2. In Streamlit Community Cloud, create an app from your repository and select `app.py` as the main file.
3. Deploy. The trained weights must be present in the repository or otherwise made available to the app at runtime.

The included model file is approximately 6.5 MB. Check the hosting provider's repository and upload limits before deployment.

## Dataset preparation and training

The training scripts use a Roboflow PKLot COCO-format export. Put the raw `train/`, `valid/`, and `test/` folders in the location passed to `prepare_data.py`, then run:

```bash
python prepare_data.py --raw-dir ./data --out-dir ./bay_data --sample-size 1500
python train_model.py --data ./bay_data/data.yaml --epochs 35
```

`train_model.py` evaluates the selected checkpoint using the dataset's test split and copies the best checkpoint to `bay_detector.pt`. A GPU is strongly recommended for training; the Streamlit app itself can run inference on CPU, though processing may be slower.

## Interpreting results

The app calculates occupancy as:

`occupied detected bays / (free detected bays + occupied detected bays) × 100`

Detections with unrecognised class names are reported separately and excluded from this calculation. The crowding labels follow the original project thresholds:

| Occupancy | Rating |
|---|---|
| Under 40% | Light |
| 40% to 75% | Steady |
| Over 75% | Heavy |

Video analysis samples frames rather than running object tracking on every frame. The exported video uses a reduced frame rate between sampled frames to keep its duration close to the source without running inference on every frame. The video report records the sampled timestamps and occupancy estimate. It is not a live CCTV or reservation system.

## Model evaluation

The repository contains training logs and validation plots under `runs/bay_detector_run/`. The final held-out test metrics should be reported from the output of `train_model.py`; no test-set score is claimed here.

## Credits

Dataset: PKLot, Almeida et al. (2015), Universidade Federal do Paraná, CC BY 4.0, as referenced in the original project. Detection framework: Ultralytics YOLOv8. App: Streamlit and OpenCV.
