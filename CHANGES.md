# CurbSense Lite — refinement notes

This copy keeps the existing YOLOv8 detector and trained `bay_detector.pt` weights. The original repository is not modified.

## Interface and workflow
- Reworked the page into a dark, responsive parking operations dashboard.
- Added separate Image analysis, Video analysis, Session history, and How it works sections.
- Added confidence, overlap/NMS, and model image-size controls.
- Added KPI cards, colour-coded statuses, parking recommendations, and a bay distribution chart.

## Image functionality
- Added WEBP support and safer image loading.
- Shows the source image alongside the annotated result.
- Exports annotated PNG files and per-bay CSV reports.

## Video functionality
- Added MP4, MOV, AVI, MKV, and M4V uploads.
- Samples video frames at a configurable interval, with a maximum sampled-frame limit.
- Exports an annotated MP4 and a frame-summary CSV.
- Displays average and peak sampled occupancy and an occupancy-over-time chart.

## Reliability and interpretation
- Looks for the root `bay_detector.pt`, then falls back to `runs/bay_detector_run/weights/best.pt`.
- Handles unrecognised model class names separately instead of silently counting them as occupied.
- Adds upload-size checks and clear error messages.
- Explains that video analysis is sampled-frame analysis, not live CCTV or object tracking.

## Validation performed
- Python syntax compilation passed.
- Helper smoke tests passed for `bay-free`, `bay-taken`, and unknown class handling.
- A synthetic 20-frame video smoke test passed: the app created a playable MP4 output and a 10-row sampled-frame report.

The trained model itself was not retrained or changed. The real parking-lot detection quality still depends on the included checkpoint and the input footage.
