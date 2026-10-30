# CurbSense Lite

**CurbSense Lite** is an AI-powered parking analysis application that uses a YOLOv8 model to identify parking bays in uploaded images and video clips. It classifies detected bays as **Free**, **Occupied**, or **Unclassified**, then turns those detections into a clear summary of parking availability.

## What the app does

CurbSense Lite provides two ways to analyse a parking area:

- **Image analysis:** Upload a JPG, JPEG, PNG, or WEBP image to detect and classify visible parking bays. The app displays the annotated result alongside parking statistics and detection details.
- **Video analysis:** Upload an MP4, MOV, AVI, MKV, or M4V clip. The app samples frames from the clip, annotates detected bays, and summarises how the estimated occupancy changes across the analysed frames.

## Key features

- **Automated bay detection:** Uses the trained YOLOv8 parking-bay detector to locate visible bays.
- **Colour-coded results:** Free bays are shown in teal/green, occupied bays in orange, and unclassified detections in purple.
- **Occupancy dashboard:** Summarises detected bays, available bays, occupied bays, and occupancy percentage.
- **Detection controls:** Lets users adjust confidence threshold, overlap/NMS threshold, and model image size.
- **Parking insights:** Provides a simple crowding rating and availability recommendation based on the classified detections.
- **Visual analytics:** Presents the distribution of free, occupied, and unclassified bays, along with individual detection details and confidence scores.
- **Exportable results:** Supports downloads of annotated images, processed video, image detection data, and video frame-summary reports in CSV format.
- **Session history:** Keeps a record of recent analyses during the current app session.

## How occupancy is calculated

The occupancy percentage is calculated from bays that the model classifies as either Free or Occupied:

```text
Occupancy (%) = Occupied bays / (Free bays + Occupied bays) × 100
```

Unclassified detections are displayed separately and are not included in the occupancy calculation.

The app uses the following crowding labels:

| Occupancy | Status |
| --- | --- |
| Below 40% | Light |
| 40% to 75% | Steady |
| Above 75% | Heavy |

The recommendation also considers the number of detected free bays. For example, it may suggest checking another parking area when no free bays are detected.

## Video analysis

Video analysis is based on **sampled frames**, rather than continuous object tracking across every frame. Each analysed frame produces an occupancy estimate and timestamp. The frame summaries are displayed in the app and can be exported as a CSV report.

Because the estimates come from sampled frames, they provide an overview of the analysed clip—not a guaranteed record of every vehicle movement or a live count of the parking lot.

## Understanding the results

CurbSense Lite reports what its model detects in the uploaded media. Results can be affected by camera angle, image quality, lighting, occlusion, distance, and the visibility of parking-bay markings. A detected bay that cannot be mapped to a recognised Free or Occupied class is labelled Unclassified and excluded from the occupancy percentage.

The application analyses uploaded media only. It does not connect to live CCTV feeds, maintain a live parking database, reserve spaces, or guarantee that a displayed space remains available after the media was captured.

## Technology

- **Python** — application logic
- **Streamlit** — interactive application interface
- **Ultralytics YOLOv8** — parking-bay object detection
- **OpenCV** — image and video processing
- **Pandas and NumPy** — detection summaries and analysis

## Credits

The project uses the PKLot parking-lot dataset referenced in the original project and the Ultralytics YOLOv8 framework. Dataset attribution and licensing should be retained with the project materials.
