# WITROV Real-Time Measurement System

Real-time underwater object measurement using two red laser dots as a fixed reference, built with OpenCV and Python for MATE ROV competition.

## What it does
- Detects two red laser dots using HSV color masking
- Calculates pixels-per-cm ratio from known laser separation distance
- Detects white objects (PVC pipes) using HSV masking
- Measures object width and height in real-world cm
- Displays dimensions as live overlay on video feed

## Requirements
Install dependencies by running:

    pip install opencv-python numpy

## How to run

    python Real_Time_Measurement.py

## Things to change before running
- **Line 7** — camera index: `0` = built-in webcam, `1` = external/ROV camera
- **Line 52** — `laserDistanceCm` = your actual laser separation in cm (currently 3.65)

## How it works
1. Camera captures live video frame
2. Frame converted to HSV color space
3. Red laser dots detected using HSV masking + circularity filter
4. Pixel distance between dots calculated using Euclidean distance
5. Pixels-per-cm ratio computed from known laser distance
6. White object detected using HSV masking + morphological cleanup
7. Object dimensions measured using minAreaRect
8. Width and height displayed on frame in centimeters

## Tuning guide
| Parameter | Location | What to change |
|---|---|---|
| Camera index | Line 7 | 0 or 1 |
| Laser distance | Line 52 | your actual cm measurement |
| Red HSV range | Lines 16-21 | if dots not detected |
| White HSV range | Lines 63-64 | if object not detected |
| Area threshold | Line 76 | increase to reduce noise |
| Kernel size | Line 12 | increase to remove larger noise |

## Built with
- Python 3
- OpenCV 4
- NumPy

## Acknowledgements
Built with guidance from Claude (Anthropic) as a learning project for WITROV ROV team.
