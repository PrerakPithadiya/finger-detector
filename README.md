# Finger Detector

A Python application that uses computer vision to detect and count fingers in real-time using a webcam.

## Features

- Real-time finger detection and counting
- Works with one or two hands simultaneously
- Displays finger count with visual feedback
- Elegant UI with status messages and timestamps
- Robust finger detection algorithm that works in various lighting conditions

## Project Structure

```
finger-detector/
├── .gitignore           # Git ignore file
├── README.md            # Project documentation
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
├── setup.py             # Package installation script
└── src/                 # Source code directory
    ├── __init__.py      # Package initialization
    └── finger_detector.py  # Core finger detection logic
```

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- Webcam

## Installation

### Method 1: Using pip

```bash
# Clone the repository
git clone https://github.com/PrerakPithadiya/finger-detector.git
cd finger-detector

# Install the package and dependencies
pip install -e .
```

### Method 2: Using requirements.txt

```bash
# Clone the repository
git clone https://github.com/PrerakPithadiya/finger-detector.git
cd finger-detector

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python main.py
```

- Show your hand(s) to the camera
- The application will detect and count your fingers in real-time
- Press 'q' to exit the application

## How it Works

This application uses MediaPipe's hand tracking solution to detect hand landmarks and then applies custom algorithms to determine which fingers are extended. The detection works by:

1. Identifying 21 key landmarks on each hand using MediaPipe
2. Calculating angles between finger joints
3. Measuring distances between fingertips and palm center
4. Using adaptive thresholds based on hand size for more accurate detection
5. Applying special logic for thumb detection, which is typically harder to track

The UI provides real-time feedback with color-coded information and status messages.
