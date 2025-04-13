# Finger Detector

A Python application that uses computer vision to detect and count fingers in real-time using a webcam.

## Features

- Real-time finger detection and counting
- Works with one or two hands simultaneously
- Displays finger count with visual feedback
- Elegant UI with status messages and timestamps

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- Webcam

## Installation

1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run the application: `python finger_detector.py`

## Usage

- Show your hand(s) to the camera
- The application will detect and count your fingers in real-time
- Press 'q' to exit the application

## How it Works

This application uses MediaPipe's hand tracking solution to detect hand landmarks and then applies custom algorithms to determine which fingers are extended. It works by analyzing the angles between finger joints and the distances between fingertips and the palm.
