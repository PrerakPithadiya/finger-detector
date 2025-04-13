#!/usr/bin/env python3
"""
Finger Detector Application
Main entry point for the application

Author: Prerak Pithadiya
Email: prerak.pithadiya@gmail.com
GitHub: https://github.com/PrerakPithadiya
Copyright (c) 2025 Prerak Pithadiya
License: MIT License
"""

from src.finger_detector import FingerDetector
import cv2
import time

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Initialize finger detector
    detector = FingerDetector()

    # FPS calculation variables
    prev_time = 0
    curr_time = 0

    while True:
        # Read frame from webcam
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from camera. Check camera index.")
            break

        # Mirror the frame horizontally for a more intuitive interaction
        frame = cv2.flip(frame, 1)

        # Process the frame
        frame, _ = detector.process_frame(
            frame
        )  # We don't need to use the returned count

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Display the frame
        cv2.imshow("Finger Detector", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
