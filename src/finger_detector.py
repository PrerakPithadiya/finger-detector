"""
Finger Detector Module
Contains the FingerDetector class for hand and finger detection
"""

import cv2
import mediapipe as mp
import time
import math


class FingerDetector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Finger tip IDs
        self.fingertips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

        # Finger second joint IDs
        self.finger_pips = [3, 6, 10, 14, 18]  # thumb, index, middle, ring, pinky

        # Finger knuckle IDs
        self.finger_mcps = [2, 5, 9, 13, 17]  # thumb, index, middle, ring, pinky

        # Wrist landmark
        self.wrist = 0

    def count_fingers(self, landmarks, hand_type):
        """Count the number of extended fingers using a more robust method"""
        count = 0

        # Get hand dimensions
        hand_width = self.get_hand_width(landmarks)
        hand_height = self.get_hand_height(landmarks)
        hand_size = max(hand_width, hand_height)

        # Improved thumb detection that works better for folded thumbs
        # Get relevant landmarks for thumb detection
        thumb_tip = landmarks[self.fingertips[0]]  # Thumb tip
        thumb_ip = landmarks[self.finger_pips[0]]  # Thumb IP joint (second joint)
        thumb_mcp = landmarks[self.finger_mcps[0]]  # Thumb MCP joint (knuckle)
        index_mcp = landmarks[self.finger_mcps[1]]  # Index finger knuckle

        # Calculate angles to determine if thumb is extended or folded
        thumb_angle = self.calculate_angle(thumb_mcp, thumb_ip, thumb_tip)

        # Calculate distance from thumb tip to index MCP (when thumb is folded into palm, this distance is small)
        thumb_to_index_distance = self.distance_3d(thumb_tip, index_mcp)

        # Calculate distance from thumb tip to palm center (landmark 9)
        palm_center = landmarks[9]  # Palm center landmark
        thumb_to_palm_distance = self.distance_3d(thumb_tip, palm_center)

        # A folded thumb will have:
        # 1. A smaller angle (typically < 150 degrees)
        # 2. Be close to the index finger knuckle or palm center
        thumb_is_extended = (
            thumb_angle > 150
            and thumb_to_index_distance > 0.15 * hand_size
            and thumb_to_palm_distance > 0.2 * hand_size
        )

        if thumb_is_extended:
            count += 1

        # For other fingers, use a more robust method that combines angle and distance
        for i in range(1, 5):  # For index, middle, ring, pinky
            # Get the three points to calculate angle: MCP, PIP, and TIP
            mcp = landmarks[self.finger_mcps[i]]
            pip = landmarks[self.finger_pips[i]]
            tip = landmarks[self.fingertips[i]]

            # Calculate the angle between the finger segments
            angle = self.calculate_angle(mcp, pip, tip)

            # Calculate distance from fingertip to palm center
            palm_center = landmarks[9]  # Palm center landmark
            tip_to_palm_distance = self.distance_3d(tip, palm_center)

            # For a closed fist, fingertips are close to the palm center
            # and the angle is typically less than 160 degrees

            # Adjust threshold based on finger (index finger might need different threshold than pinky)
            angle_threshold = (
                150 if i == 1 else 140
            )  # Higher threshold for index finger

            # Distance threshold as a proportion of hand size
            distance_threshold = 0.3 * hand_size

            # A finger is extended if:
            # 1. The angle is large enough (straighter finger)
            # 2. The fingertip is far enough from the palm center
            if angle > angle_threshold and tip_to_palm_distance > distance_threshold:
                count += 1

        return count

    def get_hand_width(self, landmarks):
        """Calculate the width of the hand"""
        # Use the distance between the thumb MCP and pinky MCP
        thumb_mcp = landmarks[self.finger_mcps[0]]
        pinky_mcp = landmarks[self.finger_mcps[4]]
        return self.distance_3d(thumb_mcp, pinky_mcp)

    def get_hand_height(self, landmarks):
        """Calculate the height of the hand"""
        # Use the distance between the wrist and middle finger tip
        wrist = landmarks[self.wrist]
        middle_tip = landmarks[self.fingertips[2]]
        return self.distance_3d(wrist, middle_tip)

    def distance_3d(self, point1, point2):
        """Calculate 3D distance between two points"""
        return math.sqrt(
            (point1.x - point2.x) ** 2
            + (point1.y - point2.y) ** 2
            + (point1.z - point2.z) ** 2
        )

    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points in degrees"""
        # Convert landmarks to numpy arrays for easier vector operations
        p1 = (point1.x, point1.y, point1.z)
        p2 = (point2.x, point2.y, point2.z)
        p3 = (point3.x, point3.y, point3.z)

        # Calculate vectors
        v1 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
        v2 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])

        # Calculate dot product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

        # Calculate magnitudes
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)

        # Calculate angle in radians and convert to degrees
        # Ensure we don't divide by zero and the value is within [-1, 1]
        cos_angle = max(-1, min(1, dot_product / (mag1 * mag2)))
        angle_rad = math.acos(cos_angle)
        angle_deg = angle_rad * 180 / math.pi

        return angle_deg

    def process_frame(self, frame):
        """Process a frame to detect hands and count fingers"""
        # Get frame dimensions
        h, w, c = frame.shape

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)

        # Variables to track finger counts
        total_finger_count = 0
        left_hand_count = 0
        right_hand_count = 0
        detected_hands = 0

        # Check if hands are detected
        if results.multi_hand_landmarks:
            detected_hands = len(results.multi_hand_landmarks)

            # Process each detected hand
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine hand type (left or right)
                hand_type = (
                    "Left"
                    if results.multi_handedness[idx].classification[0].label == "Left"
                    else "Right"
                )

                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Count fingers for this hand
                finger_count = self.count_fingers(hand_landmarks.landmark, hand_type)

                # Update counts based on hand type
                if hand_type == "Left":
                    left_hand_count = finger_count
                else:
                    right_hand_count = finger_count

                total_finger_count += finger_count

                # Calculate hand center for placing text
                cx, cy = 0, 0
                for lm in hand_landmarks.landmark:
                    cx += lm.x
                    cy += lm.y
                cx = int(cx / len(hand_landmarks.landmark) * w)
                cy = int(cy / len(hand_landmarks.landmark) * h)

                # Choose color based on hand type
                color = (0, 255, 0) if hand_type == "Right" else (0, 0, 255)

                # Display finger count near the hand
                cv2.putText(
                    frame,
                    f"{hand_type}: {finger_count}",
                    (cx - 60, cy - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

        # Choose color for the number display based on finger count
        if total_finger_count <= 3:
            number_color = (0, 0, 255)  # Red for low counts
        elif total_finger_count <= 7:
            number_color = (0, 165, 255)  # Orange for medium counts
        else:
            number_color = (0, 255, 0)  # Green for high counts

        # Display the number of detected hands
        cv2.putText(
            frame,
            f"Hands: {detected_hands}",
            (20, 30),  # Top left corner
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),  # Light gray
            1,
        )

        # Add a descriptive message based on total finger count
        if total_finger_count == 0:
            message = "NO FINGERS DETECTED"
            message_color = (0, 0, 255)  # Red
        elif total_finger_count == 10:
            message = "ALL FINGERS UP ON BOTH HANDS"
            message_color = (255, 215, 0)  # Gold
        elif detected_hands == 2:
            message = f"TOTAL: {total_finger_count} FINGER{'S' if total_finger_count > 1 else ''} (L:{left_hand_count} + R:{right_hand_count})"
            message_color = number_color
        else:
            message = f"{total_finger_count} FINGER{'S' if total_finger_count > 1 else ''} DETECTED"
            message_color = number_color

        # Create an elegant message bar at the bottom
        bar_height = 60

        # Create a gradient-like effect for the message bar
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (0, h - bar_height), (w, h), (30, 30, 35), -1  # Dark background
        )

        # Add a subtle accent line at the top of the bar
        cv2.rectangle(
            overlay,
            (0, h - bar_height),
            (w, h - bar_height + 3),
            message_color,  # Use message color for accent
            -1,
        )

        # Apply the overlay with transparency
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)  # 85% opacity

        # Calculate text position for perfect centering
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_x = w // 2 - text_size[0] // 2
        text_y = h - bar_height // 2 + 5

        # Display the message with improved typography
        cv2.putText(
            frame,
            message,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,  # Slightly smaller for elegance
            (240, 240, 240),  # Light text for better contrast
            2,
        )

        # Add a timestamp with a more elegant design
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(
            frame,
            timestamp,
            (w - 100, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),  # Subtle gray
            1,
        )

        # Add a small app title/signature in the corner
        cv2.putText(
            frame,
            "Finger Counter",
            (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),  # Subtle gray
            1,
        )

        return frame, total_finger_count
