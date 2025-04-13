"""
Finger Detector Module
Contains the FingerDetector class for hand and finger detection

Author: Prerak Pithadiya
Email: prerakpithadiya@gmail.com
GitHub: https://github.com/PrerakPithadiya
Copyright (c) 2025 Prerak Pithadiya
License: MIT License
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
        """Count the number of extended fingers using a more robust method
        
        Args:
            landmarks: MediaPipe hand landmarks (21 points)
            hand_type: String indicating 'Left' or 'Right' hand
            
        Returns:
            count: Integer representing number of extended fingers (0-5)
        """
        count = 0

        # Get hand dimensions to scale thresholds appropriately for different hand sizes and distances from camera
        hand_width = self.get_hand_width(landmarks)  # Width between thumb and pinky MCP joints
        hand_height = self.get_hand_height(landmarks)  # Height from wrist to middle fingertip
        hand_size = max(hand_width, hand_height)  # Use the larger dimension for threshold scaling

        # SECTION 1: THUMB DETECTION
        # Thumb detection is challenging because it has different movement patterns than other fingers
        # We use a combination of angle and distance metrics for more accurate detection
        
        # Get relevant landmarks for thumb detection
        thumb_tip = landmarks[self.fingertips[0]]  # Thumb tip (landmark 4)
        thumb_ip = landmarks[self.finger_pips[0]]  # Thumb IP joint/second joint (landmark 3)
        thumb_mcp = landmarks[self.finger_mcps[0]]  # Thumb MCP joint/knuckle (landmark 2)
        index_mcp = landmarks[self.finger_mcps[1]]  # Index finger knuckle (landmark 5)

        # Calculate angle between the three key thumb points to determine extension
        # A straight thumb will have an angle closer to 180 degrees
        thumb_angle = self.calculate_angle(thumb_mcp, thumb_ip, thumb_tip)

        # Calculate distance from thumb tip to index MCP 
        # When thumb is folded into palm or adducted, this distance is small
        # When thumb is extended or abducted, this distance is larger
        thumb_to_index_distance = self.distance_3d(thumb_tip, index_mcp)

        # Calculate distance from thumb tip to palm center (landmark 9)
        # This provides another metric for thumb extension
        palm_center = landmarks[9]  # Palm center landmark (between index and middle finger MCPs)
        thumb_to_palm_distance = self.distance_3d(thumb_tip, palm_center)

        # Determine if thumb is extended using multiple criteria
        # A folded/non-extended thumb will typically have:
        # 1. A smaller angle between joints (typically < 150 degrees)
        # 2. Be close to the index finger knuckle (thumb_to_index_distance is small)
        # 3. Be close to the palm center (thumb_to_palm_distance is small)
        # 
        # Note: Thresholds are scaled by hand_size to work at different distances from camera
        thumb_is_extended = (
            thumb_angle > 150  # Angle threshold for extension
            and thumb_to_index_distance > 0.15 * hand_size  # Must be far enough from index knuckle
            and thumb_to_palm_distance > 0.2 * hand_size  # Must be far enough from palm center
        )

        if thumb_is_extended:
            count += 1

        # SECTION 2: OTHER FINGERS DETECTION (Index, Middle, Ring, Pinky)
        # For non-thumb fingers, we use a combination of:
        # 1. Joint angles (straighter finger = extended)
        # 2. Distance from fingertip to palm (greater distance = extended)
        
        for i in range(1, 5):  # Iterate through index (1), middle (2), ring (3), pinky (4)
            # Get the three key points needed to calculate finger extension:
            mcp = landmarks[self.finger_mcps[i]]  # Metacarpophalangeal joint (knuckle)
            pip = landmarks[self.finger_pips[i]]  # Proximal interphalangeal joint (middle joint)
            tip = landmarks[self.fingertips[i]]  # Fingertip

            # Calculate the angle between the finger segments
            # An extended finger will have a straighter angle (closer to 180 degrees)
            # A curled finger will have a more acute angle (typically < 140-150 degrees)
            angle = self.calculate_angle(mcp, pip, tip)

            # Calculate distance from fingertip to palm center
            # Extended fingers have tips further from the palm center
            # Curled fingers have tips closer to the palm center
            palm_center = landmarks[9]  # Palm center landmark
            tip_to_palm_distance = self.distance_3d(tip, palm_center)

            # Adaptive thresholds for different fingers
            # Note: Different fingers need different thresholds because:
            # 1. Index finger tends to straighten more than others even when making a fist
            # 2. Pinky and ring fingers tend to curl more even when partially extended
            
            # Angle threshold: higher for index finger (needs to be straighter to count as extended)
            angle_threshold = (
                150 if i == 1 else 140
            )  # 150° for index finger, 140° for others

            # Distance threshold as a proportion of hand size
            # This scales automatically based on hand size/distance from camera
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
        """Calculate angle between three points in 3D space (in degrees)
        
        This function calculates the angle formed at point2 by the lines from 
        point2 to point1 and point2 to point3.
        
        Args:
            point1, point2, point3: MediaPipe landmark points with x, y, z coordinates
            
        Returns:
            angle_deg: Angle in degrees (0-180)
        """
        # Convert MediaPipe landmarks to coordinate tuples for vector calculations
        p1 = (point1.x, point1.y, point1.z)  # First point coordinates
        p2 = (point2.x, point2.y, point2.z)  # Middle point (vertex of angle)
        p3 = (point3.x, point3.y, point3.z)  # Third point coordinates

        # Calculate vectors from middle point to other points
        # v1 = vector from point2 to point1
        # v2 = vector from point2 to point3
        v1 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
        v2 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])

        # Calculate dot product of the two vectors: v1·v2 = |v1|×|v2|×cos(θ)
        dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

        # Calculate magnitudes (lengths) of both vectors
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)

        # Calculate angle using the dot product formula: cos(θ) = (v1·v2)/(|v1|×|v2|)
        # Clamp value to [-1, 1] to handle floating point errors
        cos_angle = max(-1, min(1, dot_product / (mag1 * mag2)))
        
        # Convert from radians to degrees
        angle_rad = math.acos(cos_angle)  # arccos gives angle in radians
        angle_deg = angle_rad * 180 / math.pi  # Convert to degrees

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
