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
        thumb_mcp = landmarks[self.finger_mcps[0]]
        pinky_mcp = landmarks[self.finger_mcps[4]]
        return self.distance_3d(thumb_mcp, pinky_mcp)

    def get_hand_height(self, landmarks):
        """Calculate the height of the hand"""
        middle_mcp = landmarks[self.finger_mcps[2]]
        middle_tip = landmarks[self.fingertips[2]]
        return self.distance_3d(middle_mcp, middle_tip)

    def distance_3d(self, point1, point2):
        """Calculate 3D distance between two points"""
        return math.sqrt(
            (point1.x - point2.x) ** 2
            + (point1.y - point2.y) ** 2
            + (point1.z - point2.z) ** 2
        )

    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points in degrees"""
        # Convert to 3D vectors
        vector1 = [point1.x - point2.x, point1.y - point2.y, point1.z - point2.z]
        vector2 = [point3.x - point2.x, point3.y - point2.y, point3.z - point2.z]

        # Calculate dot product
        dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))

        # Calculate magnitudes
        mag1 = math.sqrt(sum(v**2 for v in vector1))
        mag2 = math.sqrt(sum(v**2 for v in vector2))

        # Calculate angle in degrees
        if mag1 * mag2 == 0:
            return 0  # Avoid division by zero

        cos_angle = max(-1, min(1, dot_product / (mag1 * mag2)))
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def process_frame(self, frame):
        """Process a frame to detect hands and count fingers"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)

        # Default finger count variables
        total_finger_count = 0
        left_hand_count = 0
        right_hand_count = 0
        detected_hands = 0

        # Get frame dimensions
        h, w, _ = frame.shape

        # Draw hand landmarks and count fingers if hands are detected
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # Get hand type (Left or Right)
                hand_type = handedness.classification[0].label
                detected_hands += 1

                # Draw hand landmarks with different colors for better visibility
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Count fingers for this hand
                hand_finger_count = self.count_fingers(
                    hand_landmarks.landmark, hand_type
                )

                # Update finger counts based on hand type
                if hand_type == "Left":
                    left_hand_count = hand_finger_count
                else:  # Right hand
                    right_hand_count = hand_finger_count

                # Calculate total finger count from both hands
                total_finger_count = left_hand_count + right_hand_count

                # Display hand type and finger count with an elegant design
                wrist_landmark = hand_landmarks.landmark[self.wrist]
                wrist_x, wrist_y = int(wrist_landmark.x * w), int(wrist_landmark.y * h)

                # Choose more harmonious colors based on hand type
                if hand_type == "Left":
                    hand_text_color = (205, 90, 106)  # Soft rose color for left hand
                    hand_label = "LEFT"
                else:
                    hand_text_color = (94, 185, 160)  # Soft teal for right hand
                    hand_label = "RIGHT"

                # Create a small semi-transparent background for the hand label
                text_size = cv2.getTextSize(
                    f"{hand_label}: {hand_finger_count}",
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    2,
                )[0]

                # Draw a rounded rectangle background
                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (wrist_x - 50, wrist_y - 40),
                    (wrist_x - 50 + text_size[0] + 20, wrist_y - 5),
                    (40, 40, 40),  # Dark background
                    -1,
                )
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)  # 60% opacity

                # Add a colored accent line
                cv2.line(
                    frame,
                    (wrist_x - 50, wrist_y - 40),
                    (wrist_x - 50 + text_size[0] + 20, wrist_y - 40),
                    hand_text_color,
                    3,
                )

                # Display the hand label with improved typography
                cv2.putText(
                    frame,
                    f"{hand_label}: {hand_finger_count}",  # Show count for each hand
                    (wrist_x - 40, wrist_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,  # Slightly smaller for elegance
                    (
                        240,
                        240,
                        240,
                    ),  # Light text on dark background for better contrast
                    2,
                )

                # Highlight fingertips that are extended
                landmarks = hand_landmarks.landmark

                # For thumb, use the improved detection method
                thumb_tip = landmarks[self.fingertips[0]]
                thumb_ip = landmarks[self.finger_pips[0]]
                thumb_mcp = landmarks[self.finger_mcps[0]]
                index_mcp = landmarks[self.finger_mcps[1]]
                palm_center = landmarks[9]

                # Calculate metrics for thumb extension
                thumb_angle = self.calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
                thumb_to_index_distance = self.distance_3d(thumb_tip, index_mcp)
                thumb_to_palm_distance = self.distance_3d(thumb_tip, palm_center)
                hand_size = max(
                    self.get_hand_width(landmarks), self.get_hand_height(landmarks)
                )

                # Determine if thumb is extended using the same criteria as in count_fingers
                thumb_is_extended = (
                    thumb_angle > 150
                    and thumb_to_index_distance > 0.15 * hand_size
                    and thumb_to_palm_distance > 0.2 * hand_size
                )

                # Display thumb status in an elegant, subtle way
                cv2.putText(
                    frame,
                    f"Thumb: {int(thumb_angle)}°",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (120, 120, 140),  # Subtle blue-gray for a more refined look
                    1,
                )

                if thumb_is_extended:
                    # Thumb is extended, mark it
                    thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    cv2.circle(frame, (thumb_x, thumb_y), 15, (0, 255, 0), cv2.FILLED)

                # For other fingers, use the same improved method as in count_fingers
                for i in range(1, 5):  # For index, middle, ring, pinky
                    mcp = landmarks[self.finger_mcps[i]]
                    pip = landmarks[self.finger_pips[i]]
                    tip = landmarks[self.fingertips[i]]

                    # Calculate angle
                    angle = self.calculate_angle(mcp, pip, tip)

                    # Calculate distance from fingertip to palm center
                    palm_center = landmarks[9]
                    tip_to_palm_distance = self.distance_3d(tip, palm_center)

                    # Use the same thresholds as in count_fingers
                    angle_threshold = 150 if i == 1 else 140
                    distance_threshold = 0.3 * hand_size

                    # Display finger angles with a more elegant style
                    finger_names = ["Index", "Middle", "Ring", "Pinky"]
                    cv2.putText(
                        frame,
                        f"{finger_names[i-1]}: {int(angle)}°",
                        (10, 50 + (i - 1) * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,  # Slightly larger for better readability
                        (120, 120, 140),  # Subtle blue-gray for a more refined look
                        1,
                    )

                    # Check if finger is extended using both criteria
                    if (
                        angle > angle_threshold
                        and tip_to_palm_distance > distance_threshold
                    ):
                        # Finger is extended, mark it
                        tip_x, tip_y = int(tip.x * w), int(tip.y * h)
                        cv2.circle(frame, (tip_x, tip_y), 15, (0, 255, 0), cv2.FILLED)

        # Create a very simple, centered display for the number
        # Get frame dimensions for positioning
        h, w, _ = frame.shape

        # Define a more harmonious and elegant color palette based on finger count
        colors = {
            0: (120, 40, 31),  # Deep burgundy for zero
            1: (211, 84, 0),  # Burnt orange
            2: (243, 156, 18),  # Amber
            3: (39, 174, 96),  # Emerald green
            4: (41, 128, 185),  # Steel blue
            5: (142, 68, 173),  # Amethyst purple
            6: (22, 160, 133),  # Teal
            7: (192, 57, 43),  # Pomegranate red
            8: (52, 73, 94),  # Wet asphalt blue
            9: (230, 126, 34),  # Carrot orange
            10: (241, 196, 15),  # Sunflower yellow
        }
        number_color = colors.get(
            total_finger_count, (236, 240, 241)
        )  # Cloud white for numbers > 10

        # Create a very simple, centered display for the total finger count

        # Position the elements in the center top of the screen
        # Adjust position based on digit count to ensure it stays on screen
        digit_count = (
            1 if total_finger_count < 10 else 2
        )  # Check if it's a single or double digit

        # Calculate center position
        center_x = w // 2
        number_y = 100  # Lower position from the top

        # Define accent color based on finger count
        accent_color = number_color

        # Calculate text size to center properly
        text_size = cv2.getTextSize(
            str(total_finger_count),
            cv2.FONT_HERSHEY_DUPLEX,
            3.5,  # Slightly smaller font for better fit
            5,
        )[0]

        # Calculate position to center the text
        number_x = center_x - (text_size[0] // 2)

        # Display the total count with a clean, professional look
        # First draw a subtle shadow for depth
        cv2.putText(
            frame,
            str(total_finger_count),
            (number_x + 3, number_y + 3),  # Small offset for shadow
            cv2.FONT_HERSHEY_DUPLEX,
            3.5,  # Slightly smaller for better fit
            (40, 40, 40),  # Dark shadow
            5,  # Thick outline
        )

        # Draw the number on top
        cv2.putText(
            frame,
            str(total_finger_count),
            (number_x, number_y),
            cv2.FONT_HERSHEY_DUPLEX,
            3.5,  # Slightly smaller for better fit
            accent_color,  # Use the accent color
            5,  # Thick outline
        )

        # Display number of hands detected in a subtle way
        hand_icon = "👐" if detected_hands == 2 else "👋"
        cv2.putText(
            frame,
            f"{hand_icon} {detected_hands}",
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
