import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request

# ==========================================
# REQUIREMENTS:
# pip install mediapipe opencv-python numpy
# ==========================================

# Standard MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky + palm
]

def main():
    if not os.path.exists('hand_landmarker.task'):
        print("Downloading hand_landmarker.task...")
        urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'hand_landmarker.task')

    if not os.path.exists('face_landmarker.task'):
        print("Downloading face_landmarker.task...")
        urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'face_landmarker.task')

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Initialize Hand Landmarker
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hand_landmarker = HandLandmarker.create_from_options(hand_options)

    # Initialize Face Landmarker
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )
    face_landmarker = FaceLandmarker.create_from_options(face_options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting webcam... Press 'q' or 'ESC' to exit.")

    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1) # Mirror display
        h, w, _ = frame.shape
        
        # Convert the BGR image to RGB format required by MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Process using Tasks API
        hand_result = hand_landmarker.detect(mp_image)
        face_result = face_landmarker.detect(mp_image)

        # -------------------
        # HAND TRACKING HUD
        # -------------------
        if hand_result and hand_result.hand_landmarks:
            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                pts = []
                for landmark in hand_landmarks:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    pts.append((x, y))
                
                # Draw connections (Thin white lines)
                for connection in HAND_CONNECTIONS:
                    pt1 = pts[connection[0]]
                    pt2 = pts[connection[1]]
                    cv2.line(frame, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Draw joint dots (White circles)
                for pt in pts:
                    cv2.circle(frame, pt, 3, (255, 255, 255), -1)

                # Optional Neon Bounding Box for Hand
                if pts:
                    x_coords = [p[0] for p in pts]
                    y_coords = [p[1] for p in pts]
                    min_hx, max_hx = min(x_coords), max(x_coords)
                    min_hy, max_hy = min(y_coords), max(y_coords)
                    
                    hand_label = "Hand"
                    if hand_result.handedness and len(hand_result.handedness) > i:
                        hand_label = f"{hand_result.handedness[i][0].category_name} Hand"
                    
                    # Draw a subtle cyan tracking box for hands
                    cv2.rectangle(frame, (min_hx - 10, min_hy - 10), (max_hx + 10, max_hy + 10), (255, 255, 0), 1)
                    cv2.putText(frame, hand_label, (min_hx - 10, min_hy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # -------------------
        # FACE TRACKING HUD
        # -------------------
        if face_result and face_result.face_landmarks:
            face_pts = []
            for face_landmarks in face_result.face_landmarks:
                for landmark in face_landmarks:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    face_pts.append((x, y))
                    # Draw dense point-cloud (White dots)
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
                
                # Draw Neon Purple/Pink Bounding Box framing the face
                if face_pts:
                    x_coords = [p[0] for p in face_pts]
                    y_coords = [p[1] for p in face_pts]
                    min_fx, max_fx = min(x_coords), max(x_coords)
                    min_fy, max_fy = min(y_coords), max(y_coords)
                    
                    neon_purple = (255, 0, 255) # BGR
                    # Draw Main Box
                    cv2.rectangle(frame, (min_fx - 15, min_fy - 15), (max_fx + 15, max_fy + 15), neon_purple, 2)
                    
                    # Draw Crosshairs/Corner accents
                    length = 20
                    # Top Left
                    cv2.line(frame, (min_fx - 15, min_fy - 15), (min_fx - 15 + length, min_fy - 15), neon_purple, 4)
                    cv2.line(frame, (min_fx - 15, min_fy - 15), (min_fx - 15, min_fy - 15 + length), neon_purple, 4)
                    # Bottom Right
                    cv2.line(frame, (max_fx + 15, max_fy + 15), (max_fx + 15 - length, max_fy + 15), neon_purple, 4)
                    cv2.line(frame, (max_fx + 15, max_fy + 15), (max_fx + 15, max_fy + 15 - length), neon_purple, 4)

                    cv2.putText(frame, "Face", (min_fx - 15, min_fy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, neon_purple, 1)

        # -------------------
        # TIMECODE UI
        # -------------------
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        frames = int((elapsed * 30) % 30) # Simulating 30fps Timecode Format
        timecode_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"
        
        # Bottom center placing
        if timecode_str:
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(timecode_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            txt_x = (w - text_size[0]) // 2
            txt_y = h - 30
            
            # Dark grey background bar for text
            cv2.rectangle(frame, (txt_x - 15, txt_y - text_size[1] - 10), (txt_x + text_size[0] + 15, txt_y + 10), (40, 40, 40), -1)
            cv2.putText(frame, timecode_str, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        cv2.imshow('Face & Hand Tracking UI', frame)
        
        # Press 'q' or ESC to exit
        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    face_landmarker.close()

if __name__ == "__main__":
    main()
