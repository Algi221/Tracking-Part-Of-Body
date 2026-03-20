import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import os
import urllib.request

# ==========================================
# REQUIREMENTS:
# pip install mediapipe opencv-python numpy
# ==========================================

def draw_lightning(frame, pt1, pt2, color=(255, 255, 0), thickness=2, glow_layer=None):
    """
    Draws a procedural lightning bolt between pt1 and pt2.
    """
    distance = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
    if distance < 5:
        return

    segments = max(1, int(distance // 15))
    pts = [pt1]
    
    # Generate zig-zag points
    for i in range(1, segments):
        t = i / segments
        # Interpolate and add random displacement
        nx = int(pt1[0] + (pt2[0]-pt1[0])*t + random.randint(-15, 15))
        ny = int(pt1[1] + (pt2[1]-pt1[1])*t + random.randint(-15, 15))
        pts.append((nx, ny))
    pts.append(pt2)
    
    # Draw segments
    for i in range(len(pts)-1):
        cv2.line(frame, pts[i], pts[i+1], color, thickness)
        if glow_layer is not None:
            # Draw thicker line on glow layer for volumetric effect
            cv2.line(glow_layer, pts[i], pts[i+1], color, thickness + 4)
            
        # Occasional branches
        if random.random() < 0.3:
            bx = pts[i][0] + random.randint(-25, 25)
            by = pts[i][1] + random.randint(-25, 25)
            cv2.line(frame, pts[i], (bx, by), color, max(1, thickness-1))
            if glow_layer is not None:
                cv2.line(glow_layer, pts[i], (bx, by), color, max(1, thickness))

def main():
    # Model Setup for newer MediaPipe Tasks API
    # Since mediapipe.solutions is unavailable in modern Python 3.12+ 
    # we use the Tasks API and download the models explicitly.
    
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

        # Dedicated glow layer for additive blending (particles and lightning)
        glow_layer = np.zeros_like(frame, dtype=np.uint8)

        left_hand_center = None
        right_hand_center = None

        cyan = (255, 255, 0)
        blue = (255, 100, 0)

        # Find hands
        if hand_result and hand_result.hand_landmarks:
            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                # Classify rough hand index
                if i == 0:
                    wrist = hand_landmarks[0]
                    left_hand_center = (int(wrist.x * w), int(wrist.y * h))
                    
                    fingertips = [8, 12, 16, 20, 4]
                    for tip_idx in fingertips:
                        pt = hand_landmarks[tip_idx]
                        px, py = int(pt.x * w), int(pt.y * h)
                        if random.random() > 0.2:
                            cv2.circle(glow_layer, (px, py), random.randint(10, 20), cyan, -1)
                            if left_hand_center:
                                draw_lightning(frame, left_hand_center, (px, py), cyan, 2, glow_layer)
                
                if i == 1:
                    wrist = hand_landmarks[0]
                    right_hand_center = (int(wrist.x * w), int(wrist.y * h))
                    
                    fingertips = [8, 12, 16, 20, 4]
                    for tip_idx in fingertips:
                        pt = hand_landmarks[tip_idx]
                        px, py = int(pt.x * w), int(pt.y * h)
                        if random.random() > 0.2:
                            cv2.circle(glow_layer, (px, py), random.randint(10, 20), cyan, -1)
                            if right_hand_center:
                                draw_lightning(frame, right_hand_center, (px, py), cyan, 2, glow_layer)

        # Lightning Arc between hands
        distance_between_hands = float('inf')
        if left_hand_center and right_hand_center:
            distance_between_hands = math.hypot(right_hand_center[0] - left_hand_center[0], 
                                                right_hand_center[1] - left_hand_center[1])
            # Draw intense lightning if hands are somewhat close
            if distance_between_hands < 400:
                for _ in range(3): # Multiple arcs
                    draw_lightning(frame, left_hand_center, right_hand_center, blue, 3, glow_layer)

        # Point-cloud Face Mesh HUD
        if face_result and face_result.face_landmarks:
            face_color_b = 255
            face_color_g = 255
            face_color_r = 0
            
            if distance_between_hands < 400:
                intensity = int((400 - distance_between_hands) / 400 * 255)
                face_color_b = max(0, face_color_b - intensity)
                face_color_g = max(0, face_color_g - intensity)
                face_color_r = min(255, face_color_r + intensity)

            color = (face_color_b, face_color_g, face_color_r)
            
            for face_landmarks in face_result.face_landmarks:
                for landmark in face_landmarks:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, color, -1)

        # --- APPLY EFFECTS ---
        # Blur the glow layer slightly for volumetric effect
        glow_layer = cv2.GaussianBlur(glow_layer, (15, 15), 0)
        
        # Additive Blending: add the glow layer directly to the original frame
        cv2.addWeighted(glow_layer, 2.0, frame, 1.0, 0, frame)

        cv2.imshow('Electric Aura & Face Mesh', frame)
        
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
