import cv2
import mediapipe as mp
import os
import numpy as np
import time

# List of words for which we need signs
#sign_words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#sign_words = ["thank you"]
# Create a folder to store sign data
sign_words = ['z']
dataset_folder = "sign_language_dataset"
os.makedirs(dataset_folder, exist_ok=True)

# Initialize MediaPipe and video capture
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)

# Collect landmarks for each word
for word in sign_words:
    print(f"Collecting data for sign '{word}'. Press 'q' to stop.")
    word_folder = os.path.join(dataset_folder, word)
    os.makedirs(word_folder, exist_ok=True)
    
    # Duration of capture per sign
    duration = 10
    end_time = time.time() + duration
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Prepare to collect landmarks for both hands
        all_hand_landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                single_hand_landmarks = []
                for landmark in hand_landmarks.landmark:
                    single_hand_landmarks.extend([landmark.x, landmark.y, landmark.z])  # Flatten each hand's landmarks
                all_hand_landmarks.append(single_hand_landmarks)

            # Pad to ensure exactly two sets of landmarks (for two hands)
            while len(all_hand_landmarks) < 2:
                all_hand_landmarks.append([0] * 63)  # Append zero-filled landmarks for missing hand (21 points * 3 coords)

            # Combine landmarks from both hands into one flat array
            hand_landmarks = np.array(all_hand_landmarks).flatten()
            frame_data_path = os.path.join(word_folder, f"{frame_count}.npy")
            np.save(frame_data_path, hand_landmarks)
            frame_count += 1

            # Draw landmarks on the frame for both hands
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Show frame
        cv2.imshow("Recording", frame)
        
        # Stop if 'q' is pressed or duration exceeded
        if cv2.waitKey(1) & 0xFF == ord('q') or time.time() > end_time:
            break

cap.release()
cv2.destroyAllWindows()
