# mavni.py (COMPLETE FINAL VERSION)

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import time
import base64

print("Starting MAVNI module...")

try:
    model = tf.keras.models.load_model('model_file_30epochs.h5')
    print("Loaded Keras emotion model (model_file_30epochs.h5).")
except Exception as e:
    print(f"Could not load Keras model: {e}")
    exit(1)

try:
    face_cascade = cv2.CascadeClassifier('/Users/omkarsarkar/Desktop/Vyom/mavni-1/haarcascade_frontalface_default.xml')
    print("Loaded Haar Cascade face detector.")
except Exception as e:
    print(f"Could not load Haar Cascade file: {e}")
    exit(1)
    
EMOTION_LABELS = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
MODEL_INPUT_SIZE = (48, 48)

EMOTION_WEIGHTS = {
    "Angry": 0.8, "Disgust": 0.6, "Fear": 0.7,
    "Happy": 3.0,
    "Neutral": 1.0, "Sad": 0.3, "Surprise": 1.1
}
emotion_weights_array = np.array([EMOTION_WEIGHTS[EMOTION_LABELS[i]] for i in range(len(EMOTION_LABELS))])

HAPPY_INDEX = list(EMOTION_LABELS.keys())[list(EMOTION_LABELS.values()).index('Happy')]
NEUTRAL_INDEX = list(EMOTION_LABELS.keys())[list(EMOTION_LABELS.values()).index('Neutral')]
SAD_INDEX = list(EMOTION_LABELS.keys())[list(EMOTION_LABELS.values()).index('Sad')]
ANGRY_INDEX = list(EMOTION_LABELS.keys())[list(EMOTION_LABELS.values()).index('Angry')]
FEAR_INDEX = list(EMOTION_LABELS.keys())[list(EMOTION_LABELS.values()).index('Fear')]
NEGATIVE_INDICES = [SAD_INDEX, ANGRY_INDEX, FEAR_INDEX]

print("Warming up the model...")
try:
    dummy_input = np.random.rand(1, 48, 48, 1)
    _ = model.predict(dummy_input, verbose=0)
    print("Model is warmed up and ready.")
except Exception as e:
    print(f"Model warm-up failed: {e}")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

emotion_history = deque(maxlen=5) 
blinks_in_last_minute = deque()
frames_below_threshold = 0
last_known_result = np.zeros((1, len(EMOTION_LABELS)))

def calc_ear(landmarks, indices):
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in indices])
    v1 = np.linalg.norm(pts[1] - pts[15]); v2 = np.linalg.norm(pts[2] - pts[14]); v3 = np.linalg.norm(pts[3] - pts[13])
    h = np.linalg.norm(pts[0] - pts[8])
    return (v1 + v2 + v3) / (3.0 * h) if h != 0 else 0.0

# THIS IS THE FUNCTION THAT WAS MISSING
def encode_frame(frame):
    h, w = frame.shape[:2]; new_w = 320; new_h = int((new_w / w) * h)
    resized = cv2.resize(frame, (new_w, new_h))
    _, buf = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return base64.b64encode(buf).decode("utf-8")

def analyze_frame(frame):
    global frames_below_threshold, last_known_result
    
    output = {
        "fatigue_level": "Low", "primary_emotion": "---", 
        "blinks_per_minute": len(blinks_in_last_minute), "video_frame": None, 
        "face_crop_for_debug": None, "raw_emotion_scores": last_known_result
    }

    detected_emotion = "Neutral"
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    if len(faces) > 0:
        frame_center_x = frame.shape[1] / 2
        closest_face = min(faces, key=lambda f: abs((f[0] + f[2]/2) - frame_center_x))
        (x, y, w, h) = closest_face

        face_crop = gray_frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_crop, MODEL_INPUT_SIZE)
        output["face_crop_for_debug"] = resized_face
        
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

        result = model.predict(reshaped_face, verbose=0)
        
        result *= emotion_weights_array
        last_known_result = result
        output["raw_emotion_scores"] = result
        
        sorted_indices = np.argsort(result[0])
        top_1_idx = sorted_indices[-1]
        top_2_idx = sorted_indices[-2]
        final_emotion_idx = top_1_idx

        is_neutral_top = (top_1_idx == NEUTRAL_INDEX)
        is_happy_second = (top_2_idx == HAPPY_INDEX)

        if is_neutral_top and is_happy_second:
            happy_close_to_neutral = result[0][HAPPY_INDEX] > (result[0][NEUTRAL_INDEX] * 0.6)
            negative_scores = [result[0][i] for i in NEGATIVE_INDICES]
            are_negatives_low = max(negative_scores) < (result[0][NEUTRAL_INDEX] * 0.2)

            if happy_close_to_neutral and are_negatives_low:
                final_emotion_idx = HAPPY_INDEX

        detected_emotion = EMOTION_LABELS.get(final_emotion_idx, "Unknown")
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

    emotion_history.append(detected_emotion)
    output["instant_emotion"] = detected_emotion

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        avg_ear = (calc_ear(landmarks, LEFT_EYE) + calc_ear(landmarks, RIGHT_EYE)) / 2
        if avg_ear < 0.25: frames_below_threshold += 1
        else:
            if frames_below_threshold >= 2: blinks_in_last_minute.append(time.time())
            frames_below_threshold = 0
        now = time.time()
        while blinks_in_last_minute and blinks_in_last_minute[0] < now - 60:
            blinks_in_last_minute.popleft()
        bpm = len(blinks_in_last_minute)
        output["blinks_per_minute"] = bpm
        if bpm >= 22: output["fatigue_level"] = "High"
        elif bpm >= 18: output["fatigue_level"] = "Moderate"

    if emotion_history:
        stable_emotion = max(set(emotion_history), key=emotion_history.count)
        output["primary_emotion"] = stable_emotion
    else:
        output["primary_emotion"] = detected_emotion
    
    output["video_frame"] = encode_frame(frame)
    return output

# --- Standalone Testing Section ---
if __name__ == "__main__":
    video_path = "/Users/omkarsarkar/Desktop/Vyom/mavni-1/takeoff.mp4" 
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        exit()

    frame_counter = 0
    FRAME_SKIP = 2
    
    # Store the last valid result to keep the display from flickering on skipped frames
    last_result = {} 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video finished. Looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_counter += 1
        if frame_counter % FRAME_SKIP == 0:
            # Only run the heavy analysis every N frames
            last_result = analyze_frame(frame)
        
        # Always draw the annotations, but use the last known result
        if last_result:
            display_text = f"Stable: {last_result.get('primary_emotion', '...')} | Instant: {last_result.get('instant_emotion', '...')}"
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y_pos = 60
            scores_to_display = last_result.get("raw_emotion_scores", np.zeros((1, len(EMOTION_LABELS))))
            for i in range(len(EMOTION_LABELS)):
                label = EMOTION_LABELS[i]
                score = scores_to_display[0][i]
                bar_color = (0, 255, 0) if label == last_result.get('instant_emotion') else (100, 200, 100)

                cv2.putText(frame, label, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(frame, (100, y_pos - 10), (300, y_pos + 5), (50, 50, 50), -1)
                cv2.rectangle(frame, (100, y_pos - 10), (100 + int(score * 200), y_pos + 5), bar_color, -1)
                y_pos += 20
            
            if last_result.get("face_crop_for_debug") is not None:
                debug_img = cv2.resize(last_result["face_crop_for_debug"], (224, 224), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Model Input (48x48 Grayscale)", debug_img)

        cv2.imshow("MAVNI Standalone Video Test", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()