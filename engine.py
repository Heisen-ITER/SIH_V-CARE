# engine.py (WITH INCREASED SMOOTHING)

import time
import numpy as np
from collections import deque
import cv2
import base64

# --- CHANGED: Increased history length for smoother scores ---
SCORE_HISTORY_LENGTH = 15 # Was 7, now averaging over a longer period

# Constants and state variables
EMOTION_STRESS_MAP = {
    "Fear": 0.85, "Angry": 0.80, "Disgust": 0.65, "Sad": 0.55,
    "Surprise": 0.25, "Neutral": 0.0, "Happy": 0.0, "Unknown": 0.0
}
stress_history = deque([0.0] * SCORE_HISTORY_LENGTH, maxlen=SCORE_HISTORY_LENGTH)
fatigue_history = deque([0.0] * SCORE_HISTORY_LENGTH, maxlen=SCORE_HISTORY_LENGTH)
ALERT_THRESHOLD_STRESS = 75
ALERT_THRESHOLD_FATIGUE = 80
ALERT_PERSISTENCE_S = 4
last_stress_alert_time = 0
last_fatigue_alert_time = 0

# (The rest of the file is unchanged)

def _normalize(value, min_val, max_val):
    if max_val == min_val: return 0.0
    score = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, score))

def fuse_data(mavni_data, vani_data):
    global last_stress_alert_time, last_fatigue_alert_time
    
    # The engine now uses the SMOOTHED 'primary_emotion' for calculations
    emotion = mavni_data.get("primary_emotion", "Unknown")
    emotion_score = EMOTION_STRESS_MAP.get(emotion, 0.0)
    anomaly_score = vani_data.get("speech_anomaly_score", 0.0)
    
    raw_stress_score = emotion_score
    if emotion_score > 0.5 and anomaly_score > 0.4:
        raw_stress_score *= (1.0 + (anomaly_score * 1.2))
    elif emotion_score < 0.2 and anomaly_score > 0.6:
        raw_stress_score = anomaly_score * 0.5
        
    raw_stress_score = min(1.0, raw_stress_score)
    
    # Blink detection is currently disabled, so this will be 0
    blink_rate = mavni_data.get("blinks_per_minute", 0)
    raw_fatigue_score = _normalize(blink_rate, 18, 45)

    stress_history.append(raw_stress_score)
    fatigue_history.append(raw_fatigue_score)
    final_stress_score = np.median(list(stress_history))
    final_fatigue_score = np.median(list(fatigue_history))
    
    cwi = 100 * (1 - max(final_stress_score, final_fatigue_score))
    
    alert_message = None
    current_time = time.time()
    
    # Alert logic remains the same
    if len(stress_history) == SCORE_HISTORY_LENGTH and all(s * 100 > ALERT_THRESHOLD_STRESS for s in stress_history):
        if (current_time - last_stress_alert_time > ALERT_PERSISTENCE_S * 2):
            alert_message = f"HIGH STRESS DETECTED (Emotion: {emotion})"
            last_stress_alert_time = current_time

    face_crop_encoded = None
    face_crop_raw = mavni_data.get("face_crop_for_debug")
    if face_crop_raw is not None:
        _, buf = cv2.imencode('.jpg', face_crop_raw)
        face_crop_encoded = base64.b64encode(buf).decode("utf-8")

    return {
        "stress_level": round(final_stress_score * 100),
        "fatigue_level": round(final_fatigue_score * 100), 
        "cognitive_wellness_index": round(cwi),
        "alert": alert_message,
        "factors": {
            # This is the stable, smoothed emotion
            "emotion": emotion,
            "vocal_anomaly_factor": round(anomaly_score, 3),
            "blink_rate": blink_rate
        },
        "raw_data": {
            "video_frame": mavni_data.get("video_frame", None),
            "face_crop_frame": face_crop_encoded,
            # We can also send the instant emotion for debugging on the dashboard if needed
            "instant_emotion": mavni_data.get("instant_emotion", "---")
        }
    }