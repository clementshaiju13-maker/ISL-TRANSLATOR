import os
import pickle
import time
import av
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Page setup
st.set_page_config(page_title="ISL Continuous Translator", layout="centered")
st.title("Indian Sign Language (ISL) Real-Time Translator")
st.write("Hold your sign steady for **1 second** to automatically register the letter.")

# 1. Load ML Model & Labels
@st.cache_resource
def load_isl_assets():
    model = tf.keras.models.load_model("isl_nn_model.h5")
    with open("isl_classes.pickle", "rb") as f:
        class_names = pickle.load(f)

    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return model, class_names, detector

model, class_names, detector = load_isl_assets()

def extract_features(landmarks_list):
    sorted_hands = sorted(landmarks_list, key=lambda hand: hand[0].x)
    all_x = [lm.x for hand in sorted_hands for lm in hand]
    all_y = [lm.y for hand in sorted_hands for lm in hand]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    scale = max(max_x - min_x, max_y - min_y, 1e-4)

    vector = []
    for i in range(2):
        if i < len(sorted_hands):
            for lm in sorted_hands[i]:
                vector.extend([(lm.x - min_x) / scale, (lm.y - min_y) / scale])
        else:
            vector.extend([0.0] * 42)
    return np.array([vector], dtype=np.float32)

# Global variables for continuous state tracking
class AppState:
    current_word = ""
    sentence = ""
    candidate = None
    first_seen = 0.0
    last_confirmed = 0.0

HOLD_TIME = 1.0
COOLDOWN = 1.5

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    # Downscale for smooth cloud processing
    small_img = cv2.resize(img, (320, 240))
    rgb_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

    results = detector.detect(mp_image)
    now = time.time()
    detected_letter = ""
    conf = 0.0

    if results.hand_landmarks:
        feats = extract_features(results.hand_landmarks)
        preds = model.predict(feats, verbose=0)[0]
        idx = np.argmax(preds)
        conf = float(preds[idx])

        if conf > 0.40:
            detected_letter = class_names[idx]
            if AppState.candidate == detected_letter:
                elapsed = now - AppState.first_seen
                if elapsed >= HOLD_TIME and (now - AppState.last_confirmed > COOLDOWN):
                    AppState.current_word += detected_letter
                    AppState.last_confirmed = now
            else:
                AppState.candidate = detected_letter
                AppState.first_seen = now
        else:
            AppState.candidate = None
    else:
        AppState.candidate = None

    # Render on-screen HUD directly on video stream
    h, w, _ = img.shape
    cv2.rectangle(img, (0, 0), (w, 80), (30, 30, 30), -1)
    status_text = f"Sign: {detected_letter} ({conf*100:.0f}%)" if detected_letter else "Sign: --"
    cv2.putText(img, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f"Word: {AppState.current_word}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(img, (0, h - 50), (w, h), (20, 20, 20), -1)
    cv2.putText(img, f"Sentence: {AppState.sentence}", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Google STUN server configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 2. WebRTC Live Video Streamer (No shutter buttons needed)
webrtc_streamer(
    key="isl-streamer",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# 3. Interactive Web Controls
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Space (Next Word)"):
        if AppState.current_word:
            AppState.sentence = (AppState.sentence + " " + AppState.current_word).strip()
            AppState.current_word = ""

with col2:
    if st.button("Backspace"):
        AppState.current_word = AppState.current_word[:-1]

with col3:
    if st.button("Clear All"):
        AppState.current_word = ""
        AppState.sentence = ""

st.markdown("---")
st.caption("Press **Space (Next Word)** to transfer the completed word to the sentence.")
