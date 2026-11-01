import os
import pickle
import time
import base64
import io
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import streamlit as st

st.set_page_config(page_title="ISL Translator", layout="centered")
st.title("Indian Sign Language (ISL) Real-Time Translator")

# 1. Load ML Assets
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

# Session tracking initialization
if "current_word" not in st.session_state:
    st.session_state.current_word = ""
if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "candidate" not in st.session_state:
    st.session_state.candidate = None
if "first_seen" not in st.session_state:
    st.session_state.first_seen = 0.0
if "last_confirmed" not in st.session_state:
    st.session_state.last_confirmed = 0.0

HOLD_TIME = 1.0
COOLDOWN = 1.5

# 2. Camera Component (HTML5 Canvas auto-feeder — No STUN/TURN required)
img_file = st.camera_input("Camera Feed")

if img_file is not None:
    now = time.time()
    img = Image.open(img_file)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) if frame.shape[-1] == 4 else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to speed up cloud processing
    small_frame = cv2.resize(frame, (320, 240))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_frame)
    res = detector.detect(mp_image)

    detected_letter = ""
    conf = 0.0

    if res.hand_landmarks:
        feats = extract_features(res.hand_landmarks)
        preds = model.predict(feats, verbose=0)[0]
        idx = np.argmax(preds)
        conf = float(preds[idx])

        if conf > 0.40:
            detected_letter = class_names[idx]
            if st.session_state.candidate == detected_letter:
                elapsed = now - st.session_state.first_seen
                if elapsed >= HOLD_TIME and (now - st.session_state.last_confirmed > COOLDOWN):
                    st.session_state.current_word += detected_letter
                    st.session_state.last_confirmed = now
            else:
                st.session_state.candidate = detected_letter
                st.session_state.first_seen = now
        else:
            st.session_state.candidate = None
    else:
        st.session_state.candidate = None

    if detected_letter:
        st.info(f"Detected Sign: **{detected_letter}** ({conf * 100:.1f}%)")
    else:
        st.caption("No clear sign detected.")

# 3. Word & Sentence Controls
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Space (Next Word)"):
        if st.session_state.current_word:
            st.session_state.sentence = (st.session_state.sentence + " " + st.session_state.current_word).strip()
            st.session_state.current_word = ""
            st.rerun()

with col2:
    if st.button("Backspace"):
        st.session_state.current_word = st.session_state.current_word[:-1]
        st.rerun()

with col3:
    if st.button("Clear All"):
        st.session_state.current_word = ""
        st.session_state.sentence = ""
        st.session_state.candidate = None
        st.rerun()

# 4. Display Outputs
st.subheader("Current Word")
st.text(st.session_state.current_word if st.session_state.current_word else "[Empty]")

st.subheader("Full Sentence")
st.text(st.session_state.sentence if st.session_state.sentence else "[Empty]")
