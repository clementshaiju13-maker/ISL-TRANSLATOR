import os
import pickle
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import gradio as gr

# 1. Load Model & Label Classes
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

# 2. Continuous Prediction (Camera feeds frames automatically)
def predict_live(image):
    if image is None:
        return "Waiting for camera...", ""

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    res = detector.detect(mp_image)

    if not res.hand_landmarks:
        return "No hands visible", "0%"

    feats = extract_features(res.hand_landmarks)
    preds = model.predict(feats, verbose=0)[0]
    idx = np.argmax(preds)
    conf = float(preds[idx])

    if conf > 0.40:
        return class_names[idx], f"{conf * 100:.1f}%"
    return "Uncertain gesture", f"{conf * 100:.1f}%"

demo = gr.Interface(
    fn=predict_live,
    inputs=gr.Image(sources=["webcam"], streaming=True),
    outputs=[
        gr.Textbox(label="Detected Sign"),
        gr.Textbox(label="Confidence")
    ],
    title="Indian Sign Language (ISL) Recognition",
    live=True
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
