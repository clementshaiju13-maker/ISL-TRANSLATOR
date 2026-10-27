import os
import pickle
import time
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

# Global tracker for hands-free letter locking
tracker = {
    "candidate": None,
    "first_seen": 0,
    "last_confirmed_time": 0
}

HOLD_TIME = 1.2        # Must hold sign steady for 1.2s to commit
COOLDOWN_TIME = 2.0    # 2s cooldown before the same letter can be added again

def process_stream(image, current_word, current_sentence):
    now = time.time()
    
    if image is None:
        return "Camera loading...", "0%", current_word, current_sentence, "Start recording to activate camera"

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    res = detector.detect(mp_image)

    if not res.hand_landmarks:
        tracker["candidate"] = None
        return "No hands visible", "0%", current_word, current_sentence, "Place both hands in front of camera"

    feats = extract_features(res.hand_landmarks)
    preds = model.predict(feats, verbose=0)[0]
    idx = np.argmax(preds)
    conf = float(preds[idx])
    detected_letter = class_names[idx]

    if conf < 0.40:
        tracker["candidate"] = None
        return "Uncertain gesture", f"{conf * 100:.1f}%", current_word, current_sentence, "Adjust your hand shape"

    status_msg = f"Holding '{detected_letter}'..."

    # Hands-free auto-commit logic
    if tracker["candidate"] == detected_letter:
        elapsed = now - tracker["first_seen"]
        progress_pct = min(100, int((elapsed / HOLD_TIME) * 100))
        status_msg = f"Holding '{detected_letter}'... {progress_pct}%"

        if elapsed >= HOLD_TIME and (now - tracker["last_confirmed_time"] > COOLDOWN_TIME):
            current_word += detected_letter
            tracker["last_confirmed_time"] = now
            status_msg = f"Added '{detected_letter}' to word!"
    else:
        tracker["candidate"] = detected_letter
        tracker["first_seen"] = now

    return detected_letter, f"{conf * 100:.1f}%", current_word, current_sentence, status_msg

# Helper button actions (to finish words or clear)
def add_space(word, sentence):
    if word:
        sentence = (sentence + " " + word).strip()
    return "", sentence

def backspace(word):
    return word[:-1]

def clear_all():
    return "", ""

# Build Gradio UI
with gr.Blocks(title="Indian Sign Language Translator") as demo:
    gr.Markdown("# Indian Sign Language (ISL) Translator")
    gr.Markdown("Hold a sign steady for **1.2 seconds** to automatically add it to your word.")

    with gr.Row():
        with gr.Column(scale=3):
            cam = gr.Image(sources=["webcam"], streaming=True, label="Live Stream")
            status_box = gr.Textbox(label="Status / Instructions", value="Click Record to begin stream")

        with gr.Column(scale=2):
            sign_box = gr.Textbox(label="Detected Sign", text_align="center")
            conf_box = gr.Textbox(label="Confidence", text_align="center")
            word_box = gr.Textbox(label="Current Word", value="")
            sentence_box = gr.Textbox(label="Full Sentence", value="")

            with gr.Row():
                space_btn = gr.Button("Space (Next Word)")
                del_btn = gr.Button("Backspace")
                clear_btn = gr.Button("Clear All")

    # Wire up continuous live stream
    cam.stream(
        fn=process_stream,
        inputs=[cam, word_box, sentence_box],
        outputs=[sign_box, conf_box, word_box, sentence_box, status_box],
        stream_every=0.1
    )

    space_btn.click(fn=add_space, inputs=[word_box, sentence_box], outputs=[word_box, sentence_box])
    del_btn.click(fn=backspace, inputs=[word_box], outputs=[word_box])
    clear_btn.click(fn=clear_all, inputs=[], outputs=[word_box, sentence_box])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
