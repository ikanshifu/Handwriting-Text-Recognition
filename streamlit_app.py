import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Handwriting Recognition V9", page_icon="📝")

IMG_W = 128
IMG_H = 64

# Vocabulary
letters = (
    [' '] +
    [str(d) for d in range(10)] +
    [chr(c) for c in range(ord('A'), ord('Z')+1)] +
    [chr(c) for c in range(ord('a'), ord('z')+1)]
)

# ---------------------------------------------------------
# 2. MODEL LOADING (Simplified)
# ---------------------------------------------------------

# We still need the PatchedLSTM if the environments differ significantly,
# but usually, the clean model loads without it. We'll keep it just to be safe.
@tf.keras.utils.register_keras_serializable()
class PatchedLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        if 'time_major' in kwargs: kwargs.pop('time_major')
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_model():
    try:
        model_path = "HwTR_V9_inference.h5"  # <--- NEW FILE NAME
        
        if not os.path.exists(model_path):
            return None, f"File '{model_path}' not found on GitHub."

        # Load the clean inference model
        # We only need to map LSTM/Bidirectional just in case
        custom_objects = {
            "LSTM": PatchedLSTM,
            "Bidirectional": tf.keras.layers.Bidirectional
        }

        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False 
        )
        return model, None
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------

def add_padding(img, old_w, old_h, new_w, new_h):
    h1, h2 = int((new_h - old_h) / 2), int((new_h - old_h) / 2) + old_h
    w1, w2 = int((new_w - old_w) / 2), int((new_w - old_w) / 2) + old_w
    img_pad = np.ones([new_h, new_w, 1]) * 255
    img_pad[h1:h2, w1:w2, :] = img
    return img_pad

def fix_size(img, target_w, target_h):
    h, w = img.shape[:2]
    if w < target_w and h < target_h:
        img = add_padding(img, w, h, target_w, target_h)
    elif w >= target_w and h < target_h:
        new_w = target_w
        new_h = int(h * new_w / w)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img[..., None], new_w, new_h, target_w, target_h)
    elif w < target_w and h >= target_h:
        new_h = target_h
        new_w = int(w * new_h / h)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img[..., None], new_w, new_h, target_w, target_h)
    else:
        ratio = max(w / target_w, h / target_h)
        new_w = max(min(target_w, int(w / ratio)), 1)
        new_h = max(min(target_h, int(h / ratio)), 1)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img[..., None], new_w, new_h, target_w, target_h)
    return img

def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = fix_size(img, IMG_W, IMG_H)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    img = img.T
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = []
    for res in results:
        res = res.numpy()
        decoded_str = ""
        for p in res:
            if p != -1 and p < len(letters):
                decoded_str += letters[int(p)]
        output_text.append(decoded_str)
    return output_text[0]

# ---------------------------------------------------------
# 4. UI
# ---------------------------------------------------------

st.title("📝 Handwriting Recognition V9")

model, error = load_model()

if error:
    st.error(f"Error loading model: {error}")
    st.info("Did you upload 'HwTR_V9_inference.h5'?")
    st.stop()

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Original", width=300)
    
    if st.button("Recognize"):
        with st.spinner("Processing..."):
            try:
                processed = preprocess_image(uploaded_file)
                preds = model.predict(processed)
                text = decode_prediction(preds)
                st.success(f"**Result:** {text}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")