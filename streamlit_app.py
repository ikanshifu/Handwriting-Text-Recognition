import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os

# ---------------------------------------------------------
# 1. CONFIGURATION & CONSTANTS
# ---------------------------------------------------------
st.set_page_config(page_title="Handwriting Recognition", page_icon="📝")

IMG_W = 128
IMG_H = 64

# Vocabulary (Must match your training exactly)
letters = (
    [' '] +
    [str(d) for d in range(10)] +
    [chr(c) for c in range(ord('A'), ord('Z')+1)] +
    [chr(c) for c in range(ord('a'), ord('z')+1)]
)

# ---------------------------------------------------------
# 2. CUSTOM LAYERS & MODEL LOADING
# ---------------------------------------------------------

# A. Register CTCLayer so Keras knows what it is
@tf.keras.utils.register_keras_serializable()
class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        return y_pred
    
    def get_config(self):
        return super().get_config()

# B. Register PatchedLSTM to fix the "time_major" error in Keras 3
@tf.keras.utils.register_keras_serializable()
class PatchedLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        # Remove the 'time_major' argument that crashes Keras 3
        if 'time_major' in kwargs:
            kwargs.pop('time_major')
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_trained_model():
    """Loads the model with fixes for Keras 3 compatibility."""
    try:
        model_path = "HwTR_V4.h5" 
        
        if not os.path.exists(model_path):
            return None, "Model file 'HwTR_V4.h5' not found."

        # Map the old class names to our patched versions
        custom_objects = {
            "CTCLayer": CTCLayer,
            "LSTM": PatchedLSTM,
            "Bidirectional": tf.keras.layers.Bidirectional
        }

        # ---------------------------------------------------------
        # THE FIX: safe_mode=False allows the Lambda layer to load
        # ---------------------------------------------------------
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False 
        )
        
        # Extract Inference Model (Input -> Softmax)
        image_input = model.get_layer("input").input
        output_layer = model.get_layer("softmax").output
        prediction_model = tf.keras.models.Model(image_input, output_layer)
        
        return prediction_model, None
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=-1)
    img = fix_size(img, IMG_W, IMG_H)
    img = img.astype(np.float32)
    img /= 255.0
    img = np.transpose(img, (1, 0, 2)) # Transpose to (128, 64, 1)
    img = np.expand_dims(img, axis=0)  # Batch dim -> (1, 128, 64, 1)
    return img

def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = []
    for res in results:
        res = res.numpy()
        decoded = ""
        for p in res:
            if p != -1 and p < len(letters):
                 decoded += letters[int(p)]
        output_text.append(decoded)
    return output_text[0]

# ---------------------------------------------------------
# 4. APP UI
# ---------------------------------------------------------

st.title("📝 CNN-BiLSTM Handwriting Reader")
st.markdown("Upload an image of a single word (alphanumeric), and the model will transcribe it.")

model, error = load_trained_model()

if error:
    st.error(f"Failed to load model: {error}")
    st.stop()

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    
    if st.button("Recognize Text"):
        with st.spinner("Processing..."):
            try:
                processed_img = preprocess_image(uploaded_file)
                preds = model.predict(processed_img)
                text = decode_prediction(preds)
                st.success(f"**Predicted Text:** {text}")
                
                with st.expander("Debug Info"):
                    st.write(f"Processed Input Shape: {processed_img.shape}")
                    debug_img = np.transpose(processed_img[0], (1, 0, 2))
                    st.image(debug_img, clamp=True, width=300)
            except Exception as e:
                st.error(f"Error during processing: {e}")