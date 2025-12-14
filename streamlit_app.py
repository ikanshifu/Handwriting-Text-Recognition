import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os

# ---------------------------------------------------------
# 1. CONFIGURATION & CONSTANTS
# ---------------------------------------------------------
st.set_page_config(page_title="Handwriting Recognition V9", page_icon="📝")

# Constants from your HwTR.py / inference.py
IMG_W = 128
IMG_H = 64

# Vocabulary from inference.py
letters = (
    [' '] +
    [str(d) for d in range(10)] +
    [chr(c) for c in range(ord('A'), ord('Z')+1)] +
    [chr(c) for c in range(ord('a'), ord('z')+1)]
)

# ---------------------------------------------------------
# 2. CUSTOM LAYERS & COMPATIBILITY
# ---------------------------------------------------------

# We need to register the custom layer so Keras knows how to load it
@tf.keras.utils.register_keras_serializable()
class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        return y_pred
    
    def get_config(self):
        return super().get_config()

# Safety patch for LSTM compatibility between Keras versions
@tf.keras.utils.register_keras_serializable()
class PatchedLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        if 'time_major' in kwargs:
            kwargs.pop('time_major')
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_trained_model():
    """Loads HwTR_V9.h5 with Keras 3 compatibility fixes."""
    try:
        model_path = "HwTR_V9.h5"  # UPDATED MODEL NAME
        
        if not os.path.exists(model_path):
            return None, f"Model file '{model_path}' not found. Please upload it."

        # Map custom objects
        custom_objects = {
            "CTCLayer": CTCLayer,
            "LSTM": PatchedLSTM, # Maps LSTM to our patched version
            "Bidirectional": tf.keras.layers.Bidirectional
        }

        # Load model with safe_mode=False to allow Lambda layers
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False 
        )
        
        # Extract the Inference Model (Input -> Softmax)
        # Based on HwTR.py, the output layer is named 'softmax'
        try:
            image_input = model.get_layer("input").input
            output_layer = model.get_layer("softmax").output
            prediction_model = tf.keras.models.Model(image_input, output_layer)
        except:
            # Fallback if the saved model is already the inference model
            prediction_model = model
            
        return prediction_model, None
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------
# 3. PREPROCESSING (Ported from HwTR.py)
# ---------------------------------------------------------

def add_padding(img, old_w, old_h, new_w, new_h):
    # Logic from HwTR.py
    h1, h2 = int((new_h - old_h) / 2), int((new_h - old_h) / 2) + old_h
    w1, w2 = int((new_w - old_w) / 2), int((new_w - old_w) / 2) + old_w
    img_pad = np.ones([new_h, new_w, 1]) * 255
    img_pad[h1:h2, w1:w2, :] = img
    return img_pad

def fix_size(img, target_w, target_h):
    # Logic from HwTR.py
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
    # Convert Streamlit file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Preprocessing pipeline from HwTR.py
    # 1. Fix Size (Resize + Pad)
    img = fix_size(img, IMG_W, IMG_H)
    
    # 2. Clip and Grayscale
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Normalize
    img = img.astype(np.float32) / 255.0
    
    # 4. Transpose and Expand Dims
    img = img.T                        # Transpose to (128, 64)
    img = np.expand_dims(img, axis=-1) # (128, 64, 1)
    img = np.expand_dims(img, axis=0)  # Batch dim -> (1, 128, 64, 1)
    
    return img

def decode_prediction(pred):
    # Decoding logic from HwTR.py
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    
    # CTC Decode
    results = tf.keras.backend.ctc_decode(
        pred, 
        input_length=input_len, 
        greedy=True
    )[0][0]
    
    # Convert indices to text
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
# 4. APP UI
# ---------------------------------------------------------

st.title("📝 Handwriting Recognition (Model V9)")
st.markdown("Upload an image of an alphanumeric word.")

# Load Model
model, error = load_trained_model()

if error:
    st.error(f"Failed to load model: {error}")
    st.info("Make sure 'HwTR_V9.h5' is uploaded to your GitHub repository.")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Original Image", width=300)
    
    if st.button("Recognize Text"):
        with st.spinner("Processing..."):
            try:
                # Preprocess
                processed_img = preprocess_image(uploaded_file)
                
                # Predict
                preds = model.predict(processed_img)
                
                # Decode
                text = decode_prediction(preds)
                
                st.success(f"**Result:** {text}")
                
                # Debugging view
                with st.expander("See what the model sees"):
                    # Transpose back for visualization: (1, 128, 64, 1) -> (64, 128)
                    debug_img = processed_img[0, :, :, 0].T
                    st.image(debug_img, caption="Preprocessed Input", width=300, clamp=True)
                    
            except Exception as e:
                st.error(f"Error: {e}")