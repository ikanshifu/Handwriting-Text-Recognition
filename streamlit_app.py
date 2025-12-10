import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# a. Initial Configuration
st.set_page_config(page_title="Handwriting Recognition", page_icon="📝")

# Dimensions defined in your notebook
IMG_W = 128
IMG_H = 64

# Vocabulary defined in your notebook (Snippet 40)
# [' '] + Digits + Uppercase + Lowercase
letters = (
    [' '] +
    [str(d) for d in range(10)] +
    [chr(c) for c in range(ord('A'), ord('Z')+1)] +
    [chr(c) for c in range(ord('a'), ord('z')+1)]
)
num_classes = len(letters) + 1

# Custom Layer and Model Loadings

# We must define the custom layer class so Keras can load the model file
@tf.keras.utils.register_keras_serializable()
class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        return y_pred
    
    def get_config(self):
        return super().get_config()

# 2. Register PatchedLSTM (The fix for 'time_major')
@tf.keras.utils.register_keras_serializable()
class PatchedLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        # The 'time_major' argument exists in Keras 2 but was removed in Keras 3.
        # We catch it here and throw it away so Keras 3 doesn't crash.
        if 'time_major' in kwargs:
            kwargs.pop('time_major')
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_trained_model():
    try:
        model_path = "HwTR_V4.h5" 
        
        if not os.path.exists(model_path):
            return None, "Model file 'HwTR_V4.h5' not found."

        # Map the old "LSTM" to our new "PatchedLSTM"
        custom_objects = {
            "CTCLayer": CTCLayer,
            "LSTM": PatchedLSTM,
            "Bidirectional": tf.keras.layers.Bidirectional
        }

        # compile=False is crucial here to avoid other loss-function errors
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False 
        )
        
        # Extract the inference part (Input -> Softmax)
        image_input = model.get_layer("input").input
        output_layer = model.get_layer("softmax").output
        prediction_model = tf.keras.models.Model(image_input, output_layer)
        
        return prediction_model, None
    except Exception as e:
        return None, str(e)
    
# b.Preprocessing

def add_padding(img, old_w, old_h, new_w, new_h):
    h1, h2 = int((new_h - old_h) / 2), int((new_h - old_h) / 2) + old_h
    w1, w2 = int((new_w - old_w) / 2), int((new_w - old_w) / 2) + old_w
    img_pad = np.ones([new_h, new_w, 1]) * 255
    img_pad[h1:h2, w1:w2, :] = img
    return img_pad

def fix_size(img, target_w, target_h):
    h, w = img.shape[:2]
    
    # Resize logic to preserve aspect ratio
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
        # w >= target_w and h >= target_h
        ratio = max(w / target_w, h / target_h)
        new_w = max(min(target_w, int(w / ratio)), 1)
        new_h = max(min(target_h, int(h / ratio)), 1)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img[..., None], new_w, new_h, target_w, target_h)
    
    return img

def preprocess_image(uploaded_file):
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 1. Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Expand dims to match (H, W, 1) for processing
    img = np.expand_dims(img, axis=-1)
    
    # 3. Fix Size (Pad/Resize)
    img = fix_size(img, IMG_W, IMG_H)
    
    # 4. Normalize
    img = img.astype(np.float32)
    img /= 255.0
    
    # 5. Transpose to (Width, Height, Channel) for the model
    # Your model input shape is (128, 64, 1)
    img = np.transpose(img, (1, 0, 2))
    
    # 6. Add Batch Dimension -> (1, 128, 64, 1)
    img = np.expand_dims(img, axis=0)
    
    return img

def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    
    # CTC Decode
    # The model output is (Batch, Time, Classes). We use greedy search.
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    
    # Convert indices to text
    output_text = []
    for res in results:
        res = res.numpy()
        decoded = ""
        for p in res:
            if p != -1 and p < len(letters): # -1 is the blank token in some implementations
                 decoded += letters[int(p)]
        output_text.append(decoded)
        
    return output_text[0]


# c. APP UI
st.title("📝OCR Based Handwriting Reader For Cursive Writing")
st.markdown("Please upload an image of a word written on a piece of paper, and the model will transcribe it.")

# Load Model
model, error = load_trained_model()

if error:
    st.error(f"Failed to load model: {error}")
    st.warning("Please ensure your training notebook saved the model as 'HwTR_V4.h5' and it is in this folder.")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    
    if st.button("Recognize Text"):
        with st.spinner("Processing..."):
            try:
                # Preprocess
                processed_img = preprocess_image(uploaded_file)
                
                # Predict
                preds = model.predict(processed_img)
                
                # Decode
                text = decode_prediction(preds)
                
                st.success(f"**Predicted Text:** {text}")
                
                # Optional: Show debug info
                with st.expander("Debug Info"):
                    st.write(f"Processed Input Shape: {processed_img.shape}")
                    st.write("Input Image to Model (Transposed):")
                    # Visualize what the model sees (Transposed back for viewing)
                    debug_img = np.transpose(processed_img[0], (1, 0, 2))
                    st.image(debug_img, clamp=True, width=300)
                    
            except Exception as e:
                st.error(f"Error during processing: {e}")