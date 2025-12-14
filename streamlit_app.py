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
num_classes = len(letters) + 1

# ---------------------------------------------------------
# 2. MODEL RECONSTRUCTION
# ---------------------------------------------------------

def build_inference_model():
    """Manually creates the model structure."""
    input_data = tf.keras.layers.Input(name='input', shape=(IMG_W, IMG_H, 1), dtype='float32')

    # --- CNN Block ---
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', name='conv1')(input_data)
    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2), name='pool1')(x)

    x = tf.keras.layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', name='conv2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2), name='pool2')(x)

    x = tf.keras.layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', name='conv3')(x)
    x = tf.keras.layers.BatchNormalization(name='bn3')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', name='conv4')(x)
    x = tf.keras.layers.BatchNormalization(name='bn4')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((1,2), name='pool3')(x)

    x = tf.keras.layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', name='conv5')(x)
    x = tf.keras.layers.BatchNormalization(name='bn5')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', name='conv6')(x)
    x = tf.keras.layers.BatchNormalization(name='bn6')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((1,2), name='pool4')(x)
    
    x = tf.keras.layers.Conv2D(512, (2,2), padding='same', kernel_initializer='he_normal', name='conv7')(x)
    x = tf.keras.layers.BatchNormalization(name='bn7')(x)
    x = tf.keras.layers.Activation('relu')(x)

    # --- Reshape & Dense ---
    target_shape = (32, 2048) 
    x = tf.keras.layers.Reshape(target_shape=target_shape, name='reshape')(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(x)
    
    # --- RNN ---
    lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.25), name='lstm1')(x)
    lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.25), name='lstm2')(lstm1)

    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='softmax')(lstm2)
    model = tf.keras.models.Model(inputs=input_data, outputs=output)
    return model

@st.cache_resource
def load_model_weights():
    model_path = "HwTR_V9.h5"
    if not os.path.exists(model_path):
        return None, f"Model file '{model_path}' not found."
    
    try:
        model = build_inference_model()
        # load_weights only loads the numbers, ignoring file structure issues
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        return model, None
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------
# 3. PREPROCESSING (FIXED: Grayscale First)
# ---------------------------------------------------------

def add_padding(img, old_w, old_h, new_w, new_h):
    h1, h2 = int((new_h - old_h) / 2), int((new_h - old_h) / 2) + old_h
    w1, w2 = int((new_w - old_w) / 2), int((new_w - old_w) / 2) + old_w
    
    # Create a 2D White Canvas (since image is grayscale)
    img_pad = np.ones([new_h, new_w]) * 255
    img_pad[h1:h2, w1:w2] = img
    return img_pad

def fix_size(img, target_w, target_h):
    h, w = img.shape[:2]
    # No more [..., None] here, we work with simple 2D arrays
    if w < target_w and h < target_h:
        img = add_padding(img, w, h, target_w, target_h)
    elif w >= target_w and h < target_h:
        new_w = target_w
        new_h = int(h * new_w / w)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    elif w < target_w and h >= target_h:
        new_h = target_h
        new_w = int(w * new_h / h)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    else:
        ratio = max(w / target_w, h / target_h)
        new_w = max(min(target_w, int(w / ratio)), 1)
        new_h = max(min(target_h, int(h / ratio)), 1)
        new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    return img

def preprocess_image(uploaded_file):
    # 1. Load Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 2. Convert to Grayscale IMMEDIATELY
    # This prevents the shape mismatch error (3 channels vs 1 channel)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Resize & Pad
    img = fix_size(img, IMG_W, IMG_H)
    
    # 4. Normalize
    img = img.astype(np.float32) / 255.0
    
    # 5. Transpose & Expand
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
        decoded = ""
        for p in res:
            if p != -1 and p < len(letters):
                decoded += letters[int(p)]
        output_text.append(decoded)
    return output_text[0]

# ---------------------------------------------------------
# 4. UI
# ---------------------------------------------------------

st.title("📝 Handwriting Recognition V9")

model, error = load_model_weights()

if error:
    st.error(f"Failed to load model: {error}")
    st.stop()

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Input", width=300)
    if st.button("Recognize"):
        with st.spinner("Analyzing..."):
            try:
                processed = preprocess_image(uploaded_file)
                preds = model.predict(processed)
                text = decode_prediction(preds)
                st.success(f"**Result:** {text}")
            except Exception as e:
                st.error(f"Error: {e}")