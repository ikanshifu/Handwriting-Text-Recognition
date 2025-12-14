import streamlit as st
import os
import tempfile
from HwTR import HwTR 

# 1. CONFIGURATION
st.set_page_config(page_title="Handwriting Recognition V9", page_icon="📝")

# Configuration matching your training/inference scripts
IMG_W = 128
IMG_H = 64
MAX_TEXT_LENGTH = 16

# Vocabulary
letters = (
    [' '] +
    [str(d) for d in range(10)] +
    [chr(c) for c in range(ord('A'), ord('Z')+1)] +
    [chr(c) for c in range(ord('a'), ord('z')+1)]
)
num_classes = len(letters) + 1

# 2. MODEL LOADING

@st.cache_resource
def get_hwr_model():
    """
    Initializes the HwTR class and loads weights.
    Using cache_resource ensures we don't reload the model on every interaction.
    """
    model_path = "HwTR_V9.h5"
    
    if not os.path.exists(model_path):
        return None, f"Model file '{model_path}' not found."

    try:
        # Initialize the class (This builds the architecture internally)
        hwr = HwTR(
            img_w=IMG_W,
            img_h=IMG_H,
            max_text_length=MAX_TEXT_LENGTH,
            num_classes=num_classes,
            letters=letters
        )
        
        # Load the weights into the architecture
        # Note: Ensure your HwTR.load method uses .load_weights() or handles loading safely
        hwr.load(model_path)
        
        return hwr, None
    except Exception as e:
        return None, str(e)

# 3. UI

st.title("📝 Handwriting Recognition V9")

# Initialize the HwTR instance
hwr_instance, error = get_hwr_model()

if error:
    st.error(f"Failed to initialize model: {error}")
    st.info("Ensure 'HwTR.py' and 'HwTR_V9.h5' are in your GitHub repository.")
    st.stop()

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Input Image", width=300)
    
    if st.button("Recognize"):
        with st.spinner("Processing..."):
            # Create a temporary file because HwTR.preprocess expects a file path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # 1. Preprocess using your class method
                processed_img = hwr_instance.preprocess(tmp_path)
                
                # 2. Predict (using the internal inference model)
                preds = hwr_instance.inference_model.predict(processed_img)
                
                # 3. Decode using your class method
                result = hwr_instance.decode(preds)
                
                st.success(f"**Result:** {result}")
                
            except Exception as e:
                st.error(f"Error during recognition: {e}")
            finally:
                # Cleanup: remove the temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)