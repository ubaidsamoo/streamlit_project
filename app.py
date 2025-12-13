import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Global Constants ---
MODEL_PATH = 'model.h5'
IMG_SIZE = (224, 224)

# --- Custom Styling (Premium Aesthetics) ---
st.markdown("""
    <style>
        /* General App Styling */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #ffffff;
            font-weight: 700;
            text-align: center;
            text-shadow: 0px 4px 10px rgba(0,0,0,0.5);
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 20px;
            background: linear-gradient(90deg, #ff8a00 0%, #e52e71 100%);
            color: white;
            font-weight: bold;
            border: none;
            padding: 10px 24px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(229, 46, 113, 0.4);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(229, 46, 113, 0.6);
        }

        /* File Uploader */
        [data-testid="stFileUploader"] {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 2rem;
            border-radius: 12px;
            border: 2px dashed rgba(255, 255, 255, 0.2);
            text-align: center;
        }

        /* Result Cards */
        .result-card {
            background-color: rgba(0, 0, 0, 0.6);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            margin-top: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        .confidence-label {
            font-size: 1.2rem;
            color: #b0bec5;
        }
        .prediction-label {
            font-size: 2rem;
            font-weight: bold;
            margin: 10px 0;
        }
        .tumor { color: #ff5252; text-shadow: 0 0 10px rgba(255, 82, 82, 0.5); }
        .no-tumor { color: #69f0ae; text-shadow: 0 0 10px rgba(105, 240, 174, 0.5); }
        
    </style>
""", unsafe_allow_html=True)

# --- Functions ---

# Removed @st.cache_resource to ensure fresh model load and avoid stale states during development
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_image(image):
    """
    Prepares the image for the model (Resize, Scale).
    """
    # Resize to match model input (standard resize to avoid cropping edges where tumor implies)
    image = image.resize(IMG_SIZE)
    
    # Convert to array
    img_array = np.array(image)
    
    # Expand dims to create batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess (Scale [-1, 1] for MobileNetV2)
    img_array = preprocess_input(img_array)
    
    return img_array

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn3d.iconscout.com/3d/premium/thumb/big-brain-tumor-3d-icon-png-download-9841382.png", width=100)
    st.title("About My Project")
    
    # Sensitivity Slider
    THRESHOLD = st.slider("Detection Sensitivity", 0.0, 1.0, 0.5, 0.05, help="Adjust to fine-tune detection. Lower values make it more sensitive to finding tumors.")
    st.markdown("---")
    
    st.info(
        """
        **Brain Tumor Detection System**
        
        This application analyses MRI scans to detect the presence of brain tumors.
        
        **Model**: MobileNetV2 (Transfer Learning)
        **Accuracy**: Optimized for high precision.
        
        *Built for Deep Learning Assignment.*
        """
    )
    st.markdown("---")
    st.write("Developed by **M.UBAID SAMOO**")

# --- Main Interface ---
st.title("🧠 Brain Tumor Detection System")
st.markdown("### MRI Analysis")
st.markdown("Upload a Brain MRI Image to get an instant analysis.")

# Model Loading
model = load_trained_model()

if model is None:
    st.error("⚠️ Model file not found!")
    st.markdown(
        f"""
        Please ensure `model.h5` is present in the directory. 
        You can train the model by running:
        
        ```bash
        python train_model.py
        ```
        """
    )
else:
    # File Uploader
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Preview")
            image = Image.open(file).convert('RGB')
            st.image(image, use_column_width=True, caption="Uploaded MRI")
        
        with col2:
            st.markdown("#### Analysis")
            
            if st.button("Detect Tumor"):
                with st.spinner("Analyzing Image..."):
                    # Preprocess
                    processed_img = process_image(image)
                    
                    # Predict
                    prediction = model.predict(processed_img)
                    score = prediction[0][0]  # Probability of class 1
                    
                    # Assuming Class 1 = Tumor (based on directory structure 'yes')
                    # And Class 0 = No Tumor ('no')
                    
                    # If using 'yes' and 'no' folders, 'yes' usually comes alphabetically second?
                    # wait: 'no', 'yes' -> ['no', 'yes'] -> index 0: no, index 1: yes.
                    # So score close to 1 is Yes (Tumor).
                    
                    # Logic using Dynamic Threshold
                    if score > THRESHOLD:
                        label = "Tumor Detected"
                        confidence = score * 100
                        css_class = "tumor"
                        message = "The model has detected patterns consistent with a brain tumor."
                    else:
                        label = "No Tumor"
                        confidence = (1 - score) * 100
                        css_class = "no-tumor"
                        message = "The model did not detect any tumor patterns."
                    
                    # Display Results
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <p class="confidence-label">Confidence</p>
                            <h2 style="color:white;">{confidence:.2f}%</h2>
                            <p class="prediction-label {css_class}">{label}</p>
                            <p style="font-size: 0.9rem; margin-top: 10px; color: #ccc;">{message}</p>
                            <p style="font-size: 0.8rem; color: #555;">(Threshold: {THRESHOLD})</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Progress Bar
                    val = score if score > THRESHOLD else (1-score)
                    st.progress(float(val))

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        <p>Disclaimer: This tool is for educational purposes only and should not be used for medical diagnosis.</p>
    </div>
    """, 
    unsafe_allow_html=True
)
