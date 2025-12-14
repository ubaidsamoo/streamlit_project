
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# Page Config
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper for robust path finding
def get_absolute_path(relative_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if file exists relative to the script's directory (e.g. if script is in root)
    path_in_current = os.path.join(script_dir, relative_path)
    if os.path.exists(path_in_current):
        return path_in_current
    
    # Check if file exists relative to the script's parent directory (e.g. if script is in app/)
    path_in_parent = os.path.join(script_dir, "..", relative_path)
    if os.path.exists(path_in_parent):
        return path_in_parent
        
    return None

# Load CSS
def load_css(file_name):
    file_path = get_absolute_path(file_name)
    
    if file_path:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Fallback to simple relative path for logging
        st.error(f"CSS file not found: {file_name}")

load_css("assets/style.css")

# Title Section
st.markdown("""
    <div class="title-container">
        <div class="title-text">Pneumonia Detection System</div>
        <div class="subtitle-text">AI-Powered Chest X-Ray Analysis</div>
    </div>
""", unsafe_allow_html=True)

# Helper Function to Load Model
@st.cache_resource
def load_model():
    model_path = get_absolute_path("model/model.h5")
    
    if not model_path:
        st.error("Model file not found! Please check 'model/model.h5'.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Sidebar Info
with st.sidebar:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTc4KykIY998otBl-Bi3uyzI3HyC_ORykfdag&s",width=250) # Lungs icon
    st.title("System Info")
    st.info("This application uses a Convolutional Neural Network (CNN) to detect Pneumonia in Chest X-Ray images.")
    st.markdown("---")
    st.markdown("### Model Details")
    st.write("- **Architecture**: MobileNetV2 (Transfer Learning)")
    st.write("- **Input**: Chest X-Ray (Grayscale/RGB)")
    st.write("- **Classes**: Normal vs Pneumonia")
    st.markdown("---")
    st.write("- **Developed By**: M.Ubaid.Samoo")

# Main Content Layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Chest X-Ray")
    
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if file is None:
        st.info("Please upload a Chest X-Ray image to get started.")
    else:
        try:
            image = Image.open(file)
            st.image(image, caption='Uploaded X-Ray', use_column_width=True)
            
            # Prediction
            if st.button("Analyze X-Ray"):
                if model is None:
                    st.error("Model not loaded. Please checked the sidebar or logs for details.")
                else:
                    with st.spinner('Analyzing patterns...'):
                        # Preprocess
                        size = (224, 224)
                        if hasattr(Image, 'Resampling'):
                             image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                        else:
                             image_resized = ImageOps.fit(image, size, Image.ANTIALIAS)
                             
                        img_array = np.array(image_resized)
                        
                        # Ensure 3 channels (if grayscale was uploaded)
                        if len(img_array.shape) == 2:
                            img_array = np.stack((img_array,)*3, axis=-1)
                        # If 4 channels (RGBA), convert to 3
                        if img_array.shape[-1] == 4:
                            img_array = img_array[..., :3]
                            
                        img_array = img_array / 255.0  # Normalize
                        img_array = np.expand_dims(img_array, axis=0) # Batch dimension
                        
                        # Predict
                        prediction = model.predict(img_array)
                        score = prediction[0][0] 
                        
                        # Interpretation
                        # IMPORTANT: Check your training Class Indices. 
                        # Usually: {'NORMAL': 0, 'PNEUMONIA': 1}
                        # If score -> 0, it is NORMAL. If score -> 1, it is PNEUMONIA.
                        
                        labels = {0: "Normal", 1: "Pneumonia Detected"}
                        
                        if score > 0.5:
                            result = labels[1]
                            confidence = score * 100
                            css_class = "prediction-tumor" # Reusing the red style for 'Bad' news
                            sub_msg = "Signs of Pneumonia detected. Please consult a doctor."
                        else:
                            result = labels[0]
                            confidence = (1 - score) * 100
                            css_class = "prediction-safe" # Green style
                            sub_msg = "Lungs appear clear."
                        
                        # Display Result
                        st.markdown(f"""
                            <div class="prediction-box {css_class}">
                                <div class="result-text">{result}</div>
                                <div class="confidence-text">Confidence: {confidence:.2f}%</div>
                                <div style="margin-top: 10px; font-style: italic;">{sub_msg}</div>
                            </div>
                        """, unsafe_allow_html=True)
        except Exception as e:
             st.error(f"Error processing image: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 3rem; color: #64748b;">
        <p>Pneumonia Detection System © 2025</p>
        <p>Developed by M.Ubaid.Samoo</p>
    </div>
""", unsafe_allow_html=True)
