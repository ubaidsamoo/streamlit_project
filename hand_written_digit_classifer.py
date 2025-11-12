import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple
from tensorflow.keras.models import load_model


# -------------------------------
# Config
# -------------------------------
MODEL_CANDIDATES = [
    "digit_model.h5",
    "path_to_your_trained_model.h5",
    "model.h5",
]


@st.cache_resource
def load_digit_model(model_paths: list = MODEL_CANDIDATES):
    """Try to load the first available model from model_paths.

    Returns the loaded model or raises a RuntimeError if none found.
    """
    last_exc = None
    for p in model_paths:
        try:
            model = load_model(p)
            return model
        except Exception as e:
            last_exc = e
            # continue to next candidate
    raise RuntimeError(
        "No Keras model found. Tried: {}. Last error: {}".format(
            ", ".join(model_paths), str(last_exc)
        )
    )


def preprocess_image(image: Image.Image, cnn_mode: bool = True) -> Tuple[Image.Image, np.ndarray]:
    """Convert image to 28x28 grayscale and return (display_image, model_input_array)."""
    img = image.convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0

    # If background is white (typical photo), invert so digit is bright
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    if cnn_mode:
        arr = arr.reshape(1, 28, 28, 1)
    else:
        arr = arr.reshape(1, 784)

    return img, arr


def predict(model, input_array: np.ndarray) -> np.ndarray:
    """Return prediction probabilities (shape (1,10))."""
    preds = model.predict(input_array)
    return preds


def plot_probabilities(preds: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 2.2))
    probs = preds[0]
    ax.bar(range(10), probs, color="#4c78a8")
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    for i, v in enumerate(probs):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="🖊️ Handwritten Digit Classifier", page_icon="✏️", layout="wide")
    st.title("🧠 Handwritten Digit Classifier — Refactored")

    # Sidebar
    st.sidebar.header("Settings")
    model_choice = st.sidebar.selectbox("Model file to try", options=MODEL_CANDIDATES)
    use_cnn = st.sidebar.checkbox("Model is CNN (28x28x1)", value=True)
    show_raw_pixels = st.sidebar.checkbox("Show 28x28 pixel preview", value=False)

    # Load model (attempt)
    try:
        model = load_digit_model([model_choice])
    except RuntimeError as e:
        st.sidebar.error(str(e))
        st.error("Model not available. Please place a trained Keras .h5 model in the app folder and select it in the sidebar.")
        st.stop()

    st.write("Upload a handwritten digit image (PNG/JPG) or use the sample below.")

    sample_col, input_col = st.columns([1, 2])

    with sample_col:
        if st.button("Use sample digit (0)"):
            # create a very simple synthetic digit 0 (circle) as an example
            sample = Image.new("L", (28, 28), color=255)
            sample_arr = np.array(sample)
            yy, xx = np.ogrid[:28, :28]
            mask = (xx - 14) ** 2 + (yy - 14) ** 2
            sample_arr[mask < 60] = 0
            uploaded_image = Image.fromarray(sample_arr).convert("RGB")
        else:
            uploaded_image = None

    with input_col:
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file).convert("RGB")

    if uploaded_image is None:
        st.info("👆 Upload an image or pick the sample to get a prediction.")
        return

    display_img, model_input = preprocess_image(uploaded_image, cnn_mode=use_cnn)

    # Prediction
    preds = predict(model, model_input)
    predicted = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100.0

    # Layout results
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Uploaded")
        st.image(uploaded_image, use_column_width=True)
        st.subheader("Processed 28x28")
        st.image(display_img, width=140)
        if show_raw_pixels:
            st.write(np.array(display_img))

    with c2:
        st.subheader("Prediction")
        st.markdown(f"**Digit:** {predicted}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.subheader("Probabilities")
        fig = plot_probabilities(preds)
        st.pyplot(fig, clear_figure=True)


if __name__ == "__main__":
    main()