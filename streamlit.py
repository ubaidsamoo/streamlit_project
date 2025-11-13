import streamlit as st
import numpy as np
import glob
import os
from PIL import Image
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    plt = None
    _HAS_MATPLOTLIB = False
from typing import Tuple
try:
    from tensorflow.keras.models import load_model  # type: ignore
    _HAS_TF = True
except Exception:
    try:
        from keras.models import load_model  # type: ignore
        _HAS_TF = True
    except Exception:
        load_model = None
        _HAS_TF = False


# Lightweight deterministic demo model used when no .h5 is provided.
class DummyModel:
    """Simple deterministic predictor for UI/demo purposes only.

    The prediction is derived from the mean pixel intensity so results
    are consistent for the same input and give a plausible single-digit
    peak probability distribution.
    """
    def predict(self, input_array: np.ndarray) -> np.ndarray:
        # input_array expected shape: (1, 28, 28, 1) or (1, 784)
        m = float(np.mean(input_array))
        # map mean (0..1) to digit 0..9 (darker -> higher digit)
        pred_digit = int(np.clip(round((1.0 - m) * 9), 0, 9))
        probs = np.full(10, 0.01, dtype=np.float32)
        probs[pred_digit] = 0.9
        probs = probs / probs.sum()
        return probs.reshape(1, 10)


# -------------------------------
# Config
# -------------------------------
def find_model_candidates() -> list:
    """Return a sorted list of .h5 model files in the app folder."""
    try:
        file_dir = os.path.dirname(__file__)
    except NameError:
        file_dir = ""
    cwd = file_dir or os.getcwd()
    files = sorted(glob.glob(os.path.join(cwd, "*.h5")))
    # Return basenames for nicer display
    return [os.path.basename(f) for f in files]


@st.cache_resource
def load_digit_model(model_paths: list):
    """Try to load the first available model from model_paths.

    Returns the loaded model or raises a RuntimeError if none found.
    """
    if load_model is None:
        raise RuntimeError(
            "No Keras/TensorFlow loader available. Install TensorFlow or Keras: `pip install tensorflow` or `pip install keras`."
        )

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
    # If matplotlib isn't available, return the raw probability vector
    if not _HAS_MATPLOTLIB:
        return preds[0]

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
    model_files = find_model_candidates()
    if model_files:
        model_choice = st.sidebar.selectbox("Model file to try", options=model_files)
        demo_mode_active = False
    else:
        # Silently fall back to demo predictor when no model files exist.
        model_choice = None
        demo_mode_active = True
    use_cnn = st.sidebar.checkbox("Model is CNN (28x28x1)", value=True)
    show_raw_pixels = st.sidebar.checkbox("Show 28x28 pixel preview", value=False)

    # Load model (attempt). If loading fails (missing TF or .h5 file), fall back
    # to the lightweight DummyModel automatically so the UI remains usable.
    # If we already decided demo mode is active (no files), use DummyModel silently.
    if demo_mode_active:
        model = DummyModel()
    else:
        try:
            model = load_digit_model([model_choice])
            if model is None:
                st.sidebar.warning("Model loader returned no model; using demo predictor.")
                model = DummyModel()
        except RuntimeError as e:
            # Notify once in sidebar and fall back to demo model.
            st.sidebar.warning(str(e))
            model = DummyModel()

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
        st.image(uploaded_image, use_container_width=True)
        st.subheader("Processed 28x28")
        st.image(display_img, width=140)
        if show_raw_pixels:
            st.write(np.array(display_img))

    with c2:
        st.subheader("Prediction")
        st.markdown(f"**Digit:** {predicted}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.subheader("Probabilities")
        fig_or_probs = plot_probabilities(preds)
        if _HAS_MATPLOTLIB and fig_or_probs is not None:
            st.pyplot(fig_or_probs, clear_figure=True)
        else:
            # Fallback: use Streamlit's bar chart when matplotlib is not available
            probs = fig_or_probs if fig_or_probs is not None else preds[0]
            # Convert to a 2D structure that st.bar_chart accepts
            st.bar_chart(np.array([probs]), height=220)


if __name__ == "__main__":
    main()
