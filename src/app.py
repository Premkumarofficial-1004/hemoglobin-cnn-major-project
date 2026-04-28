import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import gdown

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Hemoglobin Detection", layout="centered")

st.title("🩸 Non-Invasive Hemoglobin Detection")
st.write("Capture your fingernail image using your camera")

# ---------------------------
# Download Model from GDrive
# ---------------------------
file_id = "1YJeWGQZnAn1k-RKMQidCEN-GHN9ec1MX"
model_path = "cnn_hb_model.h5"

if not os.path.exists(model_path):
    with st.spinner("Downloading model... Please wait"):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

# ---------------------------
# Camera Input
# ---------------------------
image_file = st.camera_input("📸 Capture Fingernail Image")

# ---------------------------
# Preprocessing Function
# ---------------------------
def preprocess_image(image):
    img = np.array(image)

    # Resize
    img = cv2.resize(img, (224, 224))

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE for lighting normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Normalize
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# ---------------------------
# Prediction
# ---------------------------
if image_file is not None:
    image = Image.open(image_file).convert("RGB")

    st.image(image, caption="Captured Image", use_container_width=True)

    processed = preprocess_image(image)

    prediction = model.predict(processed)

    confidence = float(prediction[0][0])
    hb_value = confidence * 20  # adjust if needed

    # Display results
    st.subheader(f"🧪 Hemoglobin: {hb_value:.2f} g/dL")

    st.progress(int(confidence * 100))
    st.write(f"Confidence: {confidence * 100:.2f}%")

    # Classification
    if hb_value < 13:
        st.error("⚠️ Result: Anemic")
    else:
        st.success("✅ Result: Non-Anemic")

    # Tips
    st.info("""
    📌 Tips for accurate results:
    - Use natural light ☀️
    - Avoid shadows
    - Keep camera steady
    - Focus on nail clearly
    """)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("⚠️ This is an AI-based estimation tool and not a medical diagnosis.")