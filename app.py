import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("healthy_vs_non_healthy_classifier.keras")

st.set_page_config(page_title="Lettuce Health Classifier", layout="centered")
st.title("🥬 Lettuce Health Classifier")
st.markdown("Upload an image of a lettuce leaf to check if it's **Healthy** or **Non-Healthy**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image = image.resize((150, 150))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(image_array)
    label = "Non-Healthy" if prediction[0][0] > 0.5 else "Healthy"
    confidence = prediction[0][0] if label == "Non-Healthy" else 1 - prediction[0][0]
    st.markdown(f"### 🧠 Prediction: **{label}**")
    st.markdown(f"### 📊 Confidence: **{confidence * 100:.2f}%**")
