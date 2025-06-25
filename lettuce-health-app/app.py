import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Custom CSS for background color
st.markdown(
    """
    <style>
    .reportview-container {
        background: #2ecc71
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
model = tf.keras.models.load_model("healthy_vs_non_healthy_classifier.keras")

# Sidebar logo
st.sidebar.image("AgriVision-removebg-preview.png", use_container_width=True)

st.sidebar.title("Navigation")
pages = ["**Classifier**", "**FAQ**"]
page = st.sidebar.radio("**Go to**", pages)

# Landing page
if page == "Landing":
    try:
        st.image("AgriVision-removebg-preview.png", use_container_width=True)
    except Exception:
        st.warning("Logo image not found or invalid. Showing default logo.")
        st.markdown("<div style='font-size:180px;'>🥬</div>", unsafe_allow_html=True)

# Classifier page
if page == "**Classifier**":
    st.title("🌱 Classifier")
    st.markdown("Upload an image of a crop leaf to check if it's **Healthy** or **Non-Healthy**.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_container_width=True)
        image = image.resize((150, 150))
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
        prediction = model.predict(image_array)
        label = "Non-Healthy" if prediction[0][0] > 0.5 else "Healthy"
        confidence = prediction[0][0] if label == "Non-Healthy" else 1 - prediction[0][0]
        st.markdown(f"### 🧠 Prediction: **{label}**")
        st.markdown(f"### 📊 Confidence: **{confidence * 100:.2f}%**")
    else:
        st.markdown("### **Please upload an image to classify.**")

# FAQ page
elif page == "**FAQ**":
    st.title("🔍 Your Questions, Answered!\n")
    st.subheader("**Q: What does this app do?**")
    st.markdown("A: It classifies lettuce leaves as healthy or non-healthy using a trained deep learning model.")
    st.subheader("**Q: How do I use it?**")
    st.markdown("A: Go to the Classifier page and upload a lettuce leaf image.")
    st.subheader("**Q: What types of images are supported?**")
    st.markdown("A: JPG, JPEG, and PNG images.")

# Example Images page
elif page == "Example Images":
    st.title("📸 Example Images")
    st.markdown("### **Healthy Lettuce Leaf:**")
    st.image("example_images/healthy.png", caption="Healthy Lettuce Leaf")
    st.markdown("### **Non-Healthy Lettuce Leaf:**")
    st.image("example_images/non_healthy.png", caption="Non-Healthy Lettuce Leaf")
