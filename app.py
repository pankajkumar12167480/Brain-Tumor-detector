import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load trained model
MODEL_PATH = "models/model.h5"   # Updated model name
model = load_model(MODEL_PATH)

# Image size used during training
IMAGE_SIZE = 128

# Class labels (update according to your dataset folder names)
class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']  # Change if needed

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI scan image and the model will predict the tumor type.")

# File upload
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded MRI Scan", use_column_width=True)

    # Convert to model format
    img = load_img(uploaded_file, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("🔍 Prediction Result")
    st.write(f"### 🧩 Tumor Type: **{predicted_class}**")
    st.write(f"### 📌 Confidence: **{confidence:.2f}%**")

    # Optional: show all class prediction scores
    st.write("---")
    st.write("### 📊 Prediction Scores:")
    for label, score in zip(class_labels, prediction[0]):
        st.write(f"- **{label}** → {score * 100:.2f}%")
