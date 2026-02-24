import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Load trained model
# ---------------------------
model = tf.keras.models.load_model("plant_disease_model.keras")

class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

treatments = {
    "Blight": "🌿 Apply neem oil spray weekly. Remove infected leaves and improve air circulation.",
    "Common_Rust": "🍂 Use organic sulfur spray. Practice crop rotation and proper spacing.",
    "Gray_Leaf_Spot": "🌾 Apply bio-fungicide (Bacillus subtilis). Remove infected crop debris.",
    "Healthy": "🌱 Plant is healthy. Maintain proper watering and balanced fertilization."
}

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌿 AI-Based Plant Disease Detection System")
st.write("Upload a leaf image to detect disease and get eco-friendly treatment suggestions.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    # Preprocess image
    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("🌿 Eco-Friendly Treatment Suggestion")
    st.write(treatments[predicted_class])