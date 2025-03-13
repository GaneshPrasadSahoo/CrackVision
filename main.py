import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"D:\All_DataSet2\Google Colab Model\Crack Dection\crack_detection_resnet.keras")  # Update with your model file
    return model

model = load_model()

# Define class labels
CLASS_NAMES = ["Negative (Not Crack)", "Positive (Crack)"]

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match model input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Crack Detection Classifier")
st.write("Upload an image to classify whether it has a crack or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0]  # Get raw output

        # Check if model outputs probability or class indexes
        if len(prediction) == 1:  # Binary classification (sigmoid output)
            crack_prob = prediction[0]
            predicted_class = "Positive (Crack)" if crack_prob > 0.5 else "Negative (Not Crack)"
            confidence = crack_prob if crack_prob > 0.5 else 1 - crack_prob
        else:  # If using softmax (unlikely for binary case)
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = np.max(prediction)

        st.write(f"**Prediction:** {predicted_class} (Confidence: {confidence:.2f})")
