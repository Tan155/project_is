import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("fashion.keras")

# Class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("Fashion MNIST Classification Demo 👕👖👟 (FNN)")
st.write("Upload an image and let the model classify it!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28)  # Reshape for model input

    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Processing image...")

    # Prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    st.write(f"### Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")