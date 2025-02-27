import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# โหลดโมเดล TFLite
tflite_model_path = "glasses_model.tflite"

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_size = (224, 224)

st.title("🕶️ MobileNetV2: Glasses Detection ")
st.write("**Upload a photo to verify that Does the person wear glasses?**")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Download and display images
    img = load_img(uploaded_file, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # make a forecast
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]

    if prediction > 0.5:
        st.write("### **Wear glasses** 🤓")
    else:
        st.write("### **Not wearing glasses** 😎")
