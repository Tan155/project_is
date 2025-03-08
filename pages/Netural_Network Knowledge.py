import streamlit as st
st.title("Netural Network Model")

st.write("## This model used analysis Picture Fashion MNIST")

st.write(
    "##### using data set from keras.datasets import fashion_mnist from Python"
)

st.image("fnn1.png")

st.write("#### First download data set from python and push data which we will train")

codePush = """import keras
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()"""
st.code(codePush, language="python")

codeCheckData = """print("x_train", x_train.shape, " x_test", x_test.shape)
print("y_train", y_train.shape, " y_test", y_test.shape)"""

st.write("#### x_train have 60,000 data and x_test have 10,000 data")
st.image("fnn2.png")

st.write("#### Do Normalization from 255 to 0-1")
codeNomal = """x_train = x_train / 255
x_test = x_test /  255"""
st.code(codeNomal, language="python")

st.write("#### Set Data for FNN have input 1 hidden 2 and output 1")
st.write("###### each hidden have 128 Neurons and output 10 Neurons")
codeFnn = """from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=[28,28]))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()"""

st.code(codeFnn, language="python")
st.image("fnn3.png")

st.write("#### Train data 5 epochs")

codeTrain = """model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
epochs = 5
history = model.fit(x_train, y_train, epochs=epochs)"""

st.code(codeTrain, language="python")

st.image("fnn4.png")

st.image("fnn5.png")

st.write("#### accuracy high more than 80% and loss less than 40%")

import streamlit as st
from PIL import Image
import io

st.write("#### Click Image to Download 📥")

# Predefined image paths
image_files = [
    "sample_fashion.png","sample_fashion (1).png", "sample_fashion (2).png", "sample_fashion (3).png",
    "sample_fashion (4).png", "sample_fashion (5).png", "sample_fashion (6).png",
    "sample_fashion (7).png", "sample_fashion (8).png", "sample_fashion (9).png",
    "sample_fashion (10).png"
]

# Display images and create download buttons
for image_file in image_files:
    image = Image.open(image_file)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    st.image(image, caption=image_file, width=100)
    st.download_button(label=f"Download {image_file}", data=img_bytes, file_name=image_file, mime="image/png")