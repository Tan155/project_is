import streamlit as st

st.title("Netural Network Model")

st.write("## This model used analysis people wearing glasses and no glasses")

st.write(
    "##### using data set from kaggle https://www.kaggle.com/datasets/sehriyarmemmedli/glasses-vs-noglasses-dataset"
)

st.image("network4.png")

st.write("### Feature have glasses and no glasses")

st.image("network1.png")

st.write("##### Download dataset from kaggle")
st.image("network2.png")

st.write("##### download datasets go to project which we using")

st.image("network3.png")

st.write("##### set path datasets and parameters")

st.image("network5.png")

st.write(
    "##### Prepare set data for train model, rescale, rotation, zoom, flip and so on."
)

st.image("network6.png")

st.write(
    "##### train_generator Use data for training the model, val_generator Use the data to check the accuracy of the model."
)

st.image("network7.png")


st.write(
    "##### Load the MobileNetV2 model that has already been trained and Cut out the Fully Connected Layer above."
)

st.write(
    "##### And Freeze model don't want to give The model changes previously learned values."
)

st.write("##### Add a new Fully Connected Layer Global Convert Feature Map to vector")

st.write(
    "##### Dense(1, activation='sigmoid') if this picture is 'with glasses' or 'without glasses' then create and complie model"
)

st.write("##### First round of model training")

st.image("network8.png")

st.write("##### Then we Unfreeze Layers Allow some layers to be relearned.")
st.write("##### Recompile the model reduce the Learning Rate.")
st.write("##### Train the model again around Fine-Tuning.")

st.image("network9.png")
