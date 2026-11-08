import pathlib

import streamlit as st
from fastai.vision.all import *

st.markdown('<h1 style="color: yellow; font-weight: 700;">Cats vs Dogs</h1>',unsafe_allow_html=True,)
st.text("Created by Kendrix")

uploaded_file = st.file_uploader(label="Choose an image file...", type=["jpg", "jpeg", "png"])


if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

def cat_or_dog(file_name):
    if file_name[0].isupper():
        return "CAt"
    else:
        return "DOG"

model = load_learner("cat_vs_dog_model_fastai2_7_19.pkl")

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    resized_img = img.resize((224, 224), Image.NEAREST)
    prediction = model.predict(resized_img)
    st.text(prediction)

    index = int(prediction[1])
    confidence = prediction[2][index] * 100

    if confidence > 90:
        label = f"I am {confidence}% sure that this is a {prediction[0]}"
    else:
        label = f"I am not sure, but I am {confidence}% sure that this is a {prediction[0]}"

    st.text(label)
    st.image(uploaded_file)