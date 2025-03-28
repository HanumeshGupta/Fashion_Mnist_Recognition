import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

load_model = load_model(r'D:\Visual Studio Code\ML\Youtube_ML_Code\Fashion Mnist Recognition\Fashion_MNIST_model.h5')

Class_name = {0:'T-shirt/top',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle boot'}

#Conveting the Image
def img_processing(img_path):
    img = Image.open(img_path)
    img = img.resize((28,28))
    img = img.convert('L')
    img = np.array(img) / 255.0
    img = img.reshape((1,28,28,1))
    return img



st.title("Fashion Item Classifier")


upload_img = st.file_uploader("Upload as Image.... ", type=['jpg','jpge','png'])


if upload_img is not None:
    image = Image.open(upload_img)
    col1 , col2 = st.columns(2)


    with col1 :
        resize_img = image.resize((100,100))
        st.image(resize_img)

    with col2 :
        if st.button("Classify"):

            img = img_processing(upload_img)

            result = load_model.predict(img)

            pred = np.argmax(result)
            prediction = Class_name[pred]

            st.success(f"Prediction : {prediction}")
