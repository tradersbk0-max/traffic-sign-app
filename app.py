import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("traffic_sign_model.keras")

st.title("Traffic Sign Recognition App 🚦")

uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img = np.array(image)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    
    st.write("### Predicted Class ID:", class_id)