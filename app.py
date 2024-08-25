import streamlit as st
import requests
import gdown
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout

@st.cache_resource(ttl = 60*60 *24 *7, show_spinner="Fetching model from cloud...")
def load_model_from_google_drive(fileid, save_path):
    # Download and Save Model
    weights_url = f'https://drive.google.com/uc?id={fileid}'
    gdown.download(weights_url, save_path, quiet=True)
    # Load and Return Model
    try:
        model_gd = tf.keras.models.load_model(save_path)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
    return model_gd

def preprocess_image(img, img_size=(224,224)):
    image = Image.open(img)
    image = image.resize(img_size)
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image,image_array


## Load the Model from Google Drive
save_path = 'model.h5'
google_drive_file_id = st.secrets["GOOGLE_DRIVE_FILE_ID"]
model = load_model_from_google_drive(google_drive_file_id, save_path)
# model = tf.keras.models.load_model(save_path)



# Streamlit UI
st.title("DeepGuard - Deepfake Guardian")
st.write("Detecting and classifying fake generated from the real images")


# Upload image
upload_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if upload_image is not None:
    image, image_array = preprocess_image(img=upload_image, img_size=(224,224))
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(image)#, use_column_width=True)
    with col2:
        prediction = model.predict(image_array)
        probability = prediction[0][0]
        
        label,clr = ("Real","Green") if probability > 0.5 else ("Fake","Red")
        st.write(f'<p style="color: {clr}; font-size: 20px;">{probability*100:.2f}% chances of image being a {label} image</p>', unsafe_allow_html=True)
        
    