import streamlit as st
import numpy as np
from PIL import Image 
import tensorflow as tf

st.title("Binary Human Detection Web App")
st.markdown("Is there a human in office space? 🧍")

## Initialize tensorflow model (This can be loaded before anything else)
path_to_model = "C:/Users/maria/Jupiter_Notebooks/Dataset_Thermal_Project/Camera_videos/Saved_models/cnn_model.h5"
model_loader = tf.keras.models.load_model(path_to_model)
model_cnn = tf.keras.models.Model(model_loader.inputs, model_loader.outputs)

## Preprocess images
def preprocessImage(photo):
    resize_photo = photo.resize((224,224))
    normalized_photo = np.array(resize_photo)/255 # a normalised 2D array                
    reshaped_photo = normalized_photo.reshape(-1, 224, 224, 3)   # to shape as (1, 224, 224, 3)
    return reshaped_photo

uploaded_file = st.sidebar.file_uploader(" ",type=['jpg', 'jpeg'])    

if uploaded_file is not None:
    ## Use a context manager to make sure to close the file!! 
    with Image.open(uploaded_file) as photo:
        tensorflow_image = preprocessImage(photo)
    
    ## Show preprocessed image
    streamlit_widget_image = st.image(tensorflow_image, 'Uploaded Image', use_column_width=True)

## Do prediction
if st.sidebar.button("Click Here to Predict"):
    
    if uploaded_file is None:
        st.sidebar.write("Please upload an Image to Classify")
    else:      
        ## Pass the preprocessed image to the cnn model (not the streamlit widget)
        pred_label = model_cnn.predict(tensorflow_image)[0]

        ## Print prediction
        st.sidebar.header("CNN results:") 
        if pred_label > 0.5: st.sidebar.info('Human is detected')
        else: st.sidebar.info('No human is detected')