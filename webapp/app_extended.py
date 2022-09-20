import streamlit as st
import numpy as np
from PIL import Image 
import tensorflow as tf

st.title("Binary Human Detection Web App")
st.markdown(""" <style> .font {
font-size:28px ; font-family: 'Tahoma'; color: #191970;} 
</style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Is there a human in office space? 🧍</p>', unsafe_allow_html=True)
#st.markdown("**Is there a human in office space?** 🧍")

st.sidebar.subheader("Select a Neural Network Model")
model_name = st.sidebar.selectbox("Model", ("CNN Model", "CNN model with regularisation", "VGG-16", "AlexNet", "ResNet50",
 "CNN-LTSM", "Vision Transformer(ViT)"))

if model_name == "CNN Model":
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
        streamlit_widget_image = st.image(tensorflow_image, 'Uploaded Image', width=440)

    ## Do prediction
    if st.sidebar.button("Click Here to Predict"):
        
        if uploaded_file is None:
            st.sidebar.write("Please upload an Image to Classify")
        else:      
            ## Pass the preprocessed image to the cnn model (not the streamlit widget)
            pred_label = model_cnn.predict(tensorflow_image)[0]

            ## Print prediction
            st.sidebar.header("CNN model results:") 
            if pred_label > 0.5: st.sidebar.info('Human is detected')
            else: st.sidebar.info('No human is detected')    


if model_name == 'CNN model with regularisation':

    ## Initialize tensorflow model (This can be loaded before anything else)
    path_to_model = "C:/Users/maria/Jupiter_Notebooks/Dataset_Thermal_Project/Camera_videos/Saved_models/cnn_model_with_reg.h5"
    model_loader = tf.keras.models.load_model(path_to_model)
    model_cnn_reg = tf.keras.models.Model(model_loader.inputs, model_loader.outputs)

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
        streamlit_widget_image = st.image(tensorflow_image, 'Uploaded Image', width=440)

    ## Do prediction
    if st.sidebar.button("Click Here to Predict"):
        
        if uploaded_file is None:
            st.sidebar.write("Please upload an Image to Classify")
        else:      
            ## Pass the preprocessed image to the cnn model (not the streamlit widget)
            pred_label = model_cnn_reg.predict(tensorflow_image)[0]

            ## Print prediction
            st.sidebar.header("CNN with regularisation results:") 
            if pred_label > 0.5: st.sidebar.info('Human is detected')
            else: st.sidebar.info('No human is detected')


if model_name == 'VGG-16':
    ## Initialize tensorflow model (This can be loaded before anything else)
    path_to_model = "C:/Users/maria/Jupiter_Notebooks/Dataset_Thermal_Project/Camera_videos/Saved_models/model_vgg16.h5"
    model_loader = tf.keras.models.load_model(path_to_model)
    model_vgg = tf.keras.models.Model(model_loader.inputs, model_loader.outputs)

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
        streamlit_widget_image = st.image(tensorflow_image, 'Uploaded Image', width=440)

    ## Do prediction
    if st.sidebar.button("Click Here to Predict"):
        
        if uploaded_file is None:
            st.sidebar.write("Please upload an Image to Classify")
        else:      
            ## Pass the preprocessed image to the cnn model (not the streamlit widget)
            pred_label = model_vgg.predict(tensorflow_image)[0]

            ## Print prediction
            st.sidebar.header("VGG-16 results:") 
            if pred_label > 0.5: st.sidebar.info('Human is detected')
            else: st.sidebar.info('No human is detected')

if model_name == 'AlexNet':
        ## Initialize tensorflow model (This can be loaded before anything else)
    path_to_model = "C:/Users/maria/Jupiter_Notebooks/Dataset_Thermal_Project/Camera_videos/Saved_models/model_alexnet.h5"
    model_loader = tf.keras.models.load_model(path_to_model)
    model_alexnet = tf.keras.models.Model(model_loader.inputs, model_loader.outputs)

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
        streamlit_widget_image = st.image(tensorflow_image, 'Uploaded Image', width=440)

    ## Do prediction
    if st.sidebar.button("Click Here to Predict"):
        
        if uploaded_file is None:
            st.sidebar.write("Please upload an Image to Classify")
        else:      
            ## Pass the preprocessed image to the cnn model (not the streamlit widget)
            pred_label = model_alexnet.predict(tensorflow_image)[0]

            ## Print prediction
            st.sidebar.header("AlexNet results:") 
            if pred_label > 0.5: st.sidebar.info('Human is detected')
            else: st.sidebar.info('No human is detected')


if model_name == 'ResNet50':
        ## Initialize tensorflow model (This can be loaded before anything else)
    path_to_model = "C:/Users/maria/Jupiter_Notebooks/Dataset_Thermal_Project/Camera_videos/Saved_models/model_resnet.h5"
    model_loader = tf.keras.models.load_model(path_to_model)
    model_resnet = tf.keras.models.Model(model_loader.inputs, model_loader.outputs)

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
        streamlit_widget_image = st.image(tensorflow_image, 'Uploaded Image', width=440)

    ## Do prediction
    if st.sidebar.button("Click Here to Predict"):
        
        if uploaded_file is None:
            st.sidebar.write("Please upload an Image to Classify")
        else:      
            ## Pass the preprocessed image to the cnn model (not the streamlit widget)
            pred_label = model_resnet.predict(tensorflow_image)[0]

            ## Print prediction
            st.sidebar.header("ResNet50 results:") 
            if pred_label > 0.5: st.sidebar.info('Human is detected')
            else: st.sidebar.info('No human is detected')    


if model_name == 'CNN-LTSM':
        ## Initialize tensorflow model (This can be loaded before anything else)
    path_to_model = "C:/Users/maria/Jupiter_Notebooks/Dataset_Thermal_Project/Camera_videos/Saved_models/cnn_ltsm_model.h5"
    model_loader = tf.keras.models.load_model(path_to_model)
    cnn_ltsm_model = tf.keras.models.Model(model_loader.inputs, model_loader.outputs)

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
        streamlit_widget_image = st.image(tensorflow_image, 'Uploaded Image', width=440)

    ## Do prediction
    if st.sidebar.button("Click Here to Predict"):
        
        if uploaded_file is None:
            st.sidebar.write("Please upload an Image to Classify")
        else:      
            ## Pass the preprocessed image to the cnn model (not the streamlit widget)
            pred_label = cnn_ltsm_model.predict(tensorflow_image)[0]

            ## Print prediction
            st.sidebar.header("CNN-LTSM results:") 
            if pred_label > 0.5: st.sidebar.info('Human is detected')
            else: st.sidebar.info('No human is detected')


if model_name == 'Vision Transformer(ViT)':
        ## Initialize tensorflow model (This can be loaded before anything else)
    path_to_model = "C:/Users/maria/Jupiter_Notebooks/Dataset_Thermal_Project/Camera_videos/Saved_models/model_vit.h5"
    model_loader = tf.keras.models.load_model(path_to_model)
    model_vit = tf.keras.models.Model(model_loader.inputs, model_loader.outputs)

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
        streamlit_widget_image = st.image(tensorflow_image, 'Uploaded Image', width=440)

    ## Do prediction
    if st.sidebar.button("Click Here to Predict"):
        
        if uploaded_file is None:
            st.sidebar.write("Please upload an Image to Classify")
        else:      
            ## Pass the preprocessed image to the cnn model (not the streamlit widget)
            pred_label = model_vit.predict(tensorflow_image)[0]

            ## Print prediction
            st.sidebar.header("ViT results:") 
            if pred_label > 0.5: st.sidebar.info('Human is detected')
            else: st.sidebar.info('No human is detected')

