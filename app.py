pip install tensorflow 
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow import keras
import tensorflow as tf
import streamlit as st
import cv2 
import numpy as np
import joblib  


st.set_page_config(layout="wide")
st.title('Welcome to our project melanoma detection')

st.sidebar.header("Select wich model you want to detect with")
#svm_m=st.sidebar.checkbox("SVM classifier")
#cnn_m=st.sidebar.checkbox("CNN modele from scrach")
#trs_m=st.sidebar.checkbox("Using Tranformers")

select_model = st.sidebar.radio(
    "choisit un model",
    ('SVM classifier', 'CNN modele frm scrach', 'Using Tranformers'))
if select_model=='SVM classifier':      

     
     #Load the trained machine learning model
     model = joblib.load('my_model2.pkl')      

     # Define the function to make predictions
     def predict(image):
        # Preprocess the image
        gray = image.convert('L')                    
        resize = gray.resize((64, 66))
        resized_image = resize.resize((336, 336))
        image_array = np.array(resize).flatten().reshape(1, -1)
        # Make a prediction using the SVM classifier
        prediction = model.predict(image_array)[0]
        return prediction
    
     st.subheader('Testing your tumor using svm classifier')
     uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
     
     # Set default image file path
     #default_image_path = r"C:\\Users\\pc\\Downloads\\UK_wildbirds-01-robin.jpg"

     # Load and display default image
     #image = cv2.imread(default_image_path)
     #st.image(image, caption='Default Image', use_column_width=True)
    
     # Make a prediction if an image is uploaded
     if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Make a prediction using the SVM classifier
        prediction = predict(image)
        # Show the uploaded image and prediction result
        st.image(image, caption='Uploaded Image',  width=250)
        if prediction == 0:
            st.write('Prediction: Non-Melanoma')
        else:
            st.write('Prediction: Melanoma')
             
if select_model=='CNN modele frm scrach':
     # Load the saved CNN model architecture and weights
     model= keras.models.load_model('my_model(1).h5')

     

     
     st.title('Melanoma Classification')
     st.write('Upload an image of a skin lesion to classify it as melanoma or non-melanoma')

     uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

     if uploaded_file is not None: 
        # Display the uploaded image
        image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(64, 64))
        st.write(' voila votre image')
        st.image(image, caption='Uploaded Image', width=200)

        # Preprocess the image
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        img_array /= 255.

        # Make a prediction using your CNN model
        prediction = model.predict(img_array)

        # Display the prediction result
        if prediction > 0.5:
            st.write('The image is predicted to be malignant.')
        else:
            st.write('The image is predicted to be benign.')  
            

if select_model=='Using Tranformers':
            st.write('nothing')

