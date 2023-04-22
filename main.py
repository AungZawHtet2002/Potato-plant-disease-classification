import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf


def loadModel():
    model = tf.keras.models.load_model(
        "Potato1.h5")
    return model


model = loadModel()
class_name=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
# Define the Streamlit app
# def app():
    # Set the title and sidebar
st.title('Potato Plant Disease Classification')
st.sidebar.title('Upload Image')
    
    # Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    
if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image for model input
        image = image.resize((256, 256)) # Resize the image to match the input size of the model
        image = np.array(image) # Convert the image to a numpy array
        image = image / 255.0 # Normalize the pixel values to [0, 1]
        image = np.expand_dims(image, axis=0) # Add an extra dimension for batch size
        
        # Make predictions with the model
        predictions=model.predict(image)
        predicted_class=class_name[np.argmax(predictions[0])]
        confidence=round(100*(np.max(predictions[0])),2)
        
        
        
        # Display the predicted label
        st.success(predicted_class)
        st.write(confidence)