import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_cropper import st_cropper

# Load the trained model
model_path = 'modelname.h5'
try:
    model = load_model(model_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Define class labels manually (replace these with your actual class labels)
class_labels = {0: 'A1', 1: 'A2', 2: 'A3.5', 3: 'A3', 4: 'A4', 5: 'B1', 6: 'B2', 7: 'B3', 8: 'B4', 9: 'C1', 10: 'C2', 11: 'C3', 12: 'C4', 13: 'D2', 14: 'D3', 15: 'D4'}

# Function to preprocess and predict the class of images
def predict_image(img, model):
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale pixel values

    predictions = model.predict(img_array)  # Predict class probabilities
    top_three_indices = predictions[0].argsort()[-16:][::-1]  # Get indices of the top 3 predictions

    # Get the class names and probabilities for the top 3 predictions
    top_three_labels = [class_labels.get(idx, 'Unknown') for idx in top_three_indices]
    top_three_probs = [predictions[0][idx] for idx in top_three_indices]

    return top_three_labels, top_three_probs

# Streamlit app
st.title("Teeth Shade Recognition")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Cropping option
    cropped_image = st_cropper(image, realtime_update=True, box_color='blue', aspect_ratio=(1, 1))
    
    if cropped_image:
        st.image(cropped_image, caption="Cropped Image", use_column_width=True)

    # Predict button
    if st.button("Predict") and model is not None:
        if cropped_image is None:
            st.error("Please crop the image before predicting.")
        else:
            with st.spinner("Predicting..."):
                top_three_labels, top_three_probs = predict_image(cropped_image, model)
                st.success("Prediction complete!")

                # Display predictions
                # st.write(f"Top 3 Predicted Classes:")
                for label, prob in zip(top_three_labels, top_three_probs):
                    st.write(f"{label}: {prob:.2%}")

                # Display bar chart
                fig, ax = plt.subplots()
                ax.barh(top_three_labels, top_three_probs, color='skyblue')
                ax.set_xlabel('Probability')
                ax.set_xlim(0, 1)
                ax.set_title('Top 3 Predictions')
                st.pyplot(fig)
    elif model is None:
        st.error("Model not loaded. Please check the model path and try again.")
