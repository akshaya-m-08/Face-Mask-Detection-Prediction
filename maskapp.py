import os
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Set up the configuration for the Streamlit app page
st.set_page_config(
    page_title="Face Mask Compliance Checker",
    page_icon="mask.webp",
    initial_sidebar_state="expanded"
)

# Load the trained model that will be used for predictions
model = load_model('my_model.keras')

# Function to load and preprocess a single image
def load_and_preprocess_image(image):
    # Open the image using PIL
    image = Image.open(image)
    # Resize the image to the required size for the model
    image = image.resize((128, 128))
    # Convert the image to RGB format
    image = image.convert('RGB')
    # Convert the image to a numpy array and normalize pixel values to [0, 1]
    image = np.array(image) / 255.0
    return image

# Main title for the Streamlit app with centered alignment and custom color
st.markdown("<h1 style='text-align: center;color:#003399;'>Face Mask Compliance Checker</h1>", unsafe_allow_html=True)

# Styled option for uploading or taking a photo, with custom background and text color
st.markdown("""
    <div style="padding: 10px; text-align: center;  background-color: #666699; border-radius: 10px; border: 1px solid #ddd;">
        <h3 style="font-size: 40px; color: #FFFFFF;">Select an Option to Proceed</h3>
    </div>
    """, unsafe_allow_html=True)

# Add some space between elements for better visual layout
st.write("")
st.write("")
st.write("")

# Create two columns for the photo upload and capture options
col1, col2 = st.columns(2)

# If the user clicks on the "Upload a photo" button, store this choice in session state
with col1:
    if st.button("📸 Upload a photo", use_container_width=True):
        st.session_state.upload_option = "Upload a photo"

# If the user clicks on the "Take a photo" button, store this choice in session state
with col2:
    if st.button("📸 Take a photo", use_container_width=True):
        st.session_state.upload_option = "Take a photo"

# Retrieve the selected option from session state (default is "Upload a photo")
upload_option = st.session_state.get('upload_option', 'Upload a photo')

# Add some spacing to adjust the layout
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# Depending on the selected option, show either a file uploader or camera input
if upload_option == "Upload a photo":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a photo")

# Add some space before the next elements
st.write("")
st.write("")

# Placeholder for dialog-like messages to be displayed later
dialog_placeholder = st.empty() 
st.write("")
st.write("")

# If an image has been uploaded or taken, process it
if uploaded_file is not None:
    # Display the uploaded or captured image on the app
    st.image(uploaded_file, caption='Uploaded/Taken Image', use_column_width=True)

    # Preprocess the image for the model
    image = load_and_preprocess_image(uploaded_file)

    # Expand dimensions to match the model input requirements (batch size, height, width, channels)
    image = np.expand_dims(image, axis=0)

    # Make a prediction using the model
    prediction = model.predict(image)
    # Convert the prediction to a binary class (0 or 1) based on a threshold of 0.5
    predicted_class = np.where(prediction > 0.5, 1, 0).flatten()[0]

    # Map the predicted class to the corresponding label
    labels = {0: 'Without Mask', 1: 'With Mask'}
    label = labels[predicted_class]

    # Display a dialog-like message based on the prediction result
    if predicted_class == 1:
        dialog_placeholder.markdown("""
            <div style="border: 1px solid #d4edda; border-radius: 5px; padding: 10px; background-color: #d4edda; color: #155724;">
                <strong>Great!</strong> You are wearing a mask. Thank you for keeping safe!
            </div>
            """, unsafe_allow_html=True)
    else:
        dialog_placeholder.markdown("""
            <div style="border: 1px solid #f8d7da; border-radius: 5px; padding: 10px; background-color: #f8d7da; color: #721c24;">
                <strong>Warning!</strong> Please wear a mask for your safety and others.
            </div>
            """, unsafe_allow_html=True)
