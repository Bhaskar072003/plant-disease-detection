import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# If you have custom objects, define them here
# For example, if you have a custom layer or function:
def custom_activation(x):
    return x

# Define the absolute path to your model
model_path = os.path.join(os.getcwd(), 'root/cnn_model.keras')

# Check if the model file exists
if os.path.exists(model_path):
    try:
        # Load your trained model with custom objects
        custom_objects = {'custom_activation': custom_activation}  # Add your custom objects here
        model = load_model(model_path, custom_objects=custom_objects)
        st.write("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error(f"Model file not found at {model_path}")

# Define the class names
classes = ['Healthy', 'Powdery', 'Rust']

# Set up the Streamlit interface
st.title("Plant Disease Detection")
st.write("Upload an image of a plant condition to classify it.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = image.resize((192, 192))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Predict the class of the image
    if 'model' in globals():
        prediction = model.predict(x)[0]
        test_pred = np.argmax(prediction)
        result = classes[test_pred]
        
        # Display the result
        st.write(f"Prediction: {result}")
        st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
    else:
        st.error("Model is not loaded. Please check the model file path.")

# Run the Streamlit app
if __name__ == "__main__":
    st._is_running_with_streamlit = True
    # st.run()
    # st.stop()
