import streamlit as st
import numpy as np
from PIL import Image
import tensorflow.keras as K
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained model
model = K.models.load_model("root/cnn_model.keras")
# Define the class names
classes = ['Healthy','Powdery','Rust']

# Set up the Streamlit interface
st.title("Plant Disease Detection")
st.write("Upload an image of a skin condition to classify it.")

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
    prediction = model.predict(x)[0]
    test_pred = np.argmax(prediction)
    result = classes[test_pred]
    
    # Display the result
    st.write(f"Prediction: {result}")
    st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")

# Run the Streamlit app
if __name__ == "__main__":
    st._is_running_with_streamlit = True
    # st.run()
    # st
