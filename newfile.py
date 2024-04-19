import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from keras_self_attention import SeqSelfAttention
import pandas as pd
from fpdf import FPDF
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import base64
from PIL import Image
import io
import shutil
import os

def is_brain_mri(image_path, model):
    new_image = load_img(image_path, target_size=(150, 150))
    new_image = image.img_to_array(new_image)
    new_image = new_image / 255.0
    new_image = np.expand_dims(new_image, axis=0)

    prediction = model.predict(new_image)
    return prediction[0][0] <= 0.5

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(176, 176), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array
def save_image_to_pdf(image_path, pdf):
    img = Image.open(image_path)
    pdf.add_page()
    pdf.image(image_path, x=10, y=10, w=100)
    return pdf

def predict(image_path, model):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    class_labels = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = predictions[0][np.argmax(predictions)]
    return predicted_class, confidence, predictions[0]

def save_uploaded_file(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    return uploaded_file.name

def home_page():
    # Add CSS for styling the components
    st.markdown("""
    <style>
        /* Change the color of the uploaded file name to white */
        div[data-testid="stFileUploadDropzone"] div:first-child,
        div[data-testid="stFileUploadDropzone"] span {
            color: white !important;
        }

        /* Change the color of 'Choose an image...' and 'Export results as' text to white */
        div[data-testid="stFileUploadLabel"],
        div[data-testid="stSelectboxLabel"] {
            color: white !important;
        }

        /* Make the error message box darker */
        div[data-testid="stError"] {
            background-color: #ffa07a !important; 
            color: white !important; /* Text color */
        }
   
    </style>
    """, unsafe_allow_html=True)

    tf.keras.utils.get_custom_objects()['SeqSelfAttention'] = SeqSelfAttention

    with tf.keras.utils.custom_object_scope({'SeqSelfAttention': SeqSelfAttention}):
        model_alzheimers = load_model("new_model.h5")
        model_mri_nonmri = tf.keras.models.load_model('mri_nonmri_classifier.h5')
        class_labels = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}

    st.markdown(
        "<h1 style='color: white;text-align:center; margin-bottom: 20px;'>Comprehensive System for Alzheimer's Disease Diagnoses</h1>",
        unsafe_allow_html=True
    )

    # st.markdown("""
    # <div style="background-color: white; border-radius: 5px;width: fit-content;">
    #     <br><h5 style='color:black;'><b>&nbsp;Choose an image...</b></h5>
    # </div>
    # """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='color:white'><b>Choose an image...</b></h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png"], key="fileUploader")

    if uploaded_file is not None:
        image_path = save_uploaded_file(uploaded_file)
        if not is_brain_mri(image_path, model_mri_nonmri):
            st.error("Uploaded file is not a brain MRI image. Please upload a correct image.")
        else:
            # Display the uploaded image with the desired width
            # st.image(uploaded_file, caption="", width=300)
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center;">
                    <img src="data:image/png;base64,{base64.b64encode(uploaded_file.getvalue()).decode()}" style="width: 50%; height: auto;">
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<h5 style='color:white;text-align:center;'><b>Uploaded Image</b></h5>", unsafe_allow_html=True)
            
            
            if st.button('Predict Results'):
                # Perform classification
                predicted_class, confidence, predictions = predict(image_path, model_alzheimers)
                # Display prediction results
                st.markdown("<h2 style='color: white;'>Classification Results:</h2>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='color: white;'>Prediction: {predicted_class}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='color: white;'>Confidence: {confidence:.2%}</h4>", unsafe_allow_html=True)
                # Display raw prediction data in a table
                raw_data = {'Class Label': list(class_labels.values()), 'Probability': predictions}
                raw_df = pd.DataFrame(raw_data)
                st.markdown("<h2 style='color: white;'>Raw Prediction Data:</h2>", unsafe_allow_html=True)
                st.dataframe(raw_df)

                st.session_state['results'] = (predicted_class, confidence, raw_df)

            # Export option for downloading results as PDF
            st.markdown("<h3 style='color:white'><b>Export results as...</b></h3>", unsafe_allow_html=True)
            export_option = st.selectbox("", ("Select format", "PDF with Image and Results"))
            if export_option == "PDF with Image and Results":
                if 'results' in st.session_state:
                    predicted_class, confidence, raw_df = st.session_state['results']

                    # Generate PDF with image and results
                    pdf = FPDF()
                    pdf = save_image_to_pdf(image_path, pdf)
                    pdf.add_page()
                    pdf.set_font("Arial", size=15)
                    results = f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}\nRaw Prediction Data:\n{raw_df.to_string()}"
                    pdf.multi_cell(0, 10, results)
                    pdf_output = pdf.output(dest="S").encode("latin1")
                    b64 = base64.b64encode(pdf_output)
                    pdf_str = b64.decode()
                    href = f'<a href="data:application/octet-stream;base64,{pdf_str}" download="results.pdf" style="color: white;"><b>Download PDF with Image and Results</b></a>'
                    st.markdown(href, unsafe_allow_html=True)



def main():
    st.markdown("""
    <style>
        div[data-baseweb="file-uploader"] > div {
            width: 5% !important;  /* Adjust the width as per your requirement */
            margin: auto !important;
        }
        .content {
            margin-top: 50px; /* Adjust the margin to avoid overlapping with the navbar */
            display: flex;
            flex-direction: row;
            justify-content: space-around;
            align-items: flex-start; /* Align items at the start of the container */
        }
        [data-testid="stAppViewContainer"] {
            background-image: url("https://assets-global.website-files.com/6143a746b89d03a1f9b4571d/61f32de7123b7dcce1c062bb_Alto%20Final%20(0-01-30-00).jpg");
            background-size: 100%;
            background-position: top left;
            background-repeat: no-repeat;
            background-attachment: local;
        }
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
            color: white;
        }
        [data-testid="stToolbar"] {
            right: 2rem;
            color: white;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-image: url("data:image/png;base64, {img}");
            background-position: center;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    page = st.experimental_get_query_params().get("page", ["home"])[0]
    
    if page == "home":
        home_page()
    elif page == "about":
        about_page()
    elif page == "contact":
        contact_page()

if __name__ == "__main__":
    main()
