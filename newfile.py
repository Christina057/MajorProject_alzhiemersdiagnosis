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
    img = image.load_img(image_path, target_size=(176, 176),color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

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
    st.image('logo.jpg', use_column_width=True, width=200)
    st.markdown(
        """
        <style>
        div[data-testid="stImage"] img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            max-width: 10%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 style='text-align: center;'>Comprehensive System for Alzheimer's Disease Diagnoses</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image_path = save_uploaded_file(uploaded_file)
        if not is_brain_mri(image_path, model_mri_nonmri):
            st.error("Uploaded file is not a brain MRI image. Please upload a correct image.")
        else:
            st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
            st.write("")
            if st.button('Show Details'):
                st.write("Classifying...")

                # Perform classification
                predicted_class, confidence, predictions = predict(image_path, model_alzheimers)

                # Display prediction results
                st.subheader("Classification Results:")
                st.write(f"Prediction: **{predicted_class}**")
                st.write(f"Confidence: **{confidence:.2%}**")

                # Display raw prediction data in a table
                raw_data = {'Class Label': list(class_labels.values()), 'Probability': predictions}
                raw_df = pd.DataFrame(raw_data)
                st.subheader("Raw Prediction Data:")
                st.dataframe(raw_df)

                st.session_state['results'] = (predicted_class, confidence, raw_df)

            # Export option for downloading results as PDF
            export_option = st.selectbox("Export results as", ("Select format", "PDF with Image and Results"))
            if export_option == "PDF with Image and Results":
                if 'results' in st.session_state:
                    predicted_class, confidence, raw_df = st.session_state['results']

                    # Generate PDF with image and results
                    pdf = FPDF()
                    pdf = save_image_to_pdf(image_path, pdf)
                    pdf.add_page()
                    pdf.set_font("Arial", size=30)
                    results_text = (
                        f"Prediction: {predicted_class}\n"
                        f"Confidence: {confidence:.2%}\n"
                        f"Raw Prediction Data:\n{raw_df.to_string(index=False)}"
                    )
                    pdf_output = pdf.output(dest="S").encode("latin1")
                    b64 = base64.b64encode(pdf_output).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="results.pdf">Download PDF with Image and Results</a>'
                    st.markdown(href, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center;'>Gayatri Vidya Parishad College of Engineering For Women</h1>", unsafe_allow_html=True)
    st.markdown("""
    <style>
        img.logo {
            position: absolute;
            top: 10px;
            left: 10px;
            width: 100px;
            height: auto;
        }
    </style>
    """, unsafe_allow_html=True)
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
            background-image: url("https://img.freepik.com/premium-photo/concept-brainstorming-artificial-intelligence-with-blue-color-human-brain-background_121658-753.jpg");
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


