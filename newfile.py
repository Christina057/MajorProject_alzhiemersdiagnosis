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

    tf.keras.utils.get_custom_objects()['SeqSelfAttention'] = SeqSelfAttention

    with tf.keras.utils.custom_object_scope({'SeqSelfAttention': SeqSelfAttention}):
        model_alzheimers = load_model("new_model.h5")
        model_mri_nonmri = tf.keras.models.load_model('mri_nonmri_classifier.h5')
        class_labels = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}

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
                predicted_class, confidence, predictions = predict(image_path, model_alzheimers)
                st.write(f"Prediction: {predicted_class}")
                st.write(f"Confidence: {confidence:.2%}")

                raw_data = {'Class Label': list(class_labels.values()), 'Probability': predictions}
                raw_df = pd.DataFrame(raw_data)
                st.write("Raw Prediction Data:")
                st.write(raw_df)

                st.session_state['results'] = (predicted_class, confidence, raw_df)

            export_option = st.selectbox("Export results as", ("Select format", "PDF with Image and Results"))
            if export_option == "PDF with Image and Results":
                if 'results' in st.session_state:
                    predicted_class, confidence, raw_df = st.session_state['results']
                    st.write(f"Prediction: {predicted_class}")
                    st.write(f"Confidence: {confidence:.2%}")
                    st.write("Raw Prediction Data:")
                    st.write(raw_df)

                pdf = FPDF()
                pdf = save_image_to_pdf(image_path, pdf)
                pdf.add_page()
                pdf.set_font("Arial", size=30)
                results = f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}\nRaw Prediction Data:\n{raw_df.to_string(index=False)}"
                pdf_output = pdf.output(dest="S").encode("latin1")
                b64 = base64.b64encode(pdf_output)
                pdf_str = b64.decode()
                href = f'<a href="data:application/octet-stream;base64,{pdf_str}" download="results.pdf">Download PDF with Image and Results</a>'
                st.markdown(href, unsafe_allow_html=True)

def about_page():
    st.title("About Us")
    st.markdown("<h2 style='text-align: left; font-size: 20px;'>Welcome to our About Us page! We are dedicated to making a positive impact in the field of healthcare, particularly in the early diagnosis of Alzheimer's disease.</h2>", unsafe_allow_html=True)

    st.markdown("<p style='font-size: 23px;'>Alzheimer's disease is a progressive brain disorder that primarily affects memory, thinking, and behavior. It is the most common cause of dementia, a general term for a decline in cognitive abilities that interfere with daily life. The exact cause of Alzheimer's disease is not yet fully understood, but it is believed to involve a combination of genetic, environmental, and lifestyle factors.</p>", unsafe_allow_html=True)

    st.markdown("<p style='font-size: 23px;'>Symptoms of Alzheimer's disease include:</p>", unsafe_allow_html=True)
    st.markdown("<ol><li style='font-size: 23px;'>Memory Loss: One of the most common early symptoms is difficulty remembering newly learned information. This may progress to forgetting important dates, events, or asking for the same information repeatedly.</li><li style='font-size: 23px;'>Cognitive Decline: Individuals may experience challenges in planning or solving problems, difficulty concentrating, and completing familiar tasks.</li><li style='font-size: 23px;'>Confusion and Disorientation: People with Alzheimer's may become confused about time, place, and person. They may get lost in familiar places and lose track of dates, seasons, and the passage of time.</li><li style='font-size: 23px;'>Language Problems: Finding the right words, following or joining a conversation, and understanding what others are saying may become increasingly difficult.</li><li style='font-size: 23px;'>Mood and Personality Changes: Individuals may undergo changes in mood and personality, such as becoming suspicious, anxious, depressed, or easily upset. They may withdraw from social activities and show a decline in interest or motivation.</li><li style='font-size: 23px;'>Difficulty in Judgement: Impaired judgment and decision-making abilities may become apparent, leading to poor financial decisions or neglect of personal hygiene.</li></ol>", unsafe_allow_html=True)

    st.markdown("<p style='font-size: 23px;'>The use of algorithms, such as Convolutional Neural Networks (CNNs), in examining brain images, especially MRI scans, is a promising avenue for improving the diagnosis of Alzheimer's disease. These algorithms can help identify subtle patterns and abnormalities in the brain that may indicate the presence of the disease at an early stage. Early detection is crucial for initiating interventions and treatments that may slow down the progression of the disease and improve the quality of life for affected individuals. The goal is to leverage advanced computational techniques to enhance the accuracy and efficiency of diagnosing Alzheimer's disease, ultimately aiding in the development of more effective therapeutic strategies.</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 23px;'>Thank you for visiting our About Us page. If you have any questions or would like to learn more about our work, feel free to reach out!</p>", unsafe_allow_html=True)

def contact_page():
    st.title("Contact Us")
    st.markdown("<p style='font-size: 25px;'>Feel free to reach out to us with any questions or inquiries. We're here to help!</p>", unsafe_allow_html=True)

    st.markdown("<p style='font-size: 25px;'>For individual inquiries, you can reach out to our team members:</p>", unsafe_allow_html=True)
    st.markdown("<ul><li style='font-size: 25px;'>M Gautami: gautamisaisatya@gmail.com</li><li style='font-size: 25px;'>K Devi: devikadali322@gmail.com</li><li style='font-size: 25px;'>K Christina: christinaprasa040@gmail.com</li><li style='font-size: 25px;'>A Mounika: mounikasai152@gmail.com</li></ul>", unsafe_allow_html=True)

    st.markdown("<p style='font-size: 25px;'>We appreciate your interest in our work and look forward to hearing from you!</p>", unsafe_allow_html=True)


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
        .navbar {
            overflow: hidden;
            background-color: #333;
            width: 100%;
        }

        .navbar a {
            float: left;
            display: block;
            color: #f2f2f2;
            text-align: center;
            padding: 12px 250px;
            text-decoration: none;
        }
        div[data-baseweb="file-uploader"] > div {
            width: 50% !important;  /* Adjust the width as per your requirement */
            margin: auto !important;
        }
        .navbar a:hover {
            background-color: #ddd;
            color: black;
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

    st.markdown("""
    <div class="navbar">
        <a href="?page=home">Home</a>
        <a href="?page=about">About Us</a>
        <a href="?page=contact">Contact Us</a>
    </div>
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


