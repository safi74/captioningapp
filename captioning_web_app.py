#!/usr/bin/env python
# coding: utf-8
from signal import signal, SIGPIPE, SIG_DFL   
signal(SIGPIPE,SIG_DFL) 

import streamlit as st

import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import Xception, preprocess_input

from tensorflow.keras.preprocessing.sequence import pad_sequences

import pyttsx3 #  text-to-speech (TTS) library: Text-to-Speech for Python



# Define a function that takes an image as input and performs resizing, feature extraction and captioning
def image_captioning(uploaded_file):
    
    if uploaded_file is None:
        return "## No image uploaded, please upload an image ##"
    
    else:
        # Resizing
        target_size=(299, 299)
        with Image.open(uploaded_file) as img:
            resized_img = img.resize(target_size, Image.LANCZOS)

        # Loading Xception model for feature extraction
        xception_model = Xception(weights = 'imagenet', include_top = False, pooling = 'avg')

        img_array = image.img_to_array(resized_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        # Extract features
        features = xception_model.predict(img_array)

        with open("mapping.pkl", 'rb') as f:
            mapping = pickle.load(f)
        with open("tokenizer.pkl", 'rb') as f:
            tokenizer = pickle.load(f)
        with open("vocab_size.pkl", 'rb') as f:
            vocab_size = pickle.load(f)
        with open("all_captions.pkl", 'rb') as f:
            all_captions = pickle.load(f)
        max_length = max(len(caption.split()) for caption in all_captions)

        # Load the saved model
        model = load_model("TrainedModel1.h5")

        def idx_to_word(integer, tokenizer):
            for word, index in tokenizer.word_index.items():
                if index == integer:
                    return word
            return None

        def predict_caption(model, image_features, tokenizer, max_length):
            # adding the start tag for generation process
            in_text = 'startseq'
            # iterate over the max length of sequence
            for i in range(max_length):
                # encode input sequence
                sequence = tokenizer.texts_to_sequences([in_text])[0]
                # pad the sequence
                sequence = pad_sequences([sequence], max_length)
                # predict next word
                yhat = model.predict([image_features, sequence])
                # get index with high probability
                yhat = np.argmax(yhat)
                # convert index to word
                word = idx_to_word(yhat, tokenizer)
                # stop if word not found
                if word is None:
                    break
                # append word as input for generating next word
                in_text += " " + word
                # stop if we reach end tag
                if word == 'endseq':
                    break
            return in_text

        # predict the caption
        predicted_caption = predict_caption(model, features, tokenizer, max_length)

        return predicted_caption


# Define a function to read aloud the generated caption
def text_to_speech(text):
    # Initialize the TTS engine
    engine = pyttsx3.init()
    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)   # Volume (0.0 to 1.0)
    # Convert text to speech
    engine.say(text)
    # Wait for speech to finish
    engine.runAndWait()


# define the main() function for streamlit
def main():
    st.set_page_config(page_icon=":computer:", layout = "wide")
    background_img = """
    <style>
    [data-testid = "stAppViewContainer"] {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("https://wallpaperbat.com/img/193957-deep-learning-wallpaper.gif");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
}
    </style>
    """
    st.markdown(background_img, unsafe_allow_html=True)

    st.image("MGIT26yearsmotivate.jpg", caption=' ', use_column_width=True)
    
    #st.title(' :rainbow[AUTOMATIC IMAGE CAPTIONING AND DEPLOYMENT] ')
    st.markdown("<h1 style='text-align: center; color: white;'>AUTOMATIC IMAGE CAPTIONING AND DEPLOYMENT</h1>", unsafe_allow_html=True)
    #st.markdown("<h1 style='text-align: center; color: teal;'>Automatic Image Captioning and Deployment</h1>", unsafe_allow_html=True)
    #st.markdown("<h1 style='text-align: center; color: aqua;'>AUTOMATIC IMAGE CAPTIONING AND DEPLOYMENT</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>(PROJECT STAGE 2)</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: right; color: white;'>By: Mohammed Safiuddin Fazil (20261A6636)<br>Razi Ahmed (20261A6649)</h5>", unsafe_allow_html=True)
    
    # Add project description
    st.markdown("""
        <p style='text-align: justify; color: white;'><b>Processing our surroundings by looking at them can be a fairly easy task for a 
        normal human being. However, a similar perception by visually impaired person or for machines to be able to generate descriptions 
        from an image is a challenging task. Automatic image captioning is the first step in solving this problem. It is a very important 
        application in enhancing accessibility for visually impaired individuals, other key applications include usage in virtual 
        assistants, improving image search algorithms, image indexing, enabling context-rich content sharing on social media platforms and 
        several natural language processing applications.</b></p>
        <p style='text-align: justify; color: white;'><b>This project aims to develop an Automatic Image Captioning system by integrating
        Convolutional Neural Networks (CNN) for image feature extraction and Long Short-Term
        Memory (LSTM) networks for sequence generation and deploying it for use in real-time.
        Through a two-step process, the CNN-LSTM model encodes the image content into meaningful
        features and generates descriptive captions. Captioning the images with proper description is a
        popular research area of Artificial Intelligence. The project's outcome contributes to the
        advancement of image understanding and natural language processing, unlocking novel ways
        of interaction between humans and machines.</p>
        """, 
        unsafe_allow_html=True)
    
    st.header(' ', divider='grey')
    
    # Display data flow diagram
    st.markdown("<p style='text-align: left; color: white;'><b>Dataflow Diagram</b></p>", unsafe_allow_html=True)
    st.image("dfd_dark.jpg", caption=' ', use_column_width=False)
    
    # Display model architecture
    st.markdown("<p style='text-align: left; color: white;'><b>Model Architecture</b></p>", unsafe_allow_html=True)
    st.image("modelllHDD.png", caption=' ', use_column_width=False)
    
    # Display sample image and prediction
    st.markdown("<p style='text-align: left; color: white;'><b>Example Prediction:</b></p>", unsafe_allow_html=True)
    st.image("Girlridingtireswing_dark2.jpg", caption=' ', use_column_width=False)
    
    st.header(' ', divider='grey')

    page_2()


def page_2():
    
    st.write("""##### Choose an image for captioning (JPG files only)...""")
        
    uploaded_file = st.file_uploader("", type=["jpg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.write("\n##### Uploaded Image:\n")
        st.image(uploaded_file, caption="", use_column_width=False)
    
    # Performing captioning when image file is uploaded
    predicted_caption = image_captioning(uploaded_file)
    # Remove <startseq> and <endseq> from the predicted caption
    predicted_caption = ' '.join(predicted_caption.split()[1:-1])
    predicted_caption = predicted_caption.capitalize() # Capitalize the first letter
    predicted_caption += '.' # Add a full stop at the end
    st.write("\n\n##### Predicted Caption:\n")
    st.markdown(f"<p style='font-size: 24px;'><em>{predicted_caption}</em></p>", unsafe_allow_html=True)
    
    # Add a speaker button
    if st.button("🔊 Play Caption"):
        # Perform text-to-speech for the predicted caption
        text_to_speech(predicted_caption)

    
if __name__ == "__main__":
    main()
