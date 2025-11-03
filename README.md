# AI-Models
This repository contains two deep learning projects developed in Python using TensorFlow and Keras. Both models are pre-trained and ready to use, with their correspondin

Repository Contents
1. Clasificador_de_imagenes.py

A Streamlit web application that classifies images from the Fashion MNIST dataset.
Users can upload a grayscale image (preferably 28x28 pixels but the program can change them to fit the scale), and the model will predict the clothing category (such as T-shirt, coat, sneaker, bag, etc.).

Pre-trained model: modelo_fashion_mnist.h5

Main features:

Loads and compiles a pre-trained CNN model.

Accepts user-uploaded images and automatically converts them to grayscale.

Displays prediction results and confidence scores.

Visualizes class probabilities with a bar chart.

2. MD_RNN.py

A Recurrent Neural Network (RNN) with an LSTM layer trained on the IMDB movie reviews dataset for sentiment analysis.
The model predicts whether a movie review expresses a positive or negative sentiment.

Pre-trained model: modelo_sentimientos_rnn.h5

Main features:

Loads and preprocesses the IMDB dataset.

Builds and trains an RNN model using LSTM layers.

Evaluates accuracy and loss on test data.

Demonstrates sample predictions with decoded reviews.

Technologies Used

Python 3.x

TensorFlow / Keras

NumPy

Streamlit

Pillow (PIL)

IMDB Dataset (Keras built-in)

Fashion MNIST Dataset (TensorFlow Datasets)
