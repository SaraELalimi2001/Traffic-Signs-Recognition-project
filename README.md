# Traffic-signs-recognition

This project focuses on traffic sign recognition, which is a computer vision application designed to identify and interpret various traffic signs in real-time. It is primarily developed for the benefit of drivers, autonomous vehicles, and transportation authorities to enhance road safety and efficiency.


![Logo](https://www-sygic.akamaized.net/content/13-what-is/0-sign-recognition/traffic-sign-recognition.png)



## Dataset

The dataset is sourced from Kaggle : [Traffic signs recognition](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)


## EDA and Data Preparation

Preliminary analyses were conducted to:

-Dataset Organization: The dataset comprises images of different traffic sign classes, organized into folders representing each class.

-Image Standardization: Images are read using OpenCV and resized to a uniform size of 32x32 pixels to ensure consistent dimensions for training.

-Label Association: Images and their corresponding labels are stored in an array, enabling easy linking of input images to traffic sign classes during training.

-Dataset Splitting: The dataset is split into training, validation, and testing sets using scikit-learn's train_test_split function to ensure proper model training, validation, and evaluation.

-Data Preprocessing: Preprocessing steps include converting images to grayscale, applying histogram equalization for contrast enhancement, and normalizing pixel values between 0 and 1.

-Data Augmentation: To increase data diversity, the ImageDataGenerator from Keras is employed, introducing various transformations such as random shifts, zooms, and rotations to the training images.


## Modeling 

The CNN model, built using Keras, employs convolutional layers to extract features, max pooling layers for dimension reduction, and dropout layers for regularization. Fully connected layers combine features for predictions. After compiling with an optimizer and loss function, the model is trained on augmented data, using backpropagation and gradient descent. Performance is evaluated, and the trained model can make predictions on new traffic sign images, contributing to road safety and intelligent transportation systems.

 
## Deployment

To enhance interaction with the model, we've developed a user-friendly graphical interface using Gradio.


