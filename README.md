# Sign Language Recognition (ISL & ASL)

This project focuses on developing a machine learning model to recognize static gestures in **Indian Sign Language (ISL)** and **American Sign Language (ASL)**. The goal is to classify images of hand gestures into corresponding symbols (letters or words) to aid communication for the hearing and speech-impaired.

## Project Overview

- **Dataset**: Uses the ISL and ASL datasets from Kaggle, containing labeled images of hand gestures.
- **Model**: A **Convolutional Neural Network (CNN)** is used to classify the images, with **transfer learning** to leverage pre-trained models like **ResNet50** or **MobileNetV2**.
- **Preprocessing**: Images are resized to a standard resolution, and **data augmentation** techniques are applied to improve the model’s performance and robustness.

## Features

- Combines ISL and ASL datasets for broader recognition.
- Visualizes class distribution using **bar plots**.
- Implements a **CNN-based classification model**.
- Trains the model using **Google Colab** for GPU support.

## Requirements

- Python
- TensorFlow
- Keras
- Matplotlib
- Seaborn
- Pillow

## References

1. **Indian Sign Language (ISL) Dataset**: [Kaggle ISL Dataset](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
2. **American Sign Language (ASL) Dataset**: [Kaggle ASL Dataset](https://www.kaggle.com/datasets/prathumarikeri/american-sign-language-09az)
3. **TensorFlow Documentation**: [TensorFlow Image Classification](https://www.tensorflow.org/)

---
