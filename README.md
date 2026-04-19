# Clothing Image Classification

## Project Overview

This project focuses on building a **Computer Vision model** to classify different types of clothing images. The model can identify what type of clothing is present in an image and also explain its prediction using visualization techniques like **Grad-CAM**.

---

## Dataset Information

The dataset used for this project contains images of clothing items from **10 categories**:

* Dress
* Hat
* Long Sleeve
* Outwear
* Pants
* Shirt
* Shoes
* Shorts
* Skirt
* T-shirt

The dataset was taken from Kaggle and includes labeled images for training and testing the model.

---

## Data Preprocessing

Before training the model, the dataset was prepared by:

* Resizing images to a fixed size
* Normalizing pixel values
* Splitting data into training, validation, and testing sets
* Organizing images into category-wise folders

---

## Model Development

A deep learning model was built using Python in VS Code:

* Used **TensorFlow/Keras** for model building
* Trained the model on labeled clothing images
* Evaluated performance using validation data
* Achieved accurate predictions for most clothing categories

---

## Prediction System

The model takes an input image and:

* Predicts the type of clothing (e.g., T-shirt, Shirt, Dress, etc.)
* Provides confidence score for the prediction

---

## Model Explainability (Grad-CAM)

To understand how the model makes decisions:

* **Grad-CAM** technique was applied
* Generates a heatmap highlighting important regions of the image
* Shows which part of the image influenced the prediction

Example: For a T-shirt image, the model focuses on the central clothing area rather than the background.

---

## Output Visualization

The output includes:

* Original Image
* Heatmap
* Grad-CAM Overlay

This helps in clearly understanding both prediction and reasoning.

---

## Tools & Technologies

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib

---

## Conclusion

This project not only classifies clothing images but also explains **why** a prediction was made. It demonstrates both model accuracy and interpretability, which is important in real-world AI applications.
