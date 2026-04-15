import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from tkinter import Tk, filedialog

# Hide root window
Tk().withdraw()

# Open file picker
file_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

# Check if user selected a file
if not file_path:
    print("No file selected!")
    exit()

print("Selected file:", file_path)

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Load model
model_path = os.path.join(BASE_DIR, "models/clothing_model.h5")
model = tf.keras.models.load_model(model_path)

# Class names
class_names = [
    'dress', 'hat', 'longsleeve', 'outwear',
    'pants', 'shirt', 'shoes', 'shorts',
    'skirt', 't-shirt'
]

# Load image
img = image.load_img(file_path, target_size=(224,224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}")