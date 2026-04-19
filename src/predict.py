import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

Tk().withdraw()

file_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not file_path:
    print("No file selected!")
    exit()

print("Selected file:", file_path)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models/clothing_model.h5")

model = tf.keras.models.load_model(model_path)

# Build model once
_ = model.predict(np.zeros((1, 224, 224, 3)))

class_names = [
    'dress', 'hat', 'longsleeve', 'outwear',
    'pants', 'shirt', 'shoes', 'shorts',
    'skirt', 't-shirt'
]

img = image.load_img(file_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

preds = model.predict(img_array)
pred_class = np.argmax(preds[0])
confidence = float(np.max(preds))

print(f"\nPrediction: {class_names[pred_class]}")
print(f"Confidence: {confidence:.2f}")

base_model = model.layers[0]

# Find last Conv layer
last_conv_layer = None
for layer in reversed(base_model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break

print("Using last conv layer:", last_conv_layer)

# Feature extractor
conv_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=base_model.get_layer(last_conv_layer).output
)

classifier_input = tf.keras.Input(shape=conv_model.output.shape[1:])
x = classifier_input

for layer in model.layers[1:]:
    x = layer(x)

classifier_model = tf.keras.models.Model(classifier_input, x)

# Compute Grad-CAM
with tf.GradientTape() as tape:
    conv_outputs = conv_model(img_array)
    tape.watch(conv_outputs)

    predictions = classifier_model(conv_outputs)
    loss = predictions[:, pred_class]

grads = tape.gradient(loss, conv_outputs)

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = np.maximum(heatmap, 0)
if np.max(heatmap) != 0:
    heatmap /= np.max(heatmap)

heatmap = np.power(heatmap, 0.5)
heatmap = cv2.resize(heatmap, (224, 224))

img_original = cv2.imread(file_path)
img_original = cv2.resize(img_original, (224, 224))

heatmap_color = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)

alpha = 0.7
superimposed_img = cv2.addWeighted(
    img_original, 1 - alpha,
    heatmap_color, alpha,
    0
)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Heatmap")
plt.imshow(heatmap, cmap='jet')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Grad-CAM Overlay")
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()