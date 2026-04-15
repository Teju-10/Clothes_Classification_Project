import os
from utils import load_data, build_model

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

train_dir = os.path.join(BASE_DIR, "dataset/Clothing Dataset/train")
val_dir = os.path.join(BASE_DIR, "dataset/Clothing Dataset/validation")
test_dir = os.path.join(BASE_DIR, "dataset/Clothing Dataset/test")

# Load data
train_data, val_data, test_data = load_data(train_dir, val_dir, test_dir)

# Build model
model = build_model(train_data.num_classes)

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Evaluate
loss, acc = model.evaluate(test_data)
print("Test Accuracy:", acc)

# Save model
model.save(os.path.join(BASE_DIR, "models/clothing_model.h5"))

print("Model saved successfully!")