import os
from utils import load_data, build_model

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

train_dir = os.path.join(BASE_DIR, "dataset/Clothing Dataset/train")
val_dir = os.path.join(BASE_DIR, "dataset/Clothing Dataset/validation")
test_dir = os.path.join(BASE_DIR, "dataset/Clothing Dataset/test")

train_data, val_data, test_data = load_data(train_dir, val_dir, test_dir)

model = build_model(train_data.num_classes)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

loss, acc = model.evaluate(test_data)
print("Test Accuracy:", acc)

model.save(os.path.join(BASE_DIR, "models/clothing_model.h5"))
print("Model saved successfully!")