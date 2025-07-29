# ---
# Just making sure we’ve got everything we need
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
import urllib.request

print("Using TensorFlow version:", tf.__version__)  # Quick check

# ---
# Grab and unzip dataset
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
local_zip = "cats_and_dogs_filtered.zip"
urllib.request.urlretrieve(dataset_url, local_zip)

with zipfile.ZipFile(local_zip, 'r') as zip_ref:
    zip_ref.extractall()

# Directory setup
base_dir = "cats_and_dogs_filtered"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

# ---
# Some constants for resizing and batches
IMG_SIZE = (360, 360)     # I bumped this up from the usual 150
BATCH_SIZE = 32

# Augmentation config (training only gets the fancy stuff)
train_gen_config = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Just normalize validation data
val_gen_config = ImageDataGenerator(rescale=1./255)

# Load images from folders
train_data = train_gen_config.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_gen_config.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ---
# Model architecture using transfer learning (VGG16 base)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

# Don’t want to mess with pretrained weights at first
for layer in base_model.layers:
    layer.trainable = False

# Just stacking some layers on top of VGG
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()  # Peek at the model

# ---
# Let’s avoid overfitting and save our best result
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Time to train!
EPOCHS = 26

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[early_stop, checkpoint]
)

# ---
# Visualization of training vs validation results
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Time')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ---
# Wrap up a handy function to make predictions later
from tensorflow.keras.preprocessing import image

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    return "Dog" if prediction > 0.5 else "Cat"

# ---
# Upload and test your own samples (in Colab)
from google.colab import files

uploaded_files = files.upload()

for fname in uploaded_files.keys():
    outcome = predict_image(fname)
    print(f"Prediction for {fname}: {outcome}")
