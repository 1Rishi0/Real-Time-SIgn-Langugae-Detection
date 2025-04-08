import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Path to dataset
dataset_folder = "sign_language_dataset"
sign_words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Prepare data and labels
X = []  # landmark data
y = []  # labels

# Load data
for label, word in enumerate(sign_words):
    word_folder = os.path.join(dataset_folder, word)
    for file in os.listdir(word_folder):
        if file.endswith(".npy"):
            # Load landmark data
            landmarks = np.load(os.path.join(word_folder, file))
            # Check if landmarks have the expected shape
            if landmarks.shape != (126,):  # Expecting shape (126,)
                print(f"Warning: Found unexpected shape for {file}: {landmarks.shape}. Skipping this file.")
                continue
            X.append(landmarks)
            y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Ensure X has the correct shape: (num_samples, 126) for two hands
if X.shape[1] != 126:
    raise ValueError("Expected each landmark entry to have a shape of (126,) for two hands.")

# Encode labels and convert to categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(126,)),  # Input shape should be 126 for two hands
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(sign_words), activation='softmax')  # Output layer for multi-class classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save the model{ "cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}

model.save("model.h5")
print("Model saved as 'model.h5'")

# Evaluate model accuracy on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")

from sklearn.metrics import classification_report

# Predict labels for test set
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Print classification report
print(classification_report(y_true, y_pred, target_names=sign_words))


