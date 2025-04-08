import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Path to dataset
dataset_folder = "sign_language_dataset"
sign_words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Prepare data and labels
X = []
y = []

def flip_landmarks(landmarks):
    """
    Horizontally flip hand landmarks for data augmentation.
    The x-coordinates are mirrored (1 - original x).
    """
    flipped = landmarks.copy()
    for i in range(0, len(flipped), 3):  # Iterate over (x, y, z)
        flipped[i] = 1 - flipped[i]  # Flip x-coordinate
    return flipped

# Load and augment data
for label, word in enumerate(sign_words):
    word_folder = os.path.join(dataset_folder, word)
    for file in os.listdir(word_folder):
        if file.endswith(".npy"):
            landmarks = np.load(os.path.join(word_folder, file))
            if landmarks.shape != (126,):  
                print(f"Warning: {file} has shape {landmarks.shape}. Skipping.")
                continue
            X.append(landmarks)
            y.append(label)
            
            # Augment by flipping
            X.append(flip_landmarks(landmarks))
            y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels and convert to categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the improved model with Batch Normalization & Tuning
model = Sequential([
    Dense(256, activation='relu', input_shape=(126,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(len(sign_words), activation='softmax')
])

# Compile with tuned learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # Lower LR for stability
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train, epochs=150, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save the model
model.save("enhanced_model.h5")
print("Model saved as 'enhanced_model.h5'")

# Evaluate model accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")

from sklearn.metrics import classification_report

# Predict labels for test set
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Print classification report
print(classification_report(y_true, y_pred, target_names=sign_words))
