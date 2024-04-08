import librosa
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow_model_optimization as tfmot

# TensorFlow Model Optimization Toolkit specific imports for sparsity
sparsity = tfmot.sparsity.keras

# Parameters
n_mels = 128
n_fft = 2048
hop_length = 512
n_classes = 8
max_files = 2  # Limit the number of files for quick testing

# Define the path to your dataset
dataset_path = 'Audio_Song_Actors_01-24'

# Emotion labels
emotion_dict = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Function to extract Mel-spectrogram
def extract_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

spectrograms = []
labels = []
files_processed = 0

# Extract features and labels
for subdir, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav') and files_processed < max_files:
            try:
                emotion_code = file.split('-')[2]
                emotion = emotion_dict.get(emotion_code, 'unknown')
                if emotion != 'unknown':
                    filepath = os.path.join(subdir, file)
                    S_DB = extract_mel_spectrogram(filepath)
                    spectrograms.append(S_DB)
                    labels.append(emotion)
                    files_processed += 1
            except Exception as e:
                print(f"Could not process file {file}: {e}")
        if files_processed >= max_files:
            break
    if files_processed >= max_files:
        break

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Padding and reshaping spectrograms
max_length = max(s.shape[1] for s in spectrograms)
X = np.array([librosa.util.fix_length(s, size=max_length, axis=1) for s in spectrograms])
X = X[..., np.newaxis]  # Adding a channel dimension

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)

# CNN Model Definition using Sequential API
# CNN Model Definition using Sequential API with fixed input shape
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(n_mels, max_length, 1)),  # Adjusted input shape
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])



# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model with callbacks for TensorBoard and possibly early stopping and model checkpointing
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    # Add early stopping and model checkpoint callbacks here if needed
]

# Train the model with pruning
model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), callbacks=callbacks)

# Saving the model and weights
model.save('path_to_save_model')
model.save_weights('path_to_save_weights.h5')

# Remove pruning wrappers and save the pruned model weights
final_model = sparsity.strip_pruning(model)
final_model.save_weights('pruned_cnn_speech_emotion_recognition_weights.h5')

# Save the pruned model's architecture to a JSON file
with open('pruned_cnn_speech_emotion_recognition_model.json', 'w') as f:
    f.write(final_model.to_json())

print("Pruned CNN model weights and architecture saved.")

