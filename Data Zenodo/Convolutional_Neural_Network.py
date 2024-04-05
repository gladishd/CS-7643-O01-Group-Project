import librosa
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Parameters
n_mels = 128
n_fft = 2048
hop_length = 512
n_classes = 8

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
    # Correct way to call librosa's melspectrogram function
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

spectrograms = []
labels = []

# Extract features and labels
for subdir, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav'):
            try:
                emotion_code = file.split('-')[2]
                emotion = emotion_dict.get(emotion_code, 'unknown')
                if emotion != 'unknown':
                    filepath = os.path.join(subdir, file)
                    S_DB = extract_mel_spectrogram(filepath)
                    spectrograms.append(S_DB)
                    labels.append(emotion)
            except Exception as e:
                print(f"Could not process file {file}: {e}")

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Padding and reshaping spectrograms
max_length = max(s.shape[1] for s in spectrograms)
X = np.array([librosa.util.fix_length(s, size=max_length, axis=1) for s in spectrograms])
X = X[..., np.newaxis]  # Adding a channel dimension

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)

# CNN Model Definition
model = models.Sequential([
    layers.Input(shape=(n_mels, max_length, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save the trained model with the recommended extension
model.save('cnn_speech_emotion_recognition_model.keras')

print("CNN model trained and saved with .keras extension.")


# To predict emotions from an audio file, load the model and call model.predict() on new spectrograms.

##############Predict emotions from audio files using the trained model###################

# Load the trained model
# After loading the model, print the input shape
model = tf.keras.models.load_model('cnn_speech_emotion_recognition_model.keras')
input_shape = model.input_shape
print("Expected input shape:", input_shape)


# Define the maximum length of the spectrograms (ensure this matches with your training data)
max_length = 216  # Adjust this based on your actual training data

# Function to extract Mel-spectrogram from an audio file
def extract_mel_spectrogram(audio_path, n_mels=128, n_fft=2048, hop_length=512):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

# Function to pad spectrograms and add a channel dimension for prediction
def prepare_spectrogram(spectrogram, max_length):
    padded_spectrogram = librosa.util.fix_length(spectrogram, size=max_length, axis=1)
    padded_spectrogram = np.expand_dims(padded_spectrogram, axis=-1)  # Add a channel dimension
    padded_spectrogram = np.expand_dims(padded_spectrogram, axis=0)  # Add a batch dimension
    return padded_spectrogram

# Initialize LabelEncoder with the same labels used during training
labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
label_encoder = LabelEncoder()
label_encoder.fit(labels)

def predict_emotions_in_directory(audio_directory, model, max_length, label_encoder):
    predictions = {}
    for subdir, dirs, files in os.walk(audio_directory):
        for file in files:
            if file.endswith('.wav'):
                try:
                    audio_path = os.path.join(subdir, file)
                    spectrogram = extract_mel_spectrogram(audio_path)
                    print("Spectrogram shape:", spectrogram.shape)
                    spectrogram_prepared = prepare_spectrogram(spectrogram, max_length)
                    print("Prepared spectrogram shape:", spectrogram_prepared.shape)
                    prediction = model.predict(spectrogram_prepared)
                    predicted_class_index = np.argmax(prediction, axis=1)[0]
                    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
                    predictions[file] = predicted_label
                except Exception as e:
                    print(f"Could not process file {file}: {e}")
    return predictions


# Path to the directory containing audio files for prediction
audio_directory = 'Audio_Song_Actors_01-24'

# Execute prediction for all audio files in the directory
predictions = predict_emotions_in_directory(audio_directory, model, max_length, label_encoder)

# Print out predictions for each file
for file, predicted_label in predictions.items():
    print(f"{file}: Predicted emotion - {predicted_label}")
