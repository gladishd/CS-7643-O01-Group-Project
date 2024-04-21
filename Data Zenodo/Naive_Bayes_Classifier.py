from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import numpy as np
import librosa
import os
from collections import Counter

# Define a function to extract MFCC features from an audio file
def extract_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)

# Define the path to your dataset
dataset_path = 'Data Zenodo/Audio_Song_Actors_01-24'

# Placeholder for the feature extraction results
features = []
labels = []

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

# Extract features from each audio file and assign labels
for subdir, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav'):
            try:
                emotion_code = file.split('-')[2]
                emotion = emotion_dict.get(emotion_code, 'unknown')
                if emotion != 'unknown':
                    mfccs = extract_features(os.path.join(subdir, file))
                    features.append(mfccs)
                    labels.append(emotion)
            except Exception as e:
                print(f"Could not process file {file}: {e}")

# Encode the labels as integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Convert the collected features to a numpy array
X = np.array(features)

# Initialize and train the Naive Bayes Classifier
print("Training Naive Bayes Classifier...")
nb = GaussianNB()
nb.fit(X, encoded_labels)

# Save the Naive Bayes model to disk
dump(nb, 'speech_emotion_recognition_nb.joblib')
print("Naive Bayes Classifier trained and saved.")

# Load the trained Naive Bayes model
nb = load('speech_emotion_recognition_nb.joblib')

# Function to predict the most common emotion from an audio file
def predict_emotion(audio_path, nb, label_encoder):
    features = extract_features(audio_path)
    features = features.reshape(1, -1)  # Reshape for single sample prediction
    predicted_probabilities = nb.predict_proba(features)
    most_likely_emotion_index = np.argmax(predicted_probabilities)
    most_likely_emotion = label_encoder.inverse_transform([most_likely_emotion_index])
    return most_likely_emotion[0]

# Dictionary to store the prediction for each file
predictions = {}

# Predict the most likely emotion for each audio file in the dataset
print("Making predictions on audio files...")
for subdir, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav'):
            audio_file_path = os.path.join(subdir, file)
            predicted_emotion = predict_emotion(audio_file_path, nb, label_encoder)
            predictions[audio_file_path] = predicted_emotion
            print(f"{file}: {predicted_emotion}")

# Save the predictions to a file
with open('nb_emotion_predictions.txt', 'w') as f:
    for path, emotion in predictions.items():
        f.write(f"{path}: {emotion}\n")

print("Predictions made for all files and saved to nb_emotion_predictions.txt")
