from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
dataset_path = 'Audio_Song_Actors_01-24'

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

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(np.array(features))

# Initialize and train the Support Vector Machine
print("Training Support Vector Machine...")
svm = SVC(probability=True, random_state=0)
svm.fit(scaled_features, encoded_labels)

# Save the SVM model to disk
dump(svm, 'speech_emotion_recognition_svm.joblib')
print("Support Vector Machine trained and saved.")

# Load the trained SVM model
svm = load('speech_emotion_recognition_svm.joblib')

# Function to predict the most common emotion from an audio file
def predict_emotion(audio_path, svm, label_encoder, scaler):
    features = extract_features(audio_path)
    scaled_features = scaler.transform([features])  # Scale features
    predicted_probabilities = svm.predict_proba(scaled_features)
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
            predicted_emotion = predict_emotion(audio_file_path, svm, label_encoder, scaler)
            predictions[audio_file_path] = predicted_emotion
            print(f"{file}: {predicted_emotion}")

# Save the predictions to a file
with open('svm_emotion_predictions.txt', 'w') as f:
    for path, emotion in predictions.items():
        f.write(f"{path}: {emotion}\n")

print("Predictions made for all files and saved to svm_emotion_predictions.txt")
