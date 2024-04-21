from sklearn.mixture import GaussianMixture
from joblib import dump, load
import numpy as np
import librosa
import os
from collections import Counter

# Define a function to extract MFCC features from an audio file
def extract_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

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
                    features.extend(mfccs)
                    labels.extend([emotion] * len(mfccs))
            except Exception as e:
                print(f"Could not process file {file}: {e}")

# Convert the collected features to a numpy array
X = np.array(features)

# Initialize and train the Gaussian Mixture Model
print("Training Gaussian Mixture Model...")
gmm = GaussianMixture(n_components=len(emotion_dict), covariance_type='full', max_iter=200, random_state=0)
gmm.fit(X)

# Save the GMM model to disk
dump(gmm, 'speech_emotion_recognition_gmm.joblib')
print("Gaussian Mixture Model trained and saved.")

# Load the trained GMM model
gmm = load('speech_emotion_recognition_gmm.joblib')

# Function to predict the most common emotion from an audio file
def predict_most_common_emotion(audio_path, gmm, emotion_dict):
    features = extract_features(audio_path)
    probabilities = gmm.predict_proba(features)
    average_probabilities = np.mean(probabilities, axis=0)
    # Fix: Create an index with leading zeros for emotion_dict keys
    most_common_emotion_index = f"{np.argmax(average_probabilities) + 1:02d}"
    most_common_emotion = emotion_dict[most_common_emotion_index]
    return most_common_emotion


# Dictionary to store the prediction for each file
predictions = {}

# Predict the most likely emotion for each audio file in the dataset
print("Making predictions on audio files...")
for subdir, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav'):
            audio_file_path = os.path.join(subdir, file)
            most_common_emotion = predict_most_common_emotion(audio_file_path, gmm, emotion_dict)
            predictions[audio_file_path] = most_common_emotion
            print(f"{file}: {most_common_emotion}")

# Save the predictions to a file
with open('gmm_emotion_predictions.txt', 'w') as f:
    for path, emotion in predictions.items():
        f.write(f"{path}: {emotion}\n")

print("Predictions made for all files and saved to gmm_emotion_predictions.txt")
