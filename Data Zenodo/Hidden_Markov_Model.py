from collections import Counter
import matplotlib.pyplot as plt
from joblib import load
from joblib import dump
from hmmlearn import hmm
import numpy as np
import os
import librosa

# Define a function to extract features from audio files
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
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
                # Extract the emotion part from the file name
                emotion_code = file.split('-')[2]
                emotion = emotion_dict.get(emotion_code, 'unknown')

                # If the emotion is one we're interested in, process the file
                if emotion != 'unknown':
                    # Extract features from the audio file
                    mfccs = extract_features(os.path.join(subdir, file))
                    features.append(mfccs)
                    labels.append(emotion)
            except Exception as e:
                print(f"Could not process file {file}: {e}")

# Convert the collected features and labels to a suitable format for hmmlearn
X = np.concatenate(features)
lengths = [len(f) for f in features]

# Create and train an HMM
# Here we use a GaussianHMM for simplicity; real applications may require more complex models
model = hmm.GaussianHMM(n_components=len(emotion_dict), covariance_type="diag", n_iter=1000)

print("Training model...")
model.fit(X, lengths)

# Save the model for future use
# You could use joblib to save the model to disk
dump(model, 'hmm_speech_emotion_recognition.joblib')

print("Model trained and saved as 'hmm_speech_emotion_recognition.joblib'")

# Create and train an HMM
model = hmm.GaussianHMM(n_components=len(emotion_dict), covariance_type="diag", n_iter=1000)
print("Training model...")
model.fit(X, lengths)

# Save the model for future use
dump(model, 'hmm_speech_emotion_recognition.joblib')
print("Model trained and saved as 'hmm_speech_emotion_recognition.joblib'")

# Visualize and save the state transition matrix
plt.figure(figsize=(14, 8))
plt.imshow(model.transmat_, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('State Transition Matrix')
plt.xlabel('From State')
plt.ylabel('To State')
plt.xticks(np.arange(len(emotion_dict)), list(emotion_dict.values()), rotation=45)
plt.yticks(np.arange(len(emotion_dict)), list(emotion_dict.values()))
plt.savefig('hmm_state_transition_matrix.png')
plt.close()

print("State transition matrix saved as 'hmm_state_transition_matrix.png'")

####################### Predictions ########################
# Load the trained model
model = load('hmm_speech_emotion_recognition.joblib')

# Function to predict the most common emotion from an audio file
def predict_most_common_emotion(audio_path, model, emotion_dict):
    features = extract_features(audio_path)
    hidden_states = model.predict(features)
    emotions = [list(emotion_dict.values())[state] for state in hidden_states]
    emotion_counts = Counter(emotions)
    most_common_emotion = emotion_counts.most_common(1)[0][0]
    return most_common_emotion

# Dictionary to store the prediction for each file
predictions = {}

# Predict the sequence of emotions for each audio file in the dataset
for subdir, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav'):
            # Full path to the current audio file
            audio_file_path = os.path.join(subdir, file)

            # Predict the most common emotion for the current audio file
            most_common_emotion = predict_most_common_emotion(audio_file_path, model, emotion_dict)
            predictions[audio_file_path] = most_common_emotion
            print(f"{file}: {most_common_emotion}")

# Save the predictions to a file
with open('hmm_emotion_predictions.txt', 'w') as f:
    for path, emotion in predictions.items():
        f.write(f"{path}: {emotion}\n")

print("All predictions made and saved to hmm_emotion_predictions.txt")
