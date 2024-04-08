

import librosa
import numpy as np
from keras.models import load_model
import os

model_path = 'emotion_model.h5'

# Check if the model file exists before loading
if not os.path.exists(model_path):
    raise Exception(f"The model file {model_path} does not exist. Please check the path.")

# Load your pre-trained emotion recognition model
model = load_model(model_path)

# Function to preprocess audio file
def preprocess_audio(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    # Extract features (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Scale features
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    # Return preprocessed features
    return mfcc_scaled

# Function to predict emotion from audio file
def predict_emotion(audio_path):
    # Preprocess the audio to get features
    features = preprocess_audio(audio_path)
    # Predict emotion
    prediction = model.predict(np.array([features]))
    # Assuming the output of your model is a one-hot encoded vector for the emotions
    emotion_labels = ['angry', 'happy', 'sad', 'neutral']  # Update with your model's labels
    # Get the index of the highest probability
    predicted_emotion = emotion_labels[np.argmax(prediction)]
    return predicted_emotion

# Function to respond to the emotion
def respond_to_emotion(emotion):
    # Logic to respond to different emotions
    responses = {
        'angry': 'I am sensing some frustration. How can I assist you better?',
        'happy': 'Glad to hear you’re in high spirits! How can I help you today?',
        'sad': 'It sounds like a tough time. I’m here to help you.',
        'neutral': 'How can I assist you today?'
    }
    # Return a response based on the predicted emotion
    return responses.get(emotion, 'I am here to help you.')

# Main function to run the empathic voice interface
def main(audio_path):
    # Predict the emotion of the audio
    emotion = predict_emotion(audio_path)
    # Generate a response based on the emotion
    response = respond_to_emotion(emotion)
    # For now, just print the response. In a real application, this could be a spoken response.
    print(response)

if __name__ == '__main__':
    # Replace 'path_to_audio_file.wav' with the path to an actual audio file
    main('path_to_audio_file.wav')
