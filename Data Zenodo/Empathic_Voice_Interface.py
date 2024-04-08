

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
    main('Audio_Song_Actors_01-24/angry/angry.wav')
    main('Audio_Song_Actors_01-24/happy/happy.wav')
    main('Audio_Song_Actors_01-24/neutral/neutral.wav')
    main('Audio_Song_Actors_01-24/sad/sad.wav')

import matplotlib.pyplot as plt

def plot_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.savefig('evi_mfcc.png')
    plt.tight_layout()
    plt.show()

# Call this function with the path to "an" audio file
plot_mfcc('Audio_Song_Actors_01-24/angry/angry.wav')
plot_mfcc('Audio_Song_Actors_01-24/happy/happy.wav')
plot_mfcc('Audio_Song_Actors_01-24/neutral/neutral.wav')
plot_mfcc('Audio_Song_Actors_01-24/sad/sad.wav')

def plot_emotion_distribution(audio_path):
    features = preprocess_audio(audio_path)
    prediction = model.predict(np.array([features]))[0]
    emotion_labels = ['angry', 'happy', 'sad', 'neutral']

    plt.bar(emotion_labels, prediction)
    plt.title('EVI Emotion Prediction Distribution')
    plt.ylabel('Probability')
    plt.savefig('evi_emotiona_prediction_distribution.png')
    plt.show()

plot_emotion_distribution('Audio_Song_Actors_01-24/angry/angry.wav')
plot_emotion_distribution('Audio_Song_Actors_01-24/happy/happy.wav')
plot_emotion_distribution('Audio_Song_Actors_01-24/neutral/neutral.wav')
plot_emotion_distribution('Audio_Song_Actors_01-24/sad/sad.wav')

from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Example usage, assuming y_test and predictions are available
# plot_confusion_matrix(y_test, predictions, emotion_labels)

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

# Call this function with the history object from the model.fit() method
# plot_training_history(history)
