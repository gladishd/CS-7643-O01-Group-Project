import os
import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split

# Step up to the parent folder
#os.chdir('..')

# Path to the Audio_Speech_Actors_01-24 folder
path_to_actors = 'Audio_Speech_Actors_01-24'

# Get .wav files from Actor_01 and Actor_02
actor_folders = ['Actor_01', 'Actor_02']
wav_files = []
for actor_folder in actor_folders:
    actor_path = os.path.join(path_to_actors, actor_folder)
    files = [os.path.join(actor_path, file) for file in os.listdir(actor_path) if file.endswith('.wav')]
    wav_files.extend(files)

# Select 10 wav files randomly
selected_files = np.random.choice(wav_files, 10, replace=False)

# Function to extract features from audio file
def extract_features(file_path):
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, sr=None)
    # Extract features
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    tonnetz = librosa.effects.harmonic(y=audio)
    tonnetz = librosa.feature.tonnetz(y=tonnetz, sr=sample_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    # Aggregate the features
    features = np.hstack([np.mean(mfcc, axis=1), np.mean(chroma, axis=1), np.mean(mel, axis=1),
                          np.mean(tonnetz, axis=1), np.mean(spectral_contrast, axis=1)])
    return features

# Extract features for selected files
features = np.array([extract_features(file) for file in selected_files])

# Dummy labels (for demonstration, replace with actual emotion labels if available)
labels = np.random.randint(0, 2, 10)  # binary labels for simplicity

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build the 1D CNN model
model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
    Conv1D(64, kernel_size=5, activation='relu'),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X_train[..., np.newaxis], y_train, epochs=10, validation_data=(X_test[..., np.newaxis], y_test))


import matplotlib.pyplot as plt

# Train the model and save the history
history = model.fit(X_train[..., np.newaxis], y_train, epochs=10, validation_data=(X_test[..., np.newaxis], y_test))

# Plotting training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plotting training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Issa Demirci And Yazici Article, Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.tight_layout()
plt.savefig('issa_demirci_and_yazici_article_model_loss_1.png')
plt.show()


import matplotlib.pyplot as plt

# Function to plot training history
def plot_training_history(history):
    # Plotting the accuracy
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plotting the loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Issa Demirci And Yazici Article, Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('issa_demirci_and_yazici_article_model_loss_2.png')
    plt.tight_layout()
    plt.show()

# Train the model and capture the history
history = model.fit(X_train[..., np.newaxis], y_train, epochs=10, validation_data=(X_test[..., np.newaxis], y_test))

# Plot the training and validation loss and accuracy
plot_training_history(history)


import librosa.display

# Function to plot the waveform, spectrogram and MFCC of an audio file
def plot_audio_analysis(file_path):
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Plot the waveform
    plt.figure(figsize=(18, 6))
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(audio, sr=sample_rate)
    plt.title('Issa Demirci And Yazici Article Waveform')


    # Plot the spectrogram
    plt.subplot(3, 1, 2)
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency Spectrogram')

    # Plot the MFCC
    plt.subplot(3, 1, 3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')

    plt.tight_layout()
    plt.savefig('issa_demirci_and_yazici_article_waveform_spectrogram_mfcc.png')
    plt.show()

# Select a sample file to plot
sample_file = selected_files[0]
plot_audio_analysis(sample_file)

import librosa.display

# Additional visualization for Chromagram
def plot_chromagram(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    chromagram = librosa.feature.chroma_stft(y=audio, sr=sample_rate)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm', sr=sample_rate)
    plt.title('Issa Demirci And Yazici Article Chromagram')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('issa_demirci_and_yazici_article_chromagram.png')
    plt.show()

# Additional visualization for Spectral Contrast
def plot_spectral_contrast(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectral_contrast, x_axis='time', cmap='coolwarm')
    plt.title('Issa Demirci And Yazici Article Spectral Contrast')
    plt.ylabel('Frequency Bands')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('issa_demirci_and_yazici_article_spectral_contrast.png')
    plt.show()

# Additional visualization for Tonnetz
def plot_tonnetz(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    tonnetz = librosa.effects.harmonic(audio)
    tonnetz = librosa.feature.tonnetz(y=tonnetz, sr=sample_rate)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(tonnetz, x_axis='time', cmap='coolwarm')
    plt.title('Issa Demirci And Yazidi Article Tonnetz')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('issa_demirci_and_yazici_article_plot_tonnetz.png')
    plt.show()

# Applying additional visualizations on the sample file
plot_chromagram(sample_file)
plot_spectral_contrast(sample_file)
plot_tonnetz(sample_file)
