# Start out with this mp3
# Free Music Sound Effects Download - Pixabay
# https://pixabay.com/sound-effects/search/music/
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, classification_report



# Path to your MP3 file
audio_path = 'cinematic-music-sketches-11-cinematic-percussion-sketch-116186.mp3'

# Load the audio file
y, sr = librosa.load(audio_path)

# Display the waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig("waveform.png")
plt.show()

# Generate a Mel-spectrogram
# A Mel-spectrogram is a spectrogram where the frequencies are converted to the Mel scale.
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)


# Convert to log scale (dB)
log_S = librosa.power_to_db(S, ref=np.max)

# Display the Mel-spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel-spectrogram')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()
plt.savefig("mel_spectrogram.png")
plt.show()

# Note: This script is the starting point. For emotion recognition, you will need to integrate this with deep learning models.

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Calculate the chroma feature
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Compute spectral contrast
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

from hmmlearn import hmm

# Define the number of states for HMM or the number of emotion categories for GMM and classification
number_of_states = 5 # for example, if you have 5 different states in your HMM
number_of_emotions = 7 # for example, if you have 7 different emotions to classify
# Example labels array, with '0' for 'neutral', '1' for 'happy', etc.
# This is just an example. In a real dataset, these labels would come from your data annotations.
# Create a labels array with a random label for each sample
labels = np.random.randint(0, number_of_emotions, size=mfccs.shape[1])


model = hmm.GaussianHMM(n_components=number_of_states)
model.fit(mfccs.T)

from sklearn.mixture import GaussianMixture

# Train a GMM
gmm = GaussianMixture(n_components=number_of_emotions)
gmm.fit(mfccs.T)

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Train a Naive Bayes classifier
nb_model = GaussianNB()
nb_model.fit(mfccs.T, labels)

# Train an SVM
svm_model = SVC()
svm_model.fit(mfccs.T, labels)

import tensorflow as tf
from tensorflow.keras import layers, models
time_steps = S.shape[1]  # where S is the Mel-spectrogram

# Example of defining a simple CNN model
model = models.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, time_steps, 1)),
layers.MaxPooling2D((2, 2)),
layers.Flatten(),
layers.Dense(64, activation='relu'),
layers.Dense(number_of_emotions, activation='softmax')
])



# Compile and train the model
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
# Splitting data into training and testing sets (this is just an example)
from sklearn.model_selection import train_test_split







# Load the audio file
audio_path = 'cinematic-music-sketches-11-cinematic-percussion-sketch-116186.mp3'
y, sr = librosa.load(audio_path)

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfccs = np.transpose(mfccs)  # Transpose so that the shape is (time_steps, n_mfcc)

# Assuming 7 emotions, generate dummy labels for demonstration purposes
labels = np.random.randint(0, 7, size=len(mfccs))

# Reshape MFCCs to add a channel dimension (samples, time_steps, features, 1)
mfccs_reshaped = np.expand_dims(mfccs, axis=-1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mfccs_reshaped, labels, test_size=0.2, random_state=42)

# Define the CNN model architecture
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(7, activation='softmax')  # Assuming 7 possible emotions
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# Make predictions on the test set
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predicted_classes)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Evaluate precision, recall, and F1 score
report = classification_report(y_test, predicted_classes, target_names = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised"]
)
print(report)




def load_and_augment(audio_path, n_steps=4, noise_factor=0.005, n_mfcc=13):
    y, sr = librosa.load(audio_path)
    y_pitched = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    y_noisy = y + noise_factor * np.random.randn(len(y))

    mfcc_original = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    mfcc_pitched = librosa.feature.mfcc(y=y_pitched, sr=sr, n_mfcc=n_mfcc).T
    mfcc_noisy = librosa.feature.mfcc(y=y_noisy, sr=sr, n_mfcc=n_mfcc).T

    return mfcc_original, mfcc_pitched, mfcc_noisy

audio_path = 'cinematic-music-sketches-11-cinematic-percussion-sketch-116186.mp3'
mfcc_original, mfcc_pitched, mfcc_noisy = load_and_augment(audio_path)

# Dummy labels for demonstration
labels = np.zeros((mfcc_original.shape[0],))
labels = np.concatenate([labels, labels, labels])  # Replicating for augmented data

# Combine all MFCC features and labels
X = np.concatenate([mfcc_original, mfcc_pitched, mfcc_noisy], axis=0)
y = labels

# Reshape for CNN input
X_reshaped = X[..., np.newaxis]  # Adding a channel dimension

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

input_shape = X_train.shape[1:]  # Getting input shape for the model

# Defining the model with Input layer
# Assuming X_train is reshaped properly for Conv1D
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),  # (time_steps, n_mfcc)
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(7, activation='softmax')  # Assuming 7 emotions/categories
])


