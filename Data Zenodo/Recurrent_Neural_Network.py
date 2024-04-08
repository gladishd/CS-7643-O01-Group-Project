# Recurrent_Neural_Network.py
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Parameters
n_mels = 128
hop_length = 512
n_fft = 2048
n_classes = 8  # Assuming 8 emotional categories as before
max_files = 100  # Adjust based on your dataset size for experimentation

# Define your dataset path
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

# Function to extract features from audio file
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db.T  # Transpose to make time steps as the first dimension

# Loading dataset
features, labels = [], []
for subdir, dirs, files in os.walk(dataset_path):
    for file in files[:max_files]:
        if file.endswith('.wav'):
            try:
                emotion_code = file.split('-')[2]
                emotion = emotion_dict.get(emotion_code, None)
                if emotion:
                    file_path = os.path.join(subdir, file)
                    mel_spectrogram_db_T = extract_features(file_path)
                    features.append(mel_spectrogram_db_T)
                    labels.append(emotion)
            except Exception as e:
                print(f"Error processing {file}: {e}")

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels, num_classes=n_classes)

# Padding sequences to have the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences
features_padded = pad_sequences(features, padding='post')

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(features_padded, categorical_labels, test_size=0.2, random_state=42)

from tensorflow.keras.layers import Input
model = Sequential([
    Input(shape=(None, n_mels)),  # Adjust based on your input dimensions
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Save the model
model.save('path_to_save_LSTM_model.h5')

# To visualize the training process, plot the history for accuracy and loss
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('RNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("rnn_model accuracy.png")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('RNN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("rnn_model loss.png")
plt.show()

import matplotlib.pyplot as plt
from matplotlib.table import Table

def plot_model_summary(summary):
    """
    Plots the model summary in a table format.

    Args:
    - summary: List of tuples containing layer information,
               typically obtained from parsing model.summary().
    """
    fig, ax = plt.subplots(figsize=(10, len(summary) * 0.5))  # Adjust figure size
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = len(summary) + 1, 4  # Adding one for the header row
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add the header
    tb.add_cell(0, 0, width, height, text='Layer (type)', loc='center', facecolor='lightblue')
    tb.add_cell(0, 1, width, height, text='Output Shape', loc='center', facecolor='lightblue')
    tb.add_cell(0, 2, width, height, text='Param #', loc='center', facecolor='lightblue')
    tb.add_cell(0, 3, width, height, text='Connected to', loc='center', facecolor='lightblue')

    # Adding data rows
    for i, row in enumerate(summary, start=1):
        for j, cell in enumerate(row):
            tb.add_cell(i, j, width, height, text=cell, loc='center')

    ax.add_table(tb)
    plt.savefig("RNN model summary.png")
    plt.show()

# Example usage with your model summary
model_summary = [
    ("lstm (LSTM)", "(None, None, 128)", "131584", ""),
    ("dropout (Dropout)", "(None, None, 128)", "0", ""),
    ("lstm_1 (LSTM)", "(None, 128)", "131584", ""),
    ("dense (Dense)", "(None, 64)", "8256", ""),
    ("dropout_1 (Dropout)", "(None, 64)", "0", ""),
    ("dense_1 (Dense)", "(None, 8)", "520", "")
]

plot_model_summary(model_summary)
