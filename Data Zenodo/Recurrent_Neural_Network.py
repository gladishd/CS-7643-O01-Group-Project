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

# Defining the RNN Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(None, n_mels)),  # Adjust the input shape
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
model.save('path_to_save_LSTM_model')

# To visualize the training process, plot the history for accuracy and loss
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("rnn_model accuracy.png")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("rnn_model loss.png")
plt.show()
