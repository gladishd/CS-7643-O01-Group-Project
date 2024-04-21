import os  # Add this import statement
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, LSTM

# "Ensure" your preprocess_audio function is defined here as well
def preprocess_audio(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Transpose the result to align with the expected input format and take the mean
    mfccs_processed = np.mean(mfccs.T,axis=0)

    return mfccs_processed

def create_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        Conv1D(64, kernel_size=5, activation='relu'),
        Dropout(0.5),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='softmax')  # Make sure y_train is accessible
    ])
    return model

def load_and_preprocess_data():
    dataset_path = 'Data Zenodo/Audio_Song_Actors_01-24'
    emotions = ['happy', 'sad', 'angry', 'neutral']
    features = []
    labels = []

    for emotion in emotions:
        # Construct the path to the emotion directory
        emotion_path = os.path.join(dataset_path, emotion)
        # Check if the emotion directory exists
        if not os.path.isdir(emotion_path):
            print(f"Directory does not exist: {emotion_path}")
            continue

        files = glob.glob(os.path.join(emotion_path, '*.wav'))
        if not files:
            print(f"No .wav files found in {emotion_path}")
            continue

        for file in files:
            mfcc = preprocess_audio(file)  # Ensure this function returns an expected array
            features.append(mfcc)
            labels.append(emotion)

    if not features or not labels:
        print("No features or labels have been loaded. Check the dataset path and structure.")
        return None, None, None, None

    features = np.array(features)
    labels = np.array(labels)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_onehot = to_categorical(labels_encoded)

    X_train, X_test, y_train, y_test = train_test_split(features, labels_onehot, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_and_preprocess_data()

if X_train is not None:
    # Proceed with model creation, compilation, and training
    # Ensure y_train is accessible for model creation
    model = create_model((X_train.shape[1], 1))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))
    model_path = 'emotion_model.h5'
    model.save(model_path)
else:
    print("Data loading failed. Training aborted.")
