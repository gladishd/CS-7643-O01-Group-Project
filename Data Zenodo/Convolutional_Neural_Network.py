import librosa
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt

# TensorFlow Model Optimization Toolkit specific imports for sparsity
sparsity = tfmot.sparsity.keras

# Parameters
n_mels = 128
n_fft = 2048
hop_length = 512
n_classes = 8
max_files = 10  # Limit the number of files for quick testing

# Define the path to your dataset
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

# Function to extract Mel-spectrogram
def extract_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

spectrograms = []
labels = []
files_processed = 0

# Extract features and labels
for subdir, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav') and files_processed < max_files:
            try:
                emotion_code = file.split('-')[2]
                emotion = emotion_dict.get(emotion_code, 'unknown')
                if emotion != 'unknown':
                    filepath = os.path.join(subdir, file)
                    S_DB = extract_mel_spectrogram(filepath)
                    spectrograms.append(S_DB)
                    labels.append(emotion)
                    files_processed += 1
            except Exception as e:
                print(f"Could not process file {file}: {e}")
        if files_processed >= max_files:
            break
    if files_processed >= max_files:
        break

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Padding and reshaping spectrograms
max_length = max(s.shape[1] for s in spectrograms)
X = np.array([librosa.util.fix_length(s, size=max_length, axis=1) for s in spectrograms])
X = X[..., np.newaxis]  # Adding a channel dimension

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)

# CNN Model Definition using Sequential API with fixed input shape
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(n_mels, max_length, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary to verify structure
model.summary()

# Train the model with callbacks for TensorBoard and possibly early stopping and model checkpointing
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    # Add early stopping and model checkpoint callbacks here if needed
]

# Training the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=callbacks)

# Saving the model and weights
model.save('path_to_save_model')
model.save_weights('path_to_save_weights.h5')

# Remove pruning wrappers and save the pruned model weights
final_model = sparsity.strip_pruning(model)
final_model.save_weights('pruned_cnn_speech_emotion_recognition_weights.h5')

# Save the pruned model's architecture to a JSON file
with open('pruned_cnn_speech_emotion_recognition_model.json', 'w') as f:
    f.write(final_model.to_json())

print("Pruned CNN model weights and architecture saved.")

####Visualizations "2"

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=callbacks)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('cnnmodel_accuracy.png')  # Saving the figure
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('cnnmodel_loss.png')  # Saving the figure
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming your model is named "model" and you've already split your data into training and testing sets
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = y_test

# Generate the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plotting
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('cnnconfusion_matrix.png')  # Saving the figure
plt.show()

def plot_predictions(images, predictions, true_labels, class_names):
    plt.figure(figsize=(15, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        col = 'green' if predictions[i] == true_labels[i] else 'red'
        plt.xlabel(f'Pred: {class_names[predictions[i]]}\nTrue: {class_names[true_labels[i]]}', color=col)
    plt.tight_layout()
    plt.savefig('cnnpredictions_visualization.png')  # Saving the figure
    plt.show()

# Assuming you've decoded your labels back to their original values
class_names = list(emotion_dict.values())
plot_predictions(X_test, predicted_classes, true_classes, class_names)

from sklearn.metrics import classification_report

print(classification_report(true_classes, predicted_classes, target_names=class_names))
