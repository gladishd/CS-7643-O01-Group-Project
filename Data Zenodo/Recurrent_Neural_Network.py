# Recurrent_Neural_Network.py
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.sequence import pad_sequences

import seaborn as sns
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from sklearn.metrics import confusion_matrix, classification_report


# Parameters
n_mels = 128
hop_length = 512
n_fft = 2048
n_classes = 8  # Assuming 8 emotional categories as before
max_files = 100  # Adjust based on your dataset size for experimentation
dataset_path = 'Audio_Song_Actors_01-24'  # Define your dataset path

# Define your dataset path
dataset_path = 'Audio_Song_Actors_01-24'

# Emotion labels
emotion_dict = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Function to extract features from audio file
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db.T

# Load dataset, encode labels, and pad sequences
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
            except IndexError as e:
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
plt.savefig("new rnn accuracy.png")
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

from tensorflow.keras.layers import Bidirectional, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

# Modify the model to add Bidirectional LSTM layers and TimeDistributed Dense layer
model = Sequential([
    Input(shape=(None, n_mels)),  # None represents variable length sequences
    Bidirectional(LSTM(128, return_sequences=True)),  # Bidirectional LSTM layer
    Dropout(0.2),
    Bidirectional(LSTM(128)),  # Another Bidirectional LSTM layer
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])

# Recompile the model with the new changes
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# Train the model with early stopping
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]  # Add early stopping to callbacks
)

# After training, plot the new accuracy and loss graphs

# Plot accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('RNN Model Accuracy with Bidirectional LSTM')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig("rnn_model_bidirectional_accuracy.png")
plt.show()

# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('RNN Model Loss with Bidirectional LSTM')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig("rnn_model_bidirectional_loss.png")
plt.show()


# Add additional metrics for performance analysis
from tensorflow.keras.metrics import Precision, Recall

# Modify the model compilation to include precision and recall
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

# Train the model with the updated metrics
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# To address the "list index out of range" error, ensure proper file parsing
def parse_file_name(file_name):
    # Modify the parsing logic according to your file naming convention
    # This is a placeholder function and needs to be adjusted
    parts = file_name.split('-')
    if len(parts) >= 3:
        emotion_code = parts[2]
    else:
        print(f"Error with file name {file_name}: Not enough parts")
        emotion_code = None
    return emotion_code

# Update dataset loading with the new file parsing logic
features, labels = [], []
for subdir, dirs, files in os.walk(dataset_path):
    for file in files[:max_files]:
        if file.endswith('.wav'):
            emotion_code = parse_file_name(file)
            emotion = emotion_dict.get(emotion_code, None)
            if emotion:
                file_path = os.path.join(subdir, file)
                try:
                    mel_spectrogram_db_T = extract_features(file_path)
                    features.append(mel_spectrogram_db_T)
                    labels.append(emotion)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Rest of the code remains the same

# Visualize metrics
plt.figure(figsize=(8, 5))
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Test Precision')
plt.title('RNN Model Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend()
plt.savefig("rnn_model_precision.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Test Recall')
plt.title('RNN Model Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend()
plt.savefig("rnn_model_recall.png")
plt.show()

# Save the model in the preferred Keras format
model.save('path_to_save_LSTM_model')

# If the file processing issue persists, further investigate the dataset directory structure and file naming convention.

# Additional code to enhance Recurrent_Neural_Network.py

# Import additional necessary packages
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime

# Adding a Learning Rate Scheduler
from tensorflow.keras.callbacks import LearningRateScheduler
import math

# Function for learning rate decay
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

# Include Learning Rate Scheduler in the callbacks
lr_scheduler = LearningRateScheduler(step_decay)

# Model Checkpoint callback to save the best model
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

# TensorBoard callback for visualization
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Compile the model with an adjustable learning rate
opt = Adam(learning_rate=0.001)
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

# Train the model with new callbacks
history = model.fit(
    X_train, y_train,
    epochs=50,  # Increase epochs for better training
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler, checkpoint, tensorboard_callback]
)

# Evaluate the model using the test data
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test)

# Display evaluation results
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")

# Plot precision and recall over epochs
plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Test Precision')
plt.title('RNN Model Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Test Recall')
plt.title('RNN Model Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.tight_layout()
plt.savefig("rnn_model_precision_recall.png")
plt.show()

# Load the best saved model
from tensorflow.keras.models import load_model
best_model = load_model('best_model.h5')

# Save the best model with the preferred Keras format
best_model.save('best_LSTM_model.keras')

# Start TensorBoard within the notebook
# %load_ext tensorboard
# %tensorboard --logdir logs/fit


# Additional imports
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Regularization with L2 and Batch Normalization
regularizer = l2(0.001)

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=regularizer), input_shape=(None, n_mels)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Bidirectional(LSTM(128, kernel_regularizer=regularizer)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=regularizer),
    BatchNormalization(),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

# Compile the model with the Adam optimizer and learning rate decay
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

# Real-time Data Augmentation with librosa (for audio data)
# This is just a pseudo-code snippet, customize it for your needs
def augment_data(y, sr):
    # Apply techniques like noise injection, time-shifting, speed change
    return y_augmented

# Update feature extraction to include augmentation
def extract_features(audio_path, augment=False):
    y, sr = librosa.load(audio_path)
    if augment:
        y = augment_data(y, sr)
    # ... rest of the feature extraction code ...

# Classification Report and Confusion Matrix
def print_classification_report_and_confusion_matrix(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig("new rnn predicted classification report.png")
    plt.show()

# After model training
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print_classification_report_and_confusion_matrix(y_true, y_pred)

# Implementing a custom callback
from tensorflow.keras.callbacks import Callback

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Custom actions at the end of each epoch
        pass

custom_callback = CustomCallback()

# Add custom callback to the fit method
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler, checkpoint, tensorboard_callback, custom_callback]
)

# Make sure you have your environment set up for TensorBoard if you want to use it
# To run TensorBoard, use the following command in your terminal:
# tensorboard --logdir=path_to_your_logs


# Learning rate scheduling
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = Adam(learning_rate=lr_schedule)

# Model checkpointing
checkpoint_filepath = 'best_model_checkpoint.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

# Compile the model with regularization and optimizer with learning rate scheduler
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

# Train the model with checkpointing
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint_callback, tensorboard_callback]
)

# Plot confusion matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('RNN Predicted labels')
plt.ylabel('True labels')
plt.savefig("new rnn predicted label report.png")
plt.show()




from itertools import product


import tensorflow as tf



# Additional functionalities to append to Recurrent_Neural_Network.py

# Import additional required modules
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report
import datetime
import seaborn as sns

# Learning Rate Scheduler function
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Add a ModelCheckpoint to save the model during training at the point when it performs the best
checkpoint = ModelCheckpoint('model_best_checkpoint.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# TensorBoard callback for advanced visualizations
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Compile the model with a learning rate scheduler
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

# Train the model with the additional callbacks
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(monitor='val_loss', patience=3),
               LearningRateScheduler(scheduler),
               checkpoint,
               tensorboard_callback]
)

# After training, plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('RNN True label')
    plt.xlabel('Predicted label')
    plt.savefig("new rnn predicted confusion report.png")
    plt.show()

# Convert predictions from one-hot encoding to labels
predicted_classes = np.argmax(model.predict(X_test), axis=1)
true_classes = np.argmax(y_test, axis=1)

# Call the function to plot confusion matrix
plot_confusion_matrix(true_classes, predicted_classes, classes=list(emotion_dict.values()))

from sklearn.metrics import classification_report

# Correct the number of classes and names to match the data
emotion_dict = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
target_names = [emotion_dict[str(i).zfill(2)] for i in range(1, 9)]

# Example usage of the classification_report
true_classes = [0, 1, 2, 3, 4, 5, 6, 7]  # Example true classes
predicted_classes = [0, 2, 2, 3, 4, 5, 6, 7]  # Example predicted classes
report = classification_report(true_classes, predicted_classes, target_names=target_names)
print(report)


# Save the TensorBoard logs
print("\nTo view TensorBoard, navigate to 'logs/fit' directory and run:")
print("tensorboard --logdir=.")

# Add a custom callback for additional monitoring
class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Log the learning rate
        lr = self.model.optimizer.lr
        print(f'Learning rate after epoch {epoch}: {lr}')

custom_callback = CustomCallback()

# Update the model fitting to include the custom callback
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[custom_callback]  # include other callbacks as well
)












# Print a classification report
report = classification_report(y_true, y_pred, target_names=emotion_dict.values())
print(report)

# Implementing a custom callback for additional insights
class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Custom actions or logging at the end of each epoch
        pass

custom_callback = CustomCallback()

# Save the best model in Keras format
best_model = load_model(checkpoint_filepath)
best_model.save('best_LSTM_model.h5')

# TensorBoard callback (modify the log directory as needed)
tensorboard_callback = TensorBoard(log_dir='./logs')

# Add the TensorBoard callback to the model.fit() method
history = model.fit(
    # ... (other parameters)
    callbacks=[early_stopping, tensorboard_callback, custom_callback]
)

# Print instructions for TensorBoard
print("\nTo view TensorBoard, run the following command in your terminal:")
print("tensorboard --logdir=./logs")
