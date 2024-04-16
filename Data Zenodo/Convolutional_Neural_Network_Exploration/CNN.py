import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LSTM, Reshape
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, SVG

# Other required imports already included in previous cells
# Add seaborn to existing imports to avoid NameError in sns usage

# Continue with the rest of the script as already provided


# Define the path to your dataset and load data
dataset_path = '../Audio_Speech_Actors_01-24'
n_mels = 128
n_fft = 2048
hop_length = 512
n_classes = 8
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

def extract_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

spectrograms = []
labels = []
files_processed = 0
max_files = 10  # Limit for testing

for subdir, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav') and files_processed < max_files:
            parts = file.split('-')
            if len(parts) > 2:
                emotion_code = parts[2]
                emotion = emotion_dict.get(emotion_code, 'unknown')
                if emotion != 'unknown':
                    filepath = os.path.join(subdir, file)
                    S_DB = extract_mel_spectrogram(filepath)
                    spectrograms.append(S_DB)
                    labels.append(emotion)
                    files_processed += 1

# Encode labels and prepare training data
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
max_length = max(s.shape[1] for s in spectrograms)
X = np.array([librosa.util.fix_length(s, size=max_length, axis=1) for s in spectrograms])[..., np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)
input_shape = (n_mels, max_length, 1)

# Model building
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Flatten(),
    Dense(1024, activation='relu'),
    Reshape((-1, 64)),
    LSTM(64),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model and weights
model.save('model_path')
model.save_weights('model_weights_path.h5')

# Model evaluation
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Generate a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.savefig("CNN_confusion_matrix.png")
plt.show()

import tensorflow as tf
import matplotlib.pyplot as plt

def plot_class_activation_maps(model, img_array, class_idx):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)  # Ensure the input is a tensor
    img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimension

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        last_conv_layer = model.get_layer('conv2d_1')
        model_outputs = model(img_tensor)
        class_channel = model_outputs[:, class_idx]
        grads = tape.gradient(class_channel, last_conv_layer.output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer.output), axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    plt.matshow(heatmap[0])
    plt.title('CNN Heatmap of Class Activation')
    plt.savefig("CNN_class_activation_heatmap.png")
    plt.show()




import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.optimizers import legacy as legacy_optimizers
import numpy as np
from tensorflow.keras.layers import Reshape
import librosa
import os
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow_model_optimization as tfmot
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LSTM, Reshape
from tensorflow.keras.models import Sequential
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
dataset_path = '../Audio_Speech_Actors_01-24'

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
            parts = file.split('-')
            if len(parts) > 2:
                try:
                    emotion_code = parts[2]
                    emotion = emotion_dict.get(emotion_code, 'unknown')
                    if emotion != 'unknown':
                        filepath = os.path.join(subdir, file)
                        S_DB = extract_mel_spectrogram(filepath)
                        spectrograms.append(S_DB)
                        labels.append(emotion)
                        files_processed += 1
                except Exception as e:
                    print(f"Could not process file {file}: {e}")
            else:
                print(f"Filename {file} does not match expected format.")
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


input_shape = (n_mels, max_length, 1)  # Adjust the number of channels to 1
from tensorflow import keras

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Flatten(),  # Flatten the entire feature map from the Conv layers
    Dense(1024, activation='relu'),  # Connect the flat output to a Dense layer if needed
    Reshape((-1, 64)),  # Now we reshape what's left after Dense into a suitable form for LSTM
    LSTM(64),
    Dropout(0.5),
    Dense(8, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Ensure callbacks and other training parameters are set correctly
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
    # Other callbacks like early stopping can be added here
]

# Training the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=callbacks)

# After training, save and potentially export the model as needed
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
plt.title('CNN Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('cnnmodel_accuracy.png')  # Saving the figure
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('CNN Model loss')
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
plt.title('CNN Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('cnnconfusion_matrix.png')  # Saving the figure
plt.show()

def plot_predictions(images, predictions, true_labels, class_names):
    num_images = len(images)
    num_columns = 5
    num_rows = np.ceil(num_images / num_columns).astype(int)  # Ensure there's enough room for all images
    plt.figure(figsize=(15, num_rows * 3))  # Adjust the figure size based on the number of rows
    for i in range(num_images):
        plt.subplot(num_rows, num_columns, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        col = 'green' if predictions[i] == true_labels[i] else 'red'
        plt.xlabel(f'Pred: {class_names[predictions[i]]}\nTrue: {class_names[true_labels[i]]}', color=col)
    plt.tight_layout()
    plt.savefig('cnnpredictions_visualization.png')  # Saving the figure
    plt.show()

from sklearn.metrics import classification_report
# Assuming that the '0' label is not expected and should be mapped to 'unknown'
emotion_dict['00'] = 'unknown'

# Get unique class labels from both true_classes and predicted_classes
unique_labels = np.unique(np.concatenate((true_classes, predicted_classes)))

# Check if all unique labels have corresponding entries in emotion_dict
assert all(str(label).zfill(2) in emotion_dict for label in unique_labels), "Not all labels have entries in emotion_dict"

# Decode labels to original class names using the emotion_dict
filtered_class_names = [emotion_dict[str(label).zfill(2)] for label in unique_labels]

# Ensure the labels parameter is set to the unique labels
print(classification_report(true_classes, predicted_classes, labels=unique_labels, target_names=filtered_class_names))

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Extract other features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    # Stack all features into one array
    features = np.stack((S_DB, chroma, spectral_contrast, tonnetz), axis=-1)
    return features

# Assuming you've decoded your labels back to their original values
class_names = list(emotion_dict.values())
plot_predictions(X_test, predicted_classes, true_classes, class_names)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.utils import model_to_dot
from IPython.display import SVG
from keras import backend as K
import seaborn as sns

# Assuming y_test is already loaded and categorical

# For ROC Curve and AUC
def plot_roc_curve(y_true, y_pred, n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true = label_binarize(y_true, classes=[i for i in range(n_classes)])
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Plot all ROC curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CNN Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("CNN_Receiver_Operating_Characteristic.png")
    plt.show()

# For Precision-Recall Curve
def plot_precision_recall_curve(y_true, y_pred, n_classes):
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        average_precision[i] = auc(recall[i], precision[i])
    # Plot all precision-recall curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label=f'Precision-recall curve of class {i} (area = {average_precision[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('CNN Precision-Recall curve')
    plt.legend(loc="upper right")
    plt.savefig("CNN_Precision_Recall_Curve.png")
    plt.show()

# Adjusting t-SNE with appropriate perplexity
def plot_tsne(features, labels, n_components=2, perplexity=30):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(features.reshape((features.shape[0], -1)))
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.title('CNN t-SNE visualization of Features')
    plt.savefig("CNN_t_SNE_Visualization_of_Features.png")
    plt.show()


def plot_heatmap_of_class_activation(model, img, class_idx):
    # Wrap the output extraction in a GradientTape to record operations
    with tf.GradientTape() as tape:
        # Set the input image as a variable and watch it in the tape
        tape.watch(img)
        # Forward pass
        preds = model(img[np.newaxis, ...])
        class_channel = preds[:, class_idx]

    # Use the tape to compute the gradients with respect to the output neuron
    grads = tape.gradient(class_channel, model.get_layer('conv2d_1').output)[0]

    # Pooling and normalization steps
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    last_conv_layer_output = model.get_layer('conv2d_1').output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # Display the heatmap
    plt.matshow(heatmap)
    plt.title('CNN Heatmap of Class Activation')
    plt.savefig("CNN_Heatmap_Of_Class_Activation.png")
    plt.show()

# For Model Architecture Visualization
def plot_model_architecture(model):
    svg_img = model_to_dot(model, show_shapes=True).create(prog='dot', format='svg')
    display(SVG(svg_img))

# For Histogram of Weights and Biases
def plot_histogram_weights_and_biases(model):
    weights = [layer.get_weights()[0] for layer in model.layers if len(layer.get_weights()) > 0]
    biases = [layer.get_weights()[1] for layer in model.layers if len(layer.get_weights()) > 1]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(np.concatenate([np.ravel(w) for w in weights]), bins=50)
    plt.title('CNN Histogram of Weights')
    plt.subplot(1, 2, 2)
    plt.hist(np.concatenate([np.ravel(b) for b in biases]), bins=50)
    plt.title('CNN Histogram of Biases')
    plt.savefig("CNN_Histogram_of_Biases.png")
    plt.show()
import tensorflow as tf

class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

# Use this callback when fitting the model
lr_logger = LearningRateLogger()
callbacks.append(lr_logger)
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=callbacks)


def plot_learning_rate_scheduler(history):
    lr = history.history.get('lr')
    if lr is None:
        print("Learning rate not recorded during training.")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(lr)
    plt.title('CNN Learning Rate over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.savefig("CNN_Learning_rate_Over_Epochs.png")
    plt.show()


# For Gradients Flow Visualization
def plot_gradients_flow(model, inputs, outputs):
    with tf.GradientTape() as tape:
        preds = model(inputs)
        loss = tf.keras.losses.sparse_categorical_crossentropy(outputs, preds)
    grads = tape.gradient(loss, model.trainable_weights)
    plt.figure(figsize=(10, 5))
    plt.hist([np.max(g) for g in grads if g is not None], bins=20)
    plt.title('CNN Gradient Flow')
    plt.xlabel('Maximum Gradient per Layer')
    plt.ylabel('Frequency')
    plt.savefig("CNN_Gradient_Flow.png")
    plt.show()

# For Attention Maps Visualization
def plot_attention_maps(model, attention_layer_name, input_data):
    layer = model.get_layer(attention_layer_name)
    attention_output = layer.output
    attention_model = tf.keras.Model(inputs=model.input, outputs=attention_output)
    attention_result = attention_model.predict(input_data)
    plt.matshow(attention_result[0], cmap='viridis')
    plt.colorbar()
    plt.title('CNN Attention Map')
    plt.savefig("CNN_Attention_Map.png")
    plt.show()



# First, make sure predictions are made and you have probabilities for each class
y_prob = model.predict(X_test)

# Since y_test is likely already encoded, ensure it's in the right format to plot ROC
y_test_bin = label_binarize(y_test, classes=[i for i in range(n_classes)])

plot_roc_curve(y_test_bin, y_prob, n_classes)
plot_precision_recall_curve(y_test_bin, y_prob, n_classes)
# First, ensure you have some features to visualize. You can use any layer output as features.
# Here's how you might extract features from a specific layer:
# Assuming you want to visualize features from the first convolutional layer
# Assuming you want to visualize features from the first convolutional layer
layer_name = 'conv2d_2'  # Update to the correct name as per the error message
intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(X_test)


plot_tsne(intermediate_output, y_test, n_components=2, perplexity=min(30, len(y_test)-1))


# You need to modify 'plot_heatmap_of_class_activation' to specify your model's last convolutional layer name.
# Assuming you have an image (e.g., a Mel-spectrogram or MFCC) to input and a class index to visualize:
# Make sure the 'last_conv_layer_name' is correct in the plot_heatmap_of_class_activation function


plot_model_architecture(model)
plot_histogram_weights_and_biases(model)
# Ensure that the 'lr' is being recorded in history if using a learning rate scheduler during training.
plot_learning_rate_scheduler(history)

# You will need some example inputs and their corresponding outputs to visualize gradient flow:
example_inputs = X_train[:10]  # Using the first 10 training examples
example_outputs = y_train[:10]  # Corresponding outputs

plot_gradients_flow(model, example_inputs, example_outputs)
# Make sure your model has an attention layer and you know the layer's name:
attention_layer_name = 'your_attention_layer_name'

plot_attention_maps(model, attention_layer_name, X_test[:1])  # Using the first test example
