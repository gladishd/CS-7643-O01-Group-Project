import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.optim import Adam

# Parameters
n_mels = 128
hop_length = 512
n_fft = 2048
n_classes = 8  # Number of emotional categories
max_files = 100  # Adjust based on your dataset size for initial experiments

# Emotion labels from a hypothetical dataset
emotion_dict = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Function to extract mel spectrogram features from audio
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db.T

# Custom dataset class for audio files
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# RNN model definition using LSTM
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Function to process the dataset and return data loaders
def process_dataset(dataset_paths, max_files, emotion_dict, label_encoder):
    features, labels = [], []
    for dataset_path in dataset_paths:
        for subdir, dirs, files in os.walk(dataset_path):
            for file in files[:max_files]:
                if file.endswith('.wav'):
                    emotion_code = file.split('-')[2]
                    emotion = emotion_dict.get(emotion_code, None)
                    if emotion:
                        file_path = os.path.join(subdir, file)
                        features.append(extract_features(file_path))
                        labels.append(emotion)
    encoded_labels = label_encoder.fit_transform(labels)
    max_length = max(len(feature) for feature in features)
    padded_features = np.zeros((len(features), max_length, n_mels))
    for idx, feature in enumerate(features):
        padded_features[idx, :len(feature), :] = feature
    dataset = AudioDataset(padded_features, encoded_labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader

# Enhanced training function with tracking and visualization of metrics
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    model.train()
    train_losses = []
    validation_accuracies = []

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        # Validation accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for features, labels in test_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        validation_accuracy = correct / total
        validation_accuracies.append(validation_accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, Validation Accuracy: {validation_accuracy:.4f}')

    # Plotting the training loss and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("RNN_PyTorch_Training_Loss_and_Validation_Accuracy.png")
    plt.show()

# Initialize model and train
model = RNNModel(input_size=n_mels, hidden_size=128, num_layers=2, num_classes=n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
dataset_paths = ['Audio_Speech_Actors_01-24_split', 'Audio_Song_Actors_01-24_split']
label_encoder = LabelEncoder()
train_loader, test_loader = process_dataset(dataset_paths, max_files, emotion_dict, label_encoder)
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)

def print_model_summary(model):
    print("Model Summary:")
    for idx, (name, param) in enumerate(model.named_parameters()):
        print(f"{idx+1}. {name} - {param.shape}")

from sklearn.metrics import precision_score, recall_score

def evaluate_precision_recall(model, loader):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    return precision, recall

# Usage:
precision, recall = evaluate_precision_recall(model, test_loader)

from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model, loader):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("RNN_PyTorch_Confusion_Matrix.png")
    plt.show()

# Usage:
plot_confusion_matrix(model, test_loader)
