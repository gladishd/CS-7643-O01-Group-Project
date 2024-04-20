import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

# Define CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes, n_mels, max_length):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.fc = nn.Linear(128 * (n_mels // 8) * (max_length // 8), num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Dataset and DataLoader
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), self.y[idx]

# Function to process a single dataset
def process_dataset(dataset_path, emotion_dict, n_mels, n_fft, hop_length, max_files):
    # Load data
    spectrograms = []
    labels = []
    files_processed = 0

    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav') and files_processed < max_files:
                emotion = os.path.basename(subdir)  # Get the emotion from the folder name
                if emotion in emotion_dict:
                    filepath = os.path.join(subdir, file)
                    S_DB = extract_mel_spectrogram(filepath)
                    spectrograms.append(S_DB)
                    labels.append(emotion)
                    files_processed += 1
                else:
                    print(f"Folder {subdir} does not match expected emotions and will be skipped.")
            if files_processed >= max_files:
                break

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Pad and reshape spectrograms
    max_length = max(s.shape[1] for s in spectrograms)
    X = np.array([librosa.util.fix_length(s, size=max_length, axis=1) for s in spectrograms])
    X = X[..., np.newaxis]  # Add channel dimension for CNN input
    X = np.transpose(X, (0, 3, 1, 2))  # Rearrange dimensions to [batch, channel, height, width]

    return X, encoded_labels, max_length

# Feature Extraction Function
def extract_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)  # Ensure the original sampling rate is used
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

# Training the model and tracking metrics
def train_model(num_epochs, model, train_loader, test_loader, criterion, optimizer, save_path):
    # Lists to track progress
    epoch_losses = []
    epoch_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (spectrograms, labels) in enumerate(train_loader):
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        # Validation loss
        model.eval()  # Set model to evaluate mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                spectrograms, labels = data
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(test_loader)
        val_epoch_accuracy = 100 * correct / total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.2f}%')

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epoch_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

# Parameters
n_mels = 128
n_fft = 2048
hop_length = 512
n_classes = 8
max_files = 1000  # Update this if you want to process more or fewer files

# Emotion labels mapped directly from folder names
emotion_dict = {
    'angry': 'angry', 'calm': 'calm', 'disgust': 'disgust', 'fearful': 'fearful',
    'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad', 'surprised': 'surprised'
}

# Datasets to process
dataset_paths = ['Audio_Speech_Actors_01-24_split', 'Audio_Song_Actors_01-24_split']

for dataset_path in dataset_paths:
    X, encoded_labels, max_length = process_dataset(dataset_path, emotion_dict, n_mels, n_fft, hop_length, max_files)

    # Split and create DataLoaders
    X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize the model, loss, and optimizer for each dataset
    model = CNN(n_classes, n_mels, max_length)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model on this dataset
    print(f"Training on dataset: {dataset_path}")
    train_model(10, model, train_loader, test_loader, criterion, optimizer, f"cnn_training_{dataset_path}.png")
