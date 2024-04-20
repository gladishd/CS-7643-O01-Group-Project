import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score

# Constants and dictionary mappings
n_mels = 128
hop_length = 512
n_fft = 2048
n_classes = 8
max_files = 100
emotion_dict = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Function to evaluate model precision and recall
def evaluate_precision_recall(model, loader):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return precision_score(y_true, y_pred, average='macro', zero_division=1), recall_score(y_true, y_pred, average='macro', zero_division=1)

# Function to extract mel spectrogram features from audio
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
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
        h0, c0 = self._init_hidden(x.size(0), x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def _init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

# Generate and display a summary table of model parameters
def display_model_parameters(model):
    params = [{"Parameter Name": name, "Size": str(param.size()), "Number of Params": param.numel()} for name, param in model.named_parameters()]
    df = pd.DataFrame(params)
    print(df)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=["#f1f1f2"]*3)
    plt.savefig("RNNPyTorch_display_model_parameters.png")
    plt.show()

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

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    losses = []
    precisions = []
    recalls = []
    accuracies = []  # List to store accuracy for each epoch

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        model.eval()  # Set the model to evaluation mode for validation
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in test_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        accuracies.append(accuracy)  # Append accuracy for the epoch

        precision, recall = evaluate_precision_recall(model, test_loader)
        precisions.append(precision)
        recalls.append(recall)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}')

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(precisions, label='Precision')
    plt.plot(recalls, label='Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig("RNNPyTorch_Training_Loss_and_Precision_And_Recall.png")
    plt.show()

    # Plotting the training results
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.plot(range(1, num_epochs + 1), losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 4, 2)
    plt.plot(range(1, num_epochs + 1), accuracies, label='Accuracy', color='b')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 4, 3)
    plt.plot(range(1, num_epochs + 1), precisions, label='Precision', color='r')
    plt.title('Precision over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.plot(range(1, num_epochs + 1), recalls, label='Recall', color='g')
    plt.title('Recall over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("RNNPyTorch_precision_over_epochs_recall_over_epochs.png")
    plt.show()

if __name__ == "__main__":
    label_encoder = LabelEncoder()
    dataset_paths = ['Audio_Speech_Actors_01-24_split', 'Audio_Song_Actors_01-24_split']
    train_loader, test_loader = process_dataset(dataset_paths, max_files, emotion_dict, label_encoder)
    model = RNNModel(input_size=n_mels, hidden_size=128, num_layers=2, num_classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, test_loader, criterion, optimizer)
    display_model_parameters(model)
