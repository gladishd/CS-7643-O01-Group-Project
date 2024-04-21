import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from itertools import cycle
os.makedirs('Recurrent_Neural_Network_PyTorch_Images', exist_ok=True)

# Assume binary classification and y_score is the score obtained from the model
def plot_roc_curve(y_test, y_score, n_classes):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 7))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("Recurrent_Neural_Network_PyTorch_Images/RNN_ROCPyTorch_RNN.png")
    plt.show()

# This function should be called from within your validation loop
# y_test is the true binary label and y_score is the score obtained from the model
# For multi-class classification, you might need to binarize the output
# You will need to modify your evaluate_metrics function to return scores instead of just predictions

# Call this function in your main function or where you're handling your evaluation
# plot_roc_curve(y_true, y_scores, n_classes)

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

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# Assume n_classes is the number of classes and y_score is the score obtained from the model
def plot_multiclass_roc_curve(y_test, y_score, n_classes):
    # Binarize the output labels for each class
    y_test = label_binarize(y_test, classes=[*range(n_classes)])
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(7, 7))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'
                                             ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("Recurrent_Neural_Network_PyTorch_Images/RNN_ROC_PyToRCH_Multiclass.png")
    plt.show()

# Modify the evaluate_metrics function to return output probabilities (y_scores)
def evaluate_metrics(model, loader, n_classes):
    y_true, y_pred, y_score = [], [], []
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            scores = nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(scores.cpu().numpy())

    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    return y_true, y_pred, y_score, precision, recall, f1

# Call the plot_multiclass_roc_curve function in your main function
# after you have collected the true labels and output scores


# # Function to evaluate model metrics
# def evaluate_metrics(model, loader):
#     y_true, y_pred = [], []
#     model.eval()
#     with torch.no_grad():
#         for features, labels in loader:
#             outputs = model(features)
#             _, predicted = torch.max(outputs, 1)
#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend(predicted.cpu().numpy())
#     precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
#     recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
#     f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
#     return precision, recall, f1

# Function to extract and normalize mel spectrogram features from audio
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(mel_spectrogram_db.T)
    return scaled_features

# Custom dataset class for audio files
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# RNN model definition using LSTM with an additional dense layer
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # Additional dense layer
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0, c0 = self._init_hidden(x.size(0), x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
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
    plt.savefig("Recurrent_Neural_Network_PyTorch_Images/RNN_PyTorch_Display_Model_Params.png")
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

def main():
    label_encoder = LabelEncoder()
    dataset_paths = [
        'Data Zenodo/Audio_Speech_Actors_01-24_split',
        'Data Zenodo/Audio_Song_Actors_01-24_split'
    ]
    train_loader, test_loader = process_dataset(dataset_paths, max_files, emotion_dict, label_encoder)
    model = RNNModel(input_size=n_mels, hidden_size=128, num_layers=2, num_classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    best_f1 = 0
    early_stop_count = 0
    num_epochs = 10

    losses = []  # List to store loss for each epoch
    precisions = []  # List to store precision for each epoch
    recalls = []  # List to store recall for each epoch
    f1_scores = []  # List to store F1-score for each epoch
    all_y_true = []  # List to store all true labels
    all_y_scores = []  # List to store all output scores

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
        losses.append(total_loss / len(train_loader))

        # Validation phase
        y_true, y_pred, y_scores, precision, recall, f1 = evaluate_metrics(model, test_loader, n_classes)
        all_y_true.extend(y_true)
        all_y_scores.extend(y_scores)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        scheduler.step(f1)
        if f1 > best_f1:
            best_f1 = f1
            early_stop_count = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stop_count += 1
            if early_stop_count > 5:
                print("Early stopping triggered.")
                break

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}')

    model.load_state_dict(torch.load('best_model.pth'))
    display_model_parameters(model)

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
    plt.savefig("Recurrent_Neural_Network_PyTorch_Images/RNNPyTorch_Training_Loss_and_Precision_And_Recall.png")
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
    plt.plot(range(1, num_epochs + 1), precisions, label='Precision', color='r')
    plt.title('Precision over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.grid(True)

    plt.subplot(1, 4, 3)
    plt.plot(range(1, num_epochs + 1), recalls, label='Recall', color='g')
    plt.title('Recall over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.plot(range(1, num_epochs + 1), f1_scores, label='F1-Score', color='b')
    plt.title('F1-Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("Recurrent_Neural_Network_PyTorch_Images/RNNPyTorch_Training_Metrics.png")
    plt.show()

    # Plot the training results
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
    plt.savefig("Recurrent_Neural_Network_PyTorch_Images/RNNPyTorch_Training_Loss_and_Precision_And_Recall.png")
    plt.show()

    # ROC Curve plotting after training completes
    y_true_np = np.array(all_y_true)
    y_scores_np = np.array(all_y_scores)
    plot_multiclass_roc_curve(y_true_np, y_scores_np, n_classes)

if __name__ == "__main__":
    main()
