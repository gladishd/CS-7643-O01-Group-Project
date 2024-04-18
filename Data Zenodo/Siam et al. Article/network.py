import os
import random
import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Path to the parent folder
parent_folder = "../Audio_Speech_Actors_01-24"

# Actor folders
actors_folders = ["Actor_01", "Actor_02"]

# Collect all WAV files
wav_files = []
for actor in actors_folders:
    actor_path = os.path.join(parent_folder, actor)
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            wav_files.append(os.path.join(actor_path, file))

# Randomly select 10 files for analysis
selected_files = random.sample(wav_files, 10)

# Feature extraction: Load audio files and extract MFCC features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.mean(axis=1)

# Extract features from each selected file
features = np.array([extract_features(f) for f in selected_files])

# Feature decomposition using PCA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Mockup labels for classification (assuming a scenario, modify as needed)
labels = np.array([i % 3 for i in range(10)])  # Assuming 3 emotion classes

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_pca, labels, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "MLP": MLPClassifier()
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(f"Classifier: {name}")
    print(classification_report(y_test, predictions))


import matplotlib.pyplot as plt

# Visualizing the PCA-transformed features
colors = ['r', 'g', 'b']
for i in range(len(labels)):
    plt.scatter(features_pca[i, 0], features_pca[i, 1], c=colors[labels[i]], label=f'Class {labels[i]}' if i == 0 or i == len(labels) // 3 or i == 2 * len(labels) // 3 else "")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Siam et al. Article: PCA of Features')
plt.legend()
plt.savefig("siam_et_al_article_pca_of_features.png")
plt.show()

# Visualizing classifier results
def plot_decision_boundary(X, y, model, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='g')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title("Siam et al. Article: " + title)
    plt.savefig("siam_et_al_article_decision_boundary_" + title + ".png")
    plt.show()

# Plot decision boundary for each classifier
for name, clf in classifiers.items():
    plot_decision_boundary(X_train, y_train, clf, f'Decision Boundary of {name}')


import seaborn as sns

# Heatmap of feature correlations
sns.heatmap(np.corrcoef(features.T), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Siam et al. Article: Feature Correlation Heatmap')
plt.savefig("siam_et_al_article_feature_correlation_heatmap.png")
plt.show()

import pandas as pd

from sklearn.metrics import classification_report
import pandas as pd

# Specify zero_division parameter to handle undefined metrics due to no predicted samples
report_options = {'zero_division': 0}  # 0, 1, or 'warn'

# Example usage with classification_report
y_true = [0, 1, 2, 2]  # True labels
y_pred = [1, 1, 2, 1]  # Predicted labels by classifier

# Generating the report
report = classification_report(y_true, y_pred, **report_options)
print(report)

# Handling the ValueError in seaborn's pairplot
try:
    # Attempt to create a pairplot
    df_features = pd.DataFrame({'PC1': [1, 2, 3], 'Siam et al. Article: PC2': [4, 5, 6]})
    labels_series = pd.Series([0, 1, 2])
    sns.pairplot(df_features, hue=labels_series.map(lambda x: f'Class {x}'))
except ValueError as e:
    print(f"Caught an error: {e}")

# Correct usage of pairplot with hue
df_features['Label'] = labels_series.map(lambda x: f'Siam et al. Article: Class {x}')
sns.pairplot(df_features, hue='Label')
plt.savefig("siam_et_al_article_pairplot.png")
plt.show()

plt.show()

# Bar plot for feature importance (for models that support it, e.g., Random Forest)
if hasattr(classifiers['Random Forest'], 'feature_importances_'):
    sns.barplot(x=['PC1', 'PC2'], y=classifiers['Random Forest'].feature_importances_)
    plt.title('Siam et al. Article: Feature Importance')
    plt.savefig("siam_et_al_article_feature_importance.png")
    plt.show()

# Confusion matrix for classifiers
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"siam_et_al_article_confusion_matrix_{title}.png")
    plt.show()

# Calculate confusion matrix for each classifier and plot
for name, clf in classifiers.items():
    cm = confusion_matrix(y_test, clf.predict(X_test))
    plot_confusion_matrix(cm, classes=np.unique(labels), title=f'Siam et al. Article: {name}')

from sklearn.manifold import TSNE

from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Update zero_division handling in classification_report to avoid warnings and errors
def evaluate_classifier(clf, X_test, y_test, name):
    predictions = clf.predict(X_test)
    report = classification_report(y_test, predictions, zero_division=0)
    print(f"Classifier: {name}\n{report}")

# Example of updated usage
for name, clf in classifiers.items():
    evaluate_classifier(clf, X_test, y_test, name)

# Correcting the t-SNE fitting error
def perform_tsne(features_scaled, labels):
    if len(features_scaled) > 1:  # Ensure there are enough samples
        tsne = TSNE(n_components=2, perplexity=min(30, len(features_scaled)-1), random_state=42)
        features_tsne = tsne.fit_transform(features_scaled)
        colors = ['r', 'g', 'b']  # Update or expand color list as necessary
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(labels):
            plt.scatter(features_tsne[i, 0], features_tsne[i, 1], color=colors[label], label=f'Class {label}' if i == 0 else "")
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        plt.title('Siam et al. Article: t-SNE Visualization of Features')
        plt.legend()
        plt.savefig("siam_et_al_article_tsne_visualization_of_features.png")
        plt.show()

# Example of updated usage
perform_tsne(features_scaled, labels)

# Addressing the ambiguous truth value in seaborn's pairplot
def create_pairplot(features_pca, labels):
    df = pd.DataFrame(features_pca, columns=['Siam et al. Article: PC1', 'PC2'])
    df['Label'] = pd.Series(labels).map(lambda x: f'Class {x}')
    sns.pairplot(df, hue='Label')
    plt.savefig("siam_et_al_article_pairplot.png")
    plt.show()

# Example of updated usage
create_pairplot(features_pca, labels)




# Learning curve for classifier performance over different training set sizes
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title("Siam et al. Article: " + title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(f"siam_et_al_article_learning_curve_{title}.png")
    plt.show()

for name, clf in classifiers.items():
    plot_learning_curve(clf, f'Siam et al. Article: Learning Curve of {name}', X_train, y_train, cv=5)

