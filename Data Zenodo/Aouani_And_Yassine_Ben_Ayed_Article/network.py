import os
import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step up to parent directory and navigate to the specified subdirectory
parent_dir = '../Audio_Speech_Actors_01-24'
actor_dirs = ['Actor_01', 'Actor_02']

# Collect file paths
wav_files = []
for actor_dir in actor_dirs:
    actor_path = os.path.join(parent_dir, actor_dir)
    files = [os.path.join(actor_path, f) for f in os.listdir(actor_path) if f.endswith('.wav')]
    wav_files.extend(files)

# Randomly select 10 wav files for analysis
selected_files = np.random.choice(wav_files, 10, replace=False)

# Define feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    teo = librosa.feature.rms(y=y)
    hnr = librosa.piptrack(y=y, sr=sr)
    hnr = np.array([np.max(hnr[0], axis=0) if np.max(hnr[0]) > 0 else np.zeros(mfccs.shape[1])])

    # Ensure all features have the same number of columns (time frames) before concatenation
    min_length = min(mfccs.shape[1], zcr.shape[1], teo.shape[1], hnr.shape[1])
    mfccs = mfccs[:, :min_length]
    zcr = zcr[:, :min_length]
    teo = teo[:, :min_length]
    hnr = hnr[:, :min_length]

    # Concatenate all features and compute the mean across time
    all_features = np.concatenate((mfccs, zcr, teo, hnr), axis=0)
    features = np.mean(all_features, axis=1)
    return features

# Extract features from selected files
features = np.array([extract_features(f) for f in selected_files])

# Assuming we have labels for the selected files for simplicity
# Normally, you'd get these from your dataset
labels = np.random.randint(0, 2, 10)  # binary classification, e.g., happy/sad

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Initialize a pipeline with feature scaling, dimensionality reduction, and classifier
pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),  # Reduce dimensionality
    SVC(kernel='linear')  # Linear kernel SVM
)

# Train the SVM classifier
pipeline.fit(X_train, y_train)

# Predict and evaluate the model
predictions = pipeline.predict(X_test)
report = classification_report(y_test, predictions)

print(report)


import matplotlib.pyplot as plt

# Visualizing feature distributions
def plot_features(features, labels):
    plt.figure(figsize=(12, 6))
    for i, label in enumerate(np.unique(labels)):
        plt.subplot(1, len(np.unique(labels)), i + 1)
        plt.hist(features[labels == label].flatten(), bins=30, alpha=0.7, label=f'Class {label}')
        plt.title(f'Aouani And Yassine Ben Ayed: Feature distribution for Class {label}')
        plt.xlabel('Feature magnitude')
        plt.ylabel('Frequency')
        plt.legend()
    plt.tight_layout()
    plt.savefig("Aouani_And_Yassine_ben_Ayed_Feature_distribution.png")
    plt.show()
    plt.close()  # Close the figure to free up memory

# Visualizing the decision regions of the trained SVM
def plot_decision_boundaries(X, y, model):
    h = .05  # Increase step size in the mesh to reduce computation
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, linewidth=1)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Aouani And Yassine Ben Ayed: Decision Boundary of the SVM Classifier')
    plt.savefig("Aouani_And_Yassine_ben_Ayed_decision_boundary_svm_classifier.png")
    plt.show()
    plt.close()  # Close the figure to free up memory

# Get PCA transformed data for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Plotting
plot_features(features, labels)
plot_decision_boundaries(X_pca, y_train, pipeline.named_steps['svc'])

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Aouani And Yassine Ben Ayed: Confusion Matrix')
    plt.savefig("Aouani_And_Yassine_ben_Ayed_confusion_matrix.png")
    plt.show()

# Plotting the PCA components scatter plot
def plot_pca_scatter(X_pca, y):
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=f'Class {label}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Aouani And Yassine Ben Ayed: PCA Components Scatter Plot')
    plt.legend()
    plt.savefig("Aouani_And_Yassine_ben_Ayed_confusion_matrix.png")
    plt.show()

# Plotting feature distributions and decision boundaries
plot_features(features, labels)
plot_decision_boundaries(X_pca, y_train, pipeline.named_steps['svc'])
plot_confusion_matrix(y_test, predictions)
plot_pca_scatter(X_pca, y_train)

# Note: Feature importance plot is not included here as it is more relevant for tree-based models.

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score


# Cross-validation
cv_scores = cross_val_score(pipeline, features, labels, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Average cross-validation score: {np.mean(cv_scores):.2f}')

# Plotting feature correlation heatmap
def plot_feature_correlation(features):
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.corrcoef(features, rowvar=False), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Aouani And Yassine Ben Ayed: Feature Correlation Matrix')
    plt.savefig("Aouani_And_Yassine_ben_Ayed_correlation_matrix.png")
    plt.show()

# Class distribution bar chart
def plot_class_distribution(labels):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=labels)
    plt.title('Aouani And Yassine Ben Ayed: Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.savefig("Aouani_And_Yassine_ben_Ayed_class_distribution.png")
    plt.show()

# Additional plots
y_scores = pipeline.decision_function(X_test)  # Get decision function scores for ROC curve
plot_feature_correlation(features)
plot_class_distribution(labels)

# All plots
plot_features(features, labels)
plot_decision_boundaries(X_pca, y_train, pipeline.named_steps['svc'])
plot_confusion_matrix(y_test, predictions)
plot_pca_scatter(X_pca, y_train)
plot_class_distribution(labels)



from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the pipeline with probability estimation enabled in SVC
pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),  # Reduce dimensionality
    SVC(kernel='linear', probability=True)  # Enable probability estimation
)

# Define a function to safely attempt to plot the ROC curve
def safe_plot_roc_curve(y_test, X_test, model):
    try:
        y_probas = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_probas[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Aouani And Yassine Ben Ayed: Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig("Aouani_And_Yassine_ben_Ayed_receiver_operating_characteristic.png")
        plt.show()
    except AttributeError as e:
        print("ROC curve plotting failed:", e)

# Example of usage after training the model
pipeline.fit(X_train, y_train)  # Ensure the model is fitted
predictions = pipeline.predict(X_test)
report = classification_report(y_test, predictions)

print(report)
safe_plot_roc_curve(y_test, X_test, pipeline)

# All plots and additional code remain unchanged.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Feature importance for linear SVM
def plot_svm_coefficients(model, feature_names):
    if hasattr(model, "coef_"):
        coefs = model.coef_.flatten()
        top_features = np.argsort(coefs)[-10:]  # Show top 10 features

        plt.figure(figsize=(8, 6))
        plt.barh(range(len(top_features)), coefs[top_features], align='center')
        plt.yticks(range(len(top_features)), np.array(feature_names)[top_features])
        plt.xlabel("Coefficient magnitude")
        plt.ylabel("Feature")
        plt.title("Aouani And Yassine Ben Ayed: Top 10 important features in SVM")
        plt.savefig("Aouani_And_Yassine_ben_Ayed_top_10_important_features_in_SVM.png")
        plt.show()

# Generate a learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, color="r", alpha=0.1)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, color="g", alpha=0.1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Usage of the learning curve plot
plot_learning_curve(pipeline, "Aouani And Yassine Ben Ayed: Learning Curve (SVM)", features, labels, cv=5)

# If you have feature names, use this function
# feature_names = ['Feature1', 'Feature2', ...]
# plot_svm_coefficients(pipeline.named_steps['svc'], feature_names)

# Don't forget to adjust the paths and ensure that all function calls are correctly referenced
def plot_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Aouani And Yassine Ben Ayed: Waveform')

    plt.subplot(4, 1, 2)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.title('Aouani And Yassine Ben Ayed: MFCC')

    plt.subplot(4, 1, 3)
    zcr = librosa.feature.zero_crossing_rate(y)
    plt.plot(zcr[0])
    plt.title('Aouani And Yassine Ben Ayed: Zero-Crossing Rate')

    plt.subplot(4, 1, 4)
    rms = librosa.feature.rms(y=y)
    plt.plot(rms[0])
    plt.title('Aouani And Yassine Ben Ayed: RMS Energy')

    plt.tight_layout()
    plt.savefig("Aouani_And_Yassine_ben_Ayed_rms_energy.png")
    plt.show()

# Example of plotting audio features for the first selected file
plot_audio_features(selected_files[0])
from sklearn.metrics import silhouette_score

def evaluate_clustering(X, labels):
    score = silhouette_score(X, labels)
    print(f"Silhouette Score: {score:.2f}")

evaluate_clustering(X_pca, y_train)
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Aouani And Yassine Ben Ayed: Learning Curve")
    plt.legend(loc="best")
    plt.savefig("Aouani_And_Yassine_ben_Ayed_learning_curve.png")
    plt.show()

plot_learning_curve(pipeline, features, labels)
from sklearn.ensemble import RandomForestClassifier

# Define a new pipeline for RandomForest
rf_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    RandomForestClassifier(n_estimators=100)
)

# Fit and evaluate RandomForest
rf_pipeline.fit(X_train, y_train)
rf_predictions = rf_pipeline.predict(X_test)
rf_report = classification_report(y_test, rf_predictions)
print(rf_report)
