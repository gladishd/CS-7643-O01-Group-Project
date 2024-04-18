import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('extracted_features.csv')

# Histogram of F0 semitone mean
plt.figure(figsize=(10, 6))
plt.hist(df['F0semitoneFrom27.5Hz_sma3nz_amean'], bins=30, color='blue', alpha=0.7)
plt.title('The Zenodo Data: Histogram of F0 Semitone Mean')
plt.xlabel('F0 Semitone Mean from 27.5 Hz')
plt.ylabel('Frequency')
plt.savefig("The Zenodo Data_Histogram_of_F0_Semitone_Mean.png")
plt.show()

# Box plot of loudness mean across different files
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='filename', y='loudness_sma3_amean')
plt.xticks(rotation=90)
plt.title('The Zenodo Data: Loudness Mean Across Files')
plt.xlabel('File')
plt.ylabel('Loudness Mean')
plt.savefig("The Zenodo Data_Loudness_Mean_Across_Files.png")
plt.show()

# Scatter plot to compare F0 mean and Loudness mean
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='F0semitoneFrom27.5Hz_sma3nz_amean', y='loudness_sma3_amean')
plt.title('The Zenodo Data: Scatter Plot of F0 Mean vs Loudness Mean')
plt.xlabel('F0 Semitone Mean from 27.5 Hz')
plt.ylabel('Loudness Mean')
plt.savefig("The Zenodo Data_Scatter_Plot_of_F0_Mean_vs_Loudness_Mean.png")
plt.show()

import numpy as np

# Correlation matrix heatmap
corr = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('The Zenodo Data: Correlation Matrix of Features')
plt.savefig("The Zenodo Data_Correlation_Matrix_of_Features.png")
plt.show()

# Example of a time series plot for the first file (modify as needed for actual time series data)
# This requires having time-indexed data; here we use the 'start' and 'end' columns as a proxy for plotting.
plt.figure(figsize=(12, 8))
time_series_data = df.loc[df['filename'] == df['filename'].unique()[0], ['start', 'loudness_sma3_amean']].sort_values('start')
plt.plot(time_series_data['start'], time_series_data['loudness_sma3_amean'], marker='o')
plt.title('The Zenodo Data: Time Series of Loudness Mean for First Audio File')
plt.xlabel('Time')
plt.ylabel('Loudness Mean')
plt.savefig("The Zenodo Data_Time_Series_of_Loudness_Mean.png")
plt.show()






import os
import pandas as pd
import opensmile

def extract_audio_features(audio_file_path):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = smile.process_file(audio_file_path)
    return features

def process_audio_files(directory_path):
    all_features = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            features = extract_audio_features(file_path)
            features['filename'] = filename  # Add filename to DataFrame
            all_features.append(features)

    # Combine all DataFrames into one
    result_df = pd.concat(all_features)
    return result_df

# Example usage
audio_directory = "../Audio_Speech_Actors_01-24/Actor_01"
features_df = process_audio_files(audio_directory)
print(features_df)

# Save to CSV
features_df.to_csv('extracted_features.csv')

import pandas as pd
import matplotlib.pyplot as plt

# Load your features CSV
df = pd.read_csv('extracted_features.csv')

# Display basic statistics
print(df.describe())

# Plot the distribution of a feature
plt.hist(df['F0semitoneFrom27.5Hz_sma3nz_amean'], bins=30, alpha=0.5, label='F0 semitone mean')
plt.xlabel('Frequency')
plt.ylabel('Count')
plt.title('OpenSMILE: Distribution of F0 Semitone Mean')
plt.legend()
plt.savefig("OpenSMILE_Distribution_of_F0_Semitone_Mean.png")
#plt.show()

# Correlation matrix heatmap
import seaborn as sns
import numpy as np
# Exclude non-numeric columns before calculating the correlation.
# In this case, exclude 'file' and 'filename' which are string columns.
numeric_df = df.select_dtypes(include=[np.number])

# Now, compute the correlation matrix
correlation_matrix = numeric_df.corr()

# Continue with your heatmap plotting
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".1f")
plt.title('OpenSMILE: Feature Correlation Matrix')
plt.savefig("OpenSMILE_Feature_Correlation_Matrix.png")  # Saving to the mounted directory
#plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your features CSV
df = pd.read_csv('extracted_features.csv')

# Exclude non-numeric columns before calculating the correlation.
numeric_df = df.select_dtypes(include=[np.number])

# Compute the correlation matrix
correlation_matrix = numeric_df.corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('OpenSMILE: Feature Correlation Matrix')
plt.savefig("OpenSMILE_Feature_Correlation_Matrix_Large1.png")  # Save the larger image
#plt.show()




plt.figure(figsize=(20, 16))  # Adjust the size to your needs
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
plt.title('OpenSMILE: Feature Correlation Matrix')
plt.savefig("OpenSMILE_Feature_Correlation_Matrix_Large2.png")
#plt.show()


# Consider correlations above 0.7 as high
high_corr = correlation_matrix[abs(correlation_matrix) > 0.7]

plt.figure(figsize=(20, 16))
sns.heatmap(high_corr, cmap='coolwarm', center=0)
plt.title('OpenSMILE: High Feature Correlation Matrix')
plt.savefig("OpenSMILE_High_Feature_Correlation_Matrix.png")
#plt.show()


# Plot histogram
plt.hist(df['F0semitoneFrom27.5Hz_sma3nz_amean'], bins=30, alpha=0.5)
plt.xlabel('Frequency')
plt.ylabel('Count')
plt.title('OpenSMILE: Distribution of F0 Semitone Mean')
plt.savefig("F0_Semitone_Mean_Distribution.png")
#plt.show()

# Correlation Matrix
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('OpenSMILE: Feature Correlation Matrix')
plt.savefig("Feature_Correlation_Matrix.png")
#plt.show()


try:
    df = pd.read_csv('extracted_features.csv')
    # Further processing
except FileNotFoundError:
    print("CSV file not found. Please check the path and filename.")




















import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
try:
    df = pd.read_csv('extracted_features.csv')
    print("DataFrame loaded successfully. Available columns:")
    print(df.columns)
except FileNotFoundError:
    print("CSV file not found. Please check the path and filename.")

# Optional: Display basic statistics and data types to understand the dataset better
print(df.describe())
print(df.dtypes)

# Plot distribution of a selected feature if exploratory analysis is your goal
plt.hist(df['F0semitoneFrom27.5Hz_sma3nz_amean'], bins=30)
plt.xlabel('F0 Semitone Mean')
plt.ylabel('Frequency')
plt.title('OpenSMILE: Distribution of F0 Semitone Mean')
plt.savefig("OpenSMILE_Distribution_of_F0_Semitone_Mean1.png")
plt.show()

# If there's a label column, adjust the script to use it for modeling
# For example:
label_column = 'your_actual_label_column_name'  # Replace with your actual label column if you're planning to model
if label_column in df.columns:
    X = df.drop(columns=[label_column, 'filename'])  # Assuming 'filename' is not a feature
    y = df[label_column]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Model accuracy on test set:", model.score(X_test, y_test))

    # Feature importance
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    plt.barh(range(len(importances)), importances[sorted_indices])
    plt.yticks(range(len(importances)), df.columns[sorted_indices])
    plt.xlabel('Importance')
    plt.title('OpenSMILE: Feature Importances')
    plt.savefig("OpenSMILE_Feature_Importances1.png")
    plt.show()

else:
    print("Label column not found. Ensure you have the correct column name if you are conducting supervised learning.")

# If just performing exploratory data analysis, continue with plotting correlations or other statistical summaries.
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix)
plt.title('OpenSMILE: Feature Correlation Matrix')
plt.savefig("OpenSMILE_Feature_Correlation_Matrix1.png")
plt.show()





import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def extract_audio_features(audio_file_path):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    try:
        features = smile.process_file(audio_file_path)
        return features
    except Exception as e:
        print(f"Error processing file {audio_file_path}: {e}")
        return None

def process_audio_files(directory_path):
    all_features = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            features = extract_audio_features(file_path)
            if features is not None:
                features['filename'] = filename  # Add filename to DataFrame
                all_features.append(features)
    if all_features:
        result_df = pd.concat(all_features)
        return result_df
    else:
        return pd.DataFrame()

# Example usage
audio_directory = "../Audio_Speech_Actors_01-24/Actor_01"
features_df = process_audio_files(audio_directory)
if not features_df.empty:
    features_df.to_csv('extracted_features.csv')
    print("Features extracted and saved to CSV.")
else:
    print("No features extracted.")

# Load data
try:
    df = pd.read_csv('extracted_features.csv')
    print("DataFrame loaded successfully.")
except FileNotFoundError:
    print("CSV file not found. Please check the path and filename.")

# Display basic statistics
print(df.describe())

# Correlation matrix heatmap
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('OpenSMILE: Feature Correlation Matrix')
plt.show()

# Plot histograms for multiple features
features_to_plot = ['F0semitoneFrom27.5Hz_sma3nz_amean', 'loudness_sma3_amean', 'spectralFlux_sma3_amean']
for feature in features_to_plot:
    plt.hist(df[feature], bins=30, alpha=0.5, label=f'Distribution of {feature}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'OpenSMILE: {feature}')
    plt.legend()
    plt.show()

# Optional: Modeling (if you have a label column)
label_column = 'YourLabelColumn'  # Adjust the label column name as necessary
if label_column in df.columns:
    X = df.drop(columns=[label_column, 'filename'])
    y = df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Model accuracy on test set: {model.score(X_test, y_test)}")
