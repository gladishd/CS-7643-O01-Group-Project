import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the base path where the folders are located
base_path = 'Audio_Song_Actors_01-24'
# Folders containing the audio files
actor_folders = ['Actor_01', 'Actor_02']
# Paths for saving the waveform and Mel spectrogram images
waveform_path = 'waveforms'
mel_spectrogram_path = 'mel_spectrograms'

# Ensure output directories exist
os.makedirs(waveform_path, exist_ok=True)
os.makedirs(mel_spectrogram_path, exist_ok=True)

for actor_folder in actor_folders:
    actor_path = os.path.join(base_path, actor_folder)
    # List all wav files in the folder
    audio_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]

    for audio_file in audio_files:
        # Construct the full path to the audio file
        audio_file_path = os.path.join(actor_path, audio_file)
        # Load the audio file
        audio_data, sampling_rate = librosa.load(audio_file_path)

        # Generate and save the waveform image
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio_data, sr=sampling_rate)
        plt.title(f"Waveform - {audio_file}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        waveform_image_path = os.path.join(waveform_path, f"{audio_file[:-4]}_waveform.png")
        plt.savefig(waveform_image_path)
        plt.close()

        # Generate and save the Mel spectrogram image
        S = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_S, sr=sampling_rate, x_axis='time', y_axis='mel')
        plt.title(f'Mel Spectrogram - {audio_file}')
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        mel_spectrogram_image_path = os.path.join(mel_spectrogram_path, f"{audio_file[:-4]}_mel_spectrogram.png")
        plt.savefig(mel_spectrogram_image_path)
        plt.close()

print("Processing completed.")
