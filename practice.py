# Start out with this mp3
# Free Music Sound Effects Download - Pixabay
# https://pixabay.com/sound-effects/search/music/
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

# Path to your MP3 file
audio_path = 'cinematic-music-sketches-11-cinematic-percussion-sketch-116186.mp3'

# Load the audio file
y, sr = librosa.load(audio_path)

# Display the waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig("waveform.png")
plt.show()

# Generate a Mel-spectrogram
# A Mel-spectrogram is a spectrogram where the frequencies are converted to the Mel scale.
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)


# Convert to log scale (dB)
log_S = librosa.power_to_db(S, ref=np.max)

# Display the Mel-spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel-spectrogram')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()
plt.savefig("mel_spectrogram.png")
plt.show()

# Note: This script is the starting point. For emotion recognition, you will need to integrate this with deep learning models.
