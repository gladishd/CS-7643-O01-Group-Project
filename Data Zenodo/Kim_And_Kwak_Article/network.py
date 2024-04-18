import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import tensorflow_hub as hub

# Load pretrained models
vggish_model = hub.load("https://tfhub.dev/google/vggish/1")
yamnet_model = hub.KerasLayer(hub.load("https://tfhub.dev/google/yamnet/1"))

# Helper function to plot spectrogram
def plot_spectrogram(spec, title=None):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.savefig("kim_and_kwak_article_spectrogram.png")
    plt.show()

# Function to apply STFT and plot
def process_audio_files(file_path):
    y, sr = librosa.load(file_path, sr=None)
    f, t, Zxx = stft(y, fs=sr, nperseg=1024, noverlap=256)
    spec = np.abs(Zxx)
    plot_spectrogram(librosa.amplitude_to_db(spec), title='Spectrogram')

    # Prepare waveform for VGGish and YAMNet
    waveform = y / 32768.0  # Assuming the audio is 16-bit PCM
    waveform = waveform.astype(np.float32)  # Ensure waveform is float32

    # Ensure waveform is in the correct shape for VGGish and YAMNet
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]  # Add batch dimension

    # Processing with VGGish
    try:
        embeddings_vggish = vggish_model(waveform)
    except Exception as e:
        print(f"Error processing with VGGish: {e}")
        embeddings_vggish = None

    # Processing with YAMNet
    try:
        scores, embeddings_yamnet, spectrogram = yamnet_model(waveform)
    except Exception as e:
        print(f"Error processing with YAMNet: {e}")
        scores, embeddings_yamnet, spectrogram = None, None, None

    return embeddings_vggish, embeddings_yamnet, spectrogram

# Main execution loop
if __name__ == "__main__":
    parent_dir = "../Audio_Speech_Actors_01-24"
    actors = ["Actor_01", "Actor_02"]
    sample_files = 10
    all_files = []
    for actor in actors:
        actor_path = os.path.join(parent_dir, actor)
        all_files.extend([os.path.join(actor_path, f) for f in os.listdir(actor_path) if f.endswith('.wav')])

    selected_files = np.random.choice(all_files, sample_files, replace=False)

    for file in selected_files:
        embeddings_vggish, embeddings_yamnet, spectrogram = process_audio_files(file)
        if embeddings_vggish is not None and embeddings_yamnet is not None:
            # Further processing can be done here
            pass
file_path = "../Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"


import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Ensure TensorFlow 2.x is being used.
tf.compat.v1.enable_eager_execution()

# Load the models directly from TensorFlow Hub
vggish_model = hub.load("https://tfhub.dev/google/vggish/1")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def prepare_waveform(file_path, target_sr=16000, duration=1.0):
    """Prepare and process the waveform."""
    audio, sr = librosa.load(file_path, sr=target_sr)
    if len(audio) > sr * duration:
        audio = audio[:int(sr * duration)]
    elif len(audio) < sr * duration:
        audio = np.pad(audio, (0, int(sr * duration) - len(audio)))

    # Normalize audio
    audio = audio.astype(np.float32) / np.max(np.abs(audio))
    return audio.reshape(1, -1)

def get_features(audio):
    """Extract features using the loaded models."""
    # Process with VGGish
    vggish_embeddings = vggish_model(audio)

    # Process with YAMNet
    yamnet_results = yamnet_model(audio)
    scores, embeddings, spectrogram = yamnet_results

    return vggish_embeddings, embeddings, spectrogram

# Example usage

audio = prepare_waveform(file_path)
vggish_embeddings, yamnet_embeddings, yamnet_spectrogram = get_features(audio)

print("VGGish Features:", vggish_embeddings)
print("YAMNet Embeddings:", yamnet_embeddings)
print("Spectrogram:", yamnet_spectrogram)
