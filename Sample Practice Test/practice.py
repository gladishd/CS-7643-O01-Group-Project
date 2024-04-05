# Start out with this mp3
# Free Music Sound Effects Download - Pixabay
# https://pixabay.com/sound-effects/search/music/
""" Now, in order to run this code you might need to install several libraries
not withstanding things like Conda and pip..if you `pip install tensorflow`
just remember that you might get some spikes in data rate but it might take.
and it might take, a few minutes.  So the first thing that you've got to do
is generate some kind of model..and actually, we don't need to generate a model
just yet. We've already got several things that we have done related to audio
processing, and how we can utilize these pre-existing libraries already..sort of
like the machine code of Machine Learning, in Deep Learning..and we can do that
in order to better "understand" what each part does. """
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, classification_report
""" The most important thing is TensorFlow. If you have some kind of trouble
running into this..then when you're doing TensorFlow you I think can install
it directly from their website. Just remember that.
The second thing that we are going to have to define is the path to the MP3
file, which is representative of this file that I have downloaded..it might be
"prudent" to make this thing judgeable but to do it in such a way that all of
the libraries that we have "imported" thus far like librosa or matplotlib,
NumPy or SKLearn, TensorFlow or SKLearn.Metrics..all of these models are
designed so that we can have a path that gives us..an MP3 file which we then
analyze, and then we can load in this file given the path to that file, and
we can do that via librosa.load. But that means that we'd have to use the audio
time series and that's where the...deconstructed parameter y comes in, with its
concurrent and constituent sampling rate that we know as sr.  """
audio_path = "cinematic-music-sketches-11-cinematic-percussion-sketch-116186.mp3"
""" And then we load in the audio file. And we do it using librosa.load so that
we can obtain the audio time series that we know and love as y..and the sampling
rate that we know as sr. """
(y, sr) = librosa.load(audio_path)
""" Then we dispplay the actual waveform and that, I think, is going to give us
the amplitude over time, the much-coveted Mel-spectrogram in which we visually..
represent, "all of the" spectrum of frequencies which occur in a sound over time,
and that is converted to the Mel scale. And I do, I still want to be more suitable
in this project for human auditory perception. I still want to convert the Mel-spectrogram
to a logarithmic scale..that we know and it's not like we're on the Bell Curve
or something like that, we're just on the dB scale and that allows us to better
match human hearing and visualize it. That allows us to display the waveform.  """
plt.figure(figsize = (10, 4))
librosa.display.waveshow(y, sr = sr)
plt.title("Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("waveform.png")
plt.show()
""" And then we generate all of our Mel-Spectrogram and when we do that, we
extract...we have this spectrogram that is and we extract the coefficients,
the cepstral coefficients which are the coefficients that collectively make up
an MFC. And these are the coefficients thata re derived, from the type of cepstral..
representation..that we get from the audio clip itself..the nonlinear "spectrum-of-a-spectrum"). And then we are able to calculate the Chroma feature,
which represents the pitch classes that come in twelve different "forms". And then,
that is how we reconstruct the notion of spectral contrast. The Mel-spectrogram
itself, that is, is a Spectrogram in which the frequencies that we have are
"consistently" converted into the Mel scale.  """
S = librosa.feature.melspectrogram(y = y, sr = sr, n_mels = 128)
""" So what happens when we convert them into the Mel scale? What we get is this
log scale..dB based implementation.  """
log_S = librosa.power_to_db(S, ref = np.max)
""" And then we want to actually pot the Spectogram. It's supposed to be a simulation
of human hearing, for "what it's worth". """
plt.figure(figsize = (10, 4))
librosa.display.specshow(log_S, sr = sr, x_axis = "time", y_axis = "mel")
plt.title('Mel-spectrogram')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()
plt.savefig("mel_spectrogram.png")
plt.show()
""" And it's like those Deep Learning videos that we get...
CS 7643 - Deep Learning - MediaSpace @ Georgia Tech
https://mediaspace.gatech.edu/channel/CS+7643+-+Deep+Learning/267756942
And we get them, we get this thing where we start out with this script for
emotion recognition, and then we try to later on integrate this with deep learning
models. So our Feature Extraction is going to be a little bit more complex. And
we take these MFCCs, we take these Mel-frequency cepstral coefficients and we
use them..we have a nonlinear "spectrum-of-a-spectrum" which gives us the "inceptive"
notion that allows us to represent this cepstral representation of the audio clip.
And the Chroma feature..represents the twelve different pitch classes. And, the
Spectral Contrast is one of a kind in that we measure the level difference that exists..
between the valleys and the peaks in the sound spectrum. """
mfccs = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 13)
""" And that is what we're doing with these MFCCs. Now, the next step is to take
this chroma feature and it is the Chroma feature that we are going to define, with
its twelve different pitch classes. """
chroma = librosa.feature.chroma_stft(y = y, sr = sr)
""" And this is our idea, I know there is a lot of dataset..the equation is, the
question is what kind of data..how do I make sure that we have the right audio dataset? And I think that there is no shortage of data, it's more like we have to compute
the answer to what is the spectral contrast?! """
spectral_contrast = librosa.feature.spectral_contrast(y = y, sr = sr)
""" And that's what it is..so quite "embarrassingly" we're going to import some
HMM, we're going to import it and that's going to be our Hidden Markov Model..but
we need to import it because what we're going to do is base our Gaussian "Mixture Model"
off of it and we're going to do that because we need to fit all that stuff that we
want. I'm pretty sure that based off this "library" we can do the Naive Bayes classifier,
we can do Support Vector Machines and potentially..we have "Every potential" to
classify these emotions based off of our audio features. """
from hmmlearn import hmm
""" Furthermore, we can define the answer to the question of, how many states
do we have for a Hidden Markov Model? What is the number of categories of emotion
for the Gaussian Mixture Model..and how do we use that for classification? How
can we define these more primitive models and then use them as a "benchmark" for
understanding how we classify emotions based on the features of our audio, like
Support Vector Machines!? """
number_of_states = 5
""" So we have 5 states, but we have 5 different states in our HMM.
Or alternatively and additionally we can add 7 different emotions to classify. """
number_of_emotions = 7
""" And the labels, we could have 0 be neutral, 1 be happy..and then so on and so
forth where we get that..and in a "real" dataset we can find these labels that do,
come from our data annotations, actually. And we also create an array for labels,
that have an..a random label for each sample. """
labels = np.random.randint(0, number_of_emotions, size = mfccs.shape[1])
""" Last but not least..."least but not last" we are going to define the
Convolutional Neural Network..that's our network that uses the TensorFlow..
the Keras API, and you're going to see it because what we're going to do is
specify the architecture.  """
model = hmm.GaussianHMM(n_components = number_of_states)
model.fit(mfccs.T)
from sklearn.mixture import GaussianMixture
""" Training our Gaussian Mixture Model """
gmm = GaussianMixture(n_components=number_of_emotions)
gmm.fit(mfccs.T)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
""" Training our "Naïve" Bayes Classifier """
nb_model = GaussianNB()
nb_model.fit(mfccs.T, labels)
""" Training our Support Vector Machine """
svm_model = SVC()
svm_model.fit(mfccs.T, labels)
""" Now, we're about to be ready to prepare our architecture. """
import tensorflow as tf
from tensorflow.keras import layers, models
time_steps = S.shape[1]
""" S is the Mel-Spectrogram! Now let's define a simple CNN model. """
model = models.Sequential([
layers.Conv2D(32,
              (3, 3),
              activation = "relu",
              input_shape = (128, time_steps, 1)),
layers.MaxPooling2D((2, 2)),
layers.Flatten(),
layers.Dense(64, activation = "relu"),
layers.Dense(number_of_emotions, activation = "softmax")
])
""" And compile and train the, comply and train the model... """
model.compile(optimizer = "adam",
loss = "sparse_categorical_crossentropy",
metrics = ["accuracy"])
""" And now for "another" example of splitting the data into testing and..
training sets """
from sklearn.model_selection import train_test_split
""" load in the audio file! Although our team's background "primarily lies in
civil engineering, we have so much experience actually and it's all in Python,
Java, and yes, Machine Learning..."""
audio_path = "cinematic-music-sketches-11-cinematic-percussion-sketch-116186.mp3"
(y, sr) = librosa.load(audio_path)
""" As "before" extract the MFCCs, and then transpose them such that the shape
is (time_steps, n_mfcc). """
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfccs = np.transpose(mfccs)
""" We also want to..."also" assume 7 emotions, and last but not least generate
some dummy labels, for the purposes..of demonstration.  """
labels = np.random.randint(0, 7, size = len(mfccs))
""" And when we do, we Reshape the MFCCs to add this dimension for the channel...
that means samples, time_steps, features, 1.  """
mfccs_reshaped = np.expand_dims(mfccs, axis = -1)
""" Then we split the dataset into its training and testing sets. """
(X_train,
 X_test,
 y_train,
 y_test) = train_test_split(
    mfccs_reshaped,
    labels,
    test_size = 0.2,
    random_state = 42
    )
""" And define the architecture. The thing about our architecture is that
we want the "Convolutional" to be the main thing we "think of"..and the
pooling, flattening, and dense layers allow us to do the classification
that we so "desperately crave". But we need to compile the CNN model and
we do so by setting the metrics for training, the loss function, and the
optimizer.  """
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    layers.Conv1D(32, 3, activation = "relu"),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation = "relu"),
    layers.Dense(7, activation = "softmax")
])
""" There are 7 possible emotions..now we compile the model. """
model.compile(optimizer = "adam",
              loss = "sparse_categorical_crossentropy",
              metrics=["accuracy"]
              )
""" And then we train the model. Behavior Classification..and Speech Expression
Recognition are the thing! That are "closely related". """
model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test))
""" After making predictions on the test set, and compiling the CNN model which
means that yes, we've got these "dense layers" and we're going to preprocess
the data by reshaping the MFCC features..and splitting them, into the testing
and training sets. And then we make the predictions on the test set like what
we've been doing on the WhatsApp. """
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis = 1)
""" And then we evaluate accuracy. But in order to evaluate accuracy, we just
need to know that we're predicting on the test set. So we need a test set.
We need..we "need" the trained CNN model and to evaluate the performance, of the
model by calculating our accuracy. """
accuracy = accuracy_score(y_test, predicted_classes)
print(f"Accuracy: {accuracy * 100:.2f}%")
""" But that's "our" accuracy in the sense that we're actually generating,
we generate this classification report with stuff like F1 Score, Recall, Precision..
and we do it for each emotion category. """
report = classification_report(y_test, predicted_classes, target_names = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised"]
)
print(report)
""" And there's this function for augmenting the audio data....by application of
"pitch shift" and "noise add" to the original audio signal..and that means we can
extract the features for MFCC from the audio that we augment. And that is how we
improve the robustness..of our model, by simulating a variety of auditory scenarios.
But as of right now we only have this audio processing integrated with what we've
got which is a single "sample audio file" and that's our machine learning, desire
and technique to aim to extract these features, from the audio data, and "hopefully"
use them for some more classification, of emotions.   """
def load_and_augment(audio_path, n_steps=4, noise_factor=0.005, n_mfcc=13):
    y, sr = librosa.load(audio_path)
    y_pitched = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    y_noisy = y + noise_factor * np.random.randn(len(y))
    mfcc_original = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    mfcc_pitched = librosa.feature.mfcc(y=y_pitched, sr=sr, n_mfcc=n_mfcc).T
    mfcc_noisy = librosa.feature.mfcc(y=y_noisy, sr=sr, n_mfcc=n_mfcc).T
    return mfcc_original, mfcc_pitched, mfcc_noisy
audio_path = "cinematic-music-sketches-11-cinematic-percussion-sketch-116186.mp3"
mfcc_original, mfcc_pitched, mfcc_noisy = load_and_augment(audio_path)
""" So "as before" let's try a different appraoch and we are going to have some
dummy labels, for the purposes of demonstration. Then we can replicate ..them,
for the augmented data. """
labels = np.zeros((mfcc_original.shape[0],))
labels = np.concatenate([labels, labels, labels])
""" And we can combine "them" for the MFCC features nad labels... """
X = np.concatenate([mfcc_original, mfcc_pitched, mfcc_noisy], axis = 0)
y = labels
""" And then we reshape it so that we can get our Convolutional Neural Network
input. The "goal" is to destructure and then add "another dimension" for the channel. """
X_reshaped = X[..., np.newaxis]
""" Split the dataset, into training and testing subsets, via the identical
method as before. We did some of this on WhatsApp but that was just the preview. """
(X_train, X_test,
 y_train, y_test) = train_test_split(
          X_reshaped,
          y,
          test_size = 0.2,
          random_state = 42
      )
""" What is the input shape, for the model!?  """
input_shape = X_train.shape[1:]
""" And, wow how do you define the model that has the Input layer, how
do you assume.."assume" that X_train is "actually" reshaped properly, for
our Conv1D? The answer lies in the fact tha twe take this thing where
we take the (time_steps, n_mfcc) and we do that for the 7 emotions and
or categories that we have.."chosen"."""
model = models.Sequential([
    layers.Input(shape = (X_train.shape[1], X_train.shape[2])),
    layers.Conv1D(32, 3, activation = "relu"),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation = "relu"),
    layers.Dense(7, activation = "softmax")
])
""" And now we have yet to be..rebuild, our model for the Deep Neural Network
aspect. We need some sort of benchmark..it's "highly unusual" to be doing
some kind of Conv2D activity,, we need to do Conv1D. And in that sense,
we're doing it because it's a temporal data. The audio data, that is. """
