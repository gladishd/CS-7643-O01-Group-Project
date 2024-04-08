""" In the development of our model, the most important thing is to
make sure that the expected input shape, and this is the kind of thing
that people like Thanushan and Peter, and "Suba" Subanky and they talk about
this a lot, the shape of the output from the preceding layer. This is some
"basic" matrix algebra of course, but then again it comes from the preceding
layer (this is the Flatten layer). And the error messages that I get indicate
something very special. They indicate that the dense layer has some kind of
expectation; it expects the inputs to have a size of 55552, but in "reality" they
actually have the inputs that have size 44800.

The critical thing to note is that Python version 3.10 is too far, for Tensorflow.
I just did
(new_env) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✔) conda install python=3.9
python -m pip install tensorflow_model_optimization
python -m pip install tensorflow-macos==2.9.0 tensorflow-metal==0.6.0 --force-reinstall
https://forums.developer.apple.com/forums/thread/710048
python -m pip uninstall tensorflow tensorflow-macos tensorflow-metal
python -m pip install tensorflow-macos tensorflow-metal


Now, this brings about some kind of discrepancy. The discrepancy "arises" from
the dimensions of the input spectrograms and "this, I think," is the way that
the input spectrograms' dimensions match with the processing of the "convolutional"
and pooling layers of our model.

The first thing that we need to do is adjust the architecture of the model, in
order to make sure that the output of the Flatten layer matches the expected
input size of the dense layer. This could involve changing the size of the
convolutional filters, the stride or size of the pooling layers, or adding/removing
layers. Or, we could dynamically calculate the input size for the dense layer
based on the output shape of the Flatten Layer. That involves modifying the
architecture of the model so that it adapts automatically to the size of its
input.

"So what we do" is we dynamically calculate the input shape for the dense layer
and thereby make sure that my spectrogram processing and workflow for emotion
prediction is intact. The modified section includes the dynamic calculation of
the input units of the dense layer, based on the shape of the flattened output.

We have made some adjustments to the model: it only processes the "first few"
audio files, which means that we can add a limit to the number of files that
"are to be" processed. That's what the variable `max_files` is for.

So for the sake of your iTerm, when you get that cnn_speech_emotion_recognition_model.keras is 28.1 GB... don't cd into "that" directory.

There are several other things that we can do for quick iterations.

"Supposedly" we can reduce the number of epochs. We can also - during the initial phase, we can train for 1-5 epochs, which will give us a good idea of whether our model starts to learn as expected.

history = model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test))

Another thing that we can do is use a subset of our data. This I think we are "already"
doing what with the processing of fewer files, which speeds up both the data
loading and the training phase.

While "we're on" that topic we could also simplify the model. By simplify the model
I mean use a simpler or smaller model with fewer layers and neurons. This reduces
the computation time required for each training step.

We also increase the batch size. Increasing the batch size can speed up training
because it reduces the number of updates per epoch. However, it can also affect the
training "dynamics" and may require adjusting the learning rate.

We could also use model checkpoints to save only improvements and and stop early
if our validation metric stops improving. This can save time by avoiding unnecessary
training epochs.

Least but not least we could use pretrained models, enable acceleration options, profiling, data pipeline optimization. The acceleration options come with the TensorFlow, they come with the GPU thing. And you Thanushan know about the GPU thing. You have to have a PC, rather than your typical macOS "slop".

Remember to pip install tensorflow-model-optimization !

    (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) pip install tensorflow-model-optimization

    Defaulting to user installation because normal site-packages is not writeable
    Collecting tensorflow-model-optimization
    Downloading tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl.metadata (904 bytes)
    Collecting absl-py~=1.2 (from tensorflow-model-optimization)
    Downloading absl_py-1.4.0-py3-none-any.whl.metadata (2.3 kB)
    Collecting dm-tree~=0.1.1 (from tensorflow-model-optimization)
    Downloading dm_tree-0.1.8-cp310-cp310-macosx_11_0_arm64.whl.metadata (1.9 kB)
    Requirement already satisfied: numpy~=1.23 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.26.0)
    Requirement already satisfied: six~=1.14 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow-model-optimization) (1.16.0)
    Downloading tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl (242 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 242.5/242.5 kB 3.6 MB/s eta 0:00:00
    Downloading absl_py-1.4.0-py3-none-any.whl (126 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 126.5/126.5 kB 12.5 MB/s eta 0:00:00
    Downloading dm_tree-0.1.8-cp310-cp310-macosx_11_0_arm64.whl (110 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 110.7/110.7 kB 19.1 MB/s eta 0:00:00
    Installing collected packages: dm-tree, absl-py, tensorflow-model-optimization
    Successfully installed absl-py-1.4.0 dm-tree-0.1.8 tensorflow-model-optimization-0.8.0
    (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)

But what we really need is a "Smart" Terminal to read and do the conda and pip
autocomplete, cross-platform meaning across pip to conda, auto-complete of
the phrase "tensorflow_model_optimization".
Then again maybe that's too "far-fetched".

Maybe base it on the past `history`!

But anyway, all these libraries you can install via pip or conda but if you
are using Conda then "just remember", that the Conda environment matters!
"Sometimes" Python will not "be able to" recognize those (not install them,
installing works).

"""


""" And we have come so far! We went from GitHub (Thanushan I got your GitHub,
"Where are you" when you and the rest of our group want to collaborate on
GitHub? What are your handles? "the GitHub?"). Also, the proposal is something,
in our proposal we go beyond the traditional. We already got this HMM Transition
Matrix, and now when you check the articles, the Siam & Aouani & Issa & Kim
articles..with the Speech Emotion Recognition ..being a nice way to "get the"
Ed. M. student in Applied Linguistics to do OMSCS but to do it with us, to do
Linguistics.

There are so many questions to ask with regard to the traveling and to the
conference. We can have a meeting "next week" and start working on the project.
We have K.T. Shan, a PhD student in Civil Engineering "who" is doing OMSCS part-time..
and that is what it is, this is why "we" focus on quiz 4 and Assignments 4
for now. And then we have a meeting and decide what is the next step.

We find a sample project for Speech Recognition that has been published, and
we have done that. All we have to do is do some modifications and use different
data sets. I would consider this Convolutional Neural Network to be one of the
first things that we do, based on our "project proposal" which is on our
WhatsApp group...

But "whether or not" you're checking emails and WhatsApp properly, the adding
of the new team member..Peter.

We can't even submit the new proposal since the deadline is already passed.

The only thing that changed is the size of our group. I was just sort of freaking
out that night 'because Peter joined in and he seems quite talented.'.

Therefore what we want to do is use our existing proposal, Peter or not on the
bottom of the document..and then have Thanushan and Subanky, we submit the
new Proposal.

This is the right project you want to be in too. We are working on language data.

What we can do is use mamba for installation, which is a faster alternative to
conda that can "massively" help, sometimes resolve the package conflicts "with"
more efficiency.
"""


