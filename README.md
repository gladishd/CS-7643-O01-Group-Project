Data sets: https://zenodo.org/records/1188976#.XrC7a5NKjOR
 https://github.com/CheyneyComputerScience/CREMA-D


pip install librosa matplotlib numpy



commit cb20a0c38e7302854929ed7000641919443c666b (HEAD -> main)
Author: gladishd <gladish.dean@gmail.com>
Date:   Tue Mar 19 03:06:27 2024 -0400

    commit

diff --git a/Problem Set For CS7641/import_boston_housing_test_part_1_question_4.py b/Problem Set For CS7641/import_boston_housing_test_part_1_question_4.py
index 29debcf..bbab326 100644
--- a/Problem Set For CS7641/import_boston_housing_test_part_1_question_4.py
+++ b/Problem Set For CS7641/import_boston_housing_test_part_1_question_4.py
@@ -1 +1,63 @@
-from sklearn.datasets import load_boston
+import pandas as pd
+from sklearn.model_selection import train_test_split
+from sklearn.tree import DecisionTreeRegressor
+from sklearn.metrics import mean_squared_error, r2_score
+from sklearn.datasets import fetch_california_housing
+
+""" But, we're not actually going to fetch the California housing dataset.
+We've got to import the Boston Housing dataset..which is the thing that we
+get when we run `from sklearn.datasets import load_boston` right..but what
+we need to get is we need to get the California housing dataset. """
+housing = fetch_california_housing()
+""" We're going to do the "exact same thing" but we're going to do it in
+the format of a Pandas DataFrame which allows us to go into more detail
+on why ...the anecdote, speaking of anecdotes I would say that the air
+quality that we have in this course is probably "worse" than the air quality
+that they have in the Boston Housing Dataset. And because of that, we need
+to import all the data that we want but we have to follow the "Ruliad" of
+housing feature names.  """
+data = pd.DataFrame(housing.data, columns=housing.feature_names)
+target = housing.target
+""" Then, we print out what are the first few rows..well what are they?
+They are things like MedIncome, Housing Age, Average Rooms, Average
+Bedrooms, to "Name" a few. However, I need to print these out and then
+we can use this as the basis for structuring this dataset..except it's a regular
+dataset in the sense that it's the "typical one for California housing". """
+print(data.head())
+""" Then, we can check for the missing values..there aren't really any
+missing values to speak of, but it doesn't "hurt" to check. """
+print(data.isnull().sum())
+""" Then, we need to load in our California Housing Dataset. We need to load
+it in by splitting the dataset into testing and training sets.. """
(base) ~/CS-7641/CS7641 Unsupervised Learning and Dimensionality Reduction/Problem Set For CS7641 (main ✔) git show
(base) ~/CS-7641/CS7641 Unsupervised Learning and Dimensionality Reduction/Problem Set For CS7641 (main ✔) git commit --amend
[main 445e521] Replace Boston Housing Dataset with California Housing for Ethical Regression Analysis
 Date: Tue Mar 19 03:06:27 2024 -0400
 1 file changed, 63 insertions(+), 1 deletion(-)
 rewrite Problem Set For CS7641/import_boston_housing_test_part_1_question_4.py (100%)
(base) ~/CS-7641/CS7641 Unsupervised Learning and Dimensionality Reduction/Problem Set For CS7641 (main ✔) cd ...
(base) ~/CS-7641 ls
CS7641                                                    Learning
CS7641 Unsupervised Learning and Dimensionality Reduction Reduction
CS7641-Randomized-Optimization-Assignment-2               Unsupervised
CS7641-Supervised-Learning-Assignment                     and
Dimensionality
(base) ~/CS-7641 cd ..
(base) ~ cd CS-7643
cd: no such file or directory: CS-7643
(base) ~ cd CS-7643-O01
(base) ~/CS-7643-O01 ls
A3                             Assignment 2, Question 2.log   Assignment 2.aux               Assignment 2.synctex.gz        assignment1-theory-problem-set
A3.zip                         Assignment 2, Question 2.pdf   Assignment 2.log               Assignment 2.tex               assignment2-spring2024
Assignment 2, Question 2.aux   Assignment 2, Question 2.tex   Assignment 2.pdf               assignment1                    transposed_convolution.py
(base) ~/CS-7643-O01 mkdir Group_Project
(base) ~/CS-7643-O01 cd Group_Project
(base) ~/CS-7643-O01/Group_Project code .
(base) ~/CS-7643-O01/Group_Project open .
(base) ~/CS-7643-O01/Group_Project touch practice.py
(base) ~/CS-7643-O01/Group_Project code .
(base) ~/CS-7643-O01/Group_Project pip install librosa matplotlib numpy

Collecting librosa
  Obtaining dependency information for librosa from https://files.pythonhosted.org/packages/e2/a2/4f639c1168d7aada749a896afb4892a831e2041bebdcf636aebfe9e86556/librosa-0.10.1-py3-none-any.whl.metadata
  Downloading librosa-0.10.1-py3-none-any.whl.metadata (8.3 kB)
Requirement already satisfied: matplotlib in /Users/deangladish/miniforge3/lib/python3.10/site-packages (3.7.2)
Requirement already satisfied: numpy in /Users/deangladish/miniforge3/lib/python3.10/site-packages (1.25.2)
Collecting audioread>=2.1.9 (from librosa)
  Obtaining dependency information for audioread>=2.1.9 from https://files.pythonhosted.org/packages/57/8d/30aa32745af16af0a9a650115fbe81bde7c610ed5c21b381fca0196f3a7f/audioread-3.0.1-py3-none-any.whl.metadata
  Downloading audioread-3.0.1-py3-none-any.whl.metadata (8.4 kB)
Requirement already satisfied: scipy>=1.2.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from librosa) (1.10.1)
Requirement already satisfied: scikit-learn>=0.20.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from librosa) (1.3.2)
Requirement already satisfied: joblib>=0.14 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from librosa) (1.3.2)
Collecting decorator>=4.3.0 (from librosa)
  Obtaining dependency information for decorator>=4.3.0 from https://files.pythonhosted.org/packages/d5/50/83c593b07763e1161326b3b8c6686f0f4b0f24d5526546bee538c89837d6/decorator-5.1.1-py3-none-any.whl.metadata
  Downloading decorator-5.1.1-py3-none-any.whl.metadata (4.0 kB)
Collecting numba>=0.51.0 (from librosa)
  Obtaining dependency information for numba>=0.51.0 from https://files.pythonhosted.org/packages/85/df/28bfa1846541892fda4790fde7d70ea6265fd66325961ea07c6d597a28ec/numba-0.59.0-cp310-cp310-macosx_11_0_arm64.whl.metadata
  Downloading numba-0.59.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (2.7 kB)
Collecting soundfile>=0.12.1 (from librosa)
  Obtaining dependency information for soundfile>=0.12.1 from https://files.pythonhosted.org/packages/71/87/31d2b9ed58975cec081858c01afaa3c43718eb0f62b5698a876d94739ad0/soundfile-0.12.1-py2.py3-none-macosx_11_0_arm64.whl.metadata
  Downloading soundfile-0.12.1-py2.py3-none-macosx_11_0_arm64.whl.metadata (14 kB)
Requirement already satisfied: pooch>=1.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from librosa) (1.8.0)
Collecting soxr>=0.3.2 (from librosa)
  Obtaining dependency information for soxr>=0.3.2 from https://files.pythonhosted.org/packages/bc/38/2635bcf180de54457d64a6b348b3e421f469aee7edafead2306a6e74cc1a/soxr-0.3.7-cp310-cp310-macosx_11_0_arm64.whl.metadata
  Downloading soxr-0.3.7-cp310-cp310-macosx_11_0_arm64.whl.metadata (5.5 kB)
Requirement already satisfied: typing-extensions>=4.1.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from librosa) (4.8.0)
Collecting lazy-loader>=0.1 (from librosa)
  Obtaining dependency information for lazy-loader>=0.1 from https://files.pythonhosted.org/packages/a1/c3/65b3814e155836acacf720e5be3b5757130346670ac454fee29d3eda1381/lazy_loader-0.3-py3-none-any.whl.metadata
  Downloading lazy_loader-0.3-py3-none-any.whl.metadata (4.3 kB)
Collecting msgpack>=1.0 (from librosa)
  Obtaining dependency information for msgpack>=1.0 from https://files.pythonhosted.org/packages/ba/13/d000e53b067aee19d57a4f26d5bffed7890e6896538ac5f97605b0f64985/msgpack-1.0.8-cp310-cp310-macosx_11_0_arm64.whl.metadata
  Downloading msgpack-1.0.8-cp310-cp310-macosx_11_0_arm64.whl.metadata (9.1 kB)
Requirement already satisfied: contourpy>=1.0.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from matplotlib) (1.1.0)
Requirement already satisfied: cycler>=0.10 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from matplotlib) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from matplotlib) (4.42.1)
Requirement already satisfied: kiwisolver>=1.0.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from matplotlib) (1.4.5)
Requirement already satisfied: packaging>=20.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from matplotlib) (23.1)
Requirement already satisfied: pillow>=6.2.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from matplotlib) (10.0.0)
Requirement already satisfied: pyparsing<3.1,>=2.3.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from matplotlib) (3.0.9)
Requirement already satisfied: python-dateutil>=2.7 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from matplotlib) (2.8.2)
Collecting llvmlite<0.43,>=0.42.0dev0 (from numba>=0.51.0->librosa)
  Obtaining dependency information for llvmlite<0.43,>=0.42.0dev0 from https://files.pythonhosted.org/packages/4f/c3/aa006e8cbd02e756352342146dc95d6d5880bc32d566be8f0c0e0f202796/llvmlite-0.42.0-cp310-cp310-macosx_11_0_arm64.whl.metadata
  Downloading llvmlite-0.42.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (4.8 kB)
Requirement already satisfied: platformdirs>=2.5.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from pooch>=1.0->librosa) (3.11.0)
Requirement already satisfied: requests>=2.19.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from pooch>=1.0->librosa) (2.31.0)
Requirement already satisfied: six>=1.5 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from scikit-learn>=0.20.0->librosa) (3.2.0)
Requirement already satisfied: cffi>=1.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from soundfile>=0.12.1->librosa) (1.15.1)
Requirement already satisfied: pycparser in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from cffi>=1.0->soundfile>=0.12.1->librosa) (2.21)
Requirement already satisfied: charset-normalizer<4,>=2 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests>=2.19.0->pooch>=1.0->librosa) (3.2.0)
Requirement already satisfied: idna<4,>=2.5 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests>=2.19.0->pooch>=1.0->librosa) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests>=2.19.0->pooch>=1.0->librosa) (2.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests>=2.19.0->pooch>=1.0->librosa) (2023.7.22)
Downloading librosa-0.10.1-py3-none-any.whl (253 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 253.7/253.7 kB 139.6 kB/s eta 0:00:00
Downloading audioread-3.0.1-py3-none-any.whl (23 kB)
Using cached decorator-5.1.1-py3-none-any.whl (9.1 kB)
Downloading lazy_loader-0.3-py3-none-any.whl (9.1 kB)
Downloading msgpack-1.0.8-cp310-cp310-macosx_11_0_arm64.whl (84 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.9/84.9 kB 145.6 kB/s eta 0:00:00
Downloading numba-0.59.0-cp310-cp310-macosx_11_0_arm64.whl (2.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.6/2.6 MB 137.6 kB/s eta 0:00:00
Downloading soundfile-0.12.1-py2.py3-none-macosx_11_0_arm64.whl (1.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 149.0 kB/s eta 0:00:00
Downloading soxr-0.3.7-cp310-cp310-macosx_11_0_arm64.whl (390 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 390.0/390.0 kB 189.7 kB/s eta 0:00:00
Downloading llvmlite-0.42.0-cp310-cp310-macosx_11_0_arm64.whl (28.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 28.8/28.8 MB 154.6 kB/s eta 0:00:00
Installing collected packages: soxr, msgpack, llvmlite, lazy-loader, decorator, audioread, soundfile, numba, librosa
Successfully installed audioread-3.0.1 decorator-5.1.1 lazy-loader-0.3 librosa-0.10.1 llvmlite-0.42.0 msgpack-1.0.8 numba-0.59.0 soundfile-0.12.1 soxr-0.3.7
(base) ~/CS-7643-O01/Group_Project



********************************************************************************



(base) ~/CS-7643-O01/Group_Project python practice.py
Traceback (most recent call last):
  File "/Users/deangladish/CS-7643-O01/Group_Project/practice.py", line 54, in <module>
    from hmmlearn import hmm
ModuleNotFoundError: No module named 'hmmlearn'
(base) ~/CS-7643-O01/Group_Project pip install hmmlearn
Collecting hmmlearn
  Obtaining dependency information for hmmlearn from https://files.pythonhosted.org/packages/a3/41/17372c10df3e450d4ec3eea47b75fac7aa830c49a9be3e801b0111acf346/hmmlearn-0.3.2-cp310-cp310-macosx_10_9_universal2.whl.metadata
  Downloading hmmlearn-0.3.2-cp310-cp310-macosx_10_9_universal2.whl.metadata (2.9 kB)
Requirement already satisfied: numpy>=1.10 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from hmmlearn) (1.25.2)
Requirement already satisfied: scikit-learn!=0.22.0,>=0.16 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from hmmlearn) (1.3.2)
Requirement already satisfied: scipy>=0.19 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from hmmlearn) (1.10.1)
Requirement already satisfied: joblib>=1.1.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from scikit-learn!=0.22.0,>=0.16->hmmlearn) (1.3.2)
Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from scikit-learn!=0.22.0,>=0.16->hmmlearn) (3.2.0)
Downloading hmmlearn-0.3.2-cp310-cp310-macosx_10_9_universal2.whl (192 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 192.6/192.6 kB 47.5 kB/s eta 0:00:00
Installing collected packages: hmmlearn
Successfully installed hmmlearn-0.3.2
(base) ~/CS-7643-O01/Group_Project



********************************************************************************



(base) ~/CS-7643-O01/Group_Project (main ✗) python practice.py
Traceback (most recent call last):
  File "/Users/deangladish/CS-7643-O01/Group_Project/practice.py", line 84, in <module>
    import tensorflow as tf
ModuleNotFoundError: No module named 'tensorflow'
(base) ~/CS-7643-O01/Group_Project (main ✗) pip install tensorflow
Collecting tensorflow
  Obtaining dependency information for tensorflow from https://files.pythonhosted.org/packages/7d/01/bee34cf4d207cc5ae4f445c0e743691697cd89359a24a5fcdcfa8372f042/tensorflow-2.16.1-cp310-cp310-macosx_12_0_arm64.whl.metadata
  Downloading tensorflow-2.16.1-cp310-cp310-macosx_12_0_arm64.whl.metadata (4.1 kB)
Collecting absl-py>=1.0.0 (from tensorflow)
  Obtaining dependency information for absl-py>=1.0.0 from https://files.pythonhosted.org/packages/a2/ad/e0d3c824784ff121c03cc031f944bc7e139a8f1870ffd2845cc2dd76f6c4/absl_py-2.1.0-py3-none-any.whl.metadata
  Downloading absl_py-2.1.0-py3-none-any.whl.metadata (2.3 kB)
Collecting astunparse>=1.6.0 (from tensorflow)
  Obtaining dependency information for astunparse>=1.6.0 from https://files.pythonhosted.org/packages/2b/03/13dde6512ad7b4557eb792fbcf0c653af6076b81e5941d36ec61f7ce6028/astunparse-1.6.3-py2.py3-none-any.whl.metadata
  Downloading astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)
Collecting flatbuffers>=23.5.26 (from tensorflow)
  Obtaining dependency information for flatbuffers>=23.5.26 from https://files.pythonhosted.org/packages/bf/45/c961e3cb6ddad76b325c163d730562bb6deb1ace5acbed0306f5fbefb90e/flatbuffers-24.3.7-py2.py3-none-any.whl.metadata
  Downloading flatbuffers-24.3.7-py2.py3-none-any.whl.metadata (849 bytes)
Collecting gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 (from tensorflow)
  Obtaining dependency information for gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 from https://files.pythonhosted.org/packages/fa/39/5aae571e5a5f4de9c3445dae08a530498e5c53b0e74410eeeb0991c79047/gast-0.5.4-py3-none-any.whl.metadata
  Downloading gast-0.5.4-py3-none-any.whl.metadata (1.3 kB)
Collecting google-pasta>=0.1.1 (from tensorflow)
  Obtaining dependency information for google-pasta>=0.1.1 from https://files.pythonhosted.org/packages/a3/de/c648ef6835192e6e2cc03f40b19eeda4382c49b5bafb43d88b931c4c74ac/google_pasta-0.2.0-py3-none-any.whl.metadata
  Downloading google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)
Collecting h5py>=3.10.0 (from tensorflow)
  Obtaining dependency information for h5py>=3.10.0 from https://files.pythonhosted.org/packages/2c/8b/b173963891023310ba849c44509e61ada94fda87123e6ba4e91ec8401183/h5py-3.10.0-cp310-cp310-macosx_11_0_arm64.whl.metadata
  Downloading h5py-3.10.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (2.5 kB)
Collecting libclang>=13.0.0 (from tensorflow)
  Obtaining dependency information for libclang>=13.0.0 from https://files.pythonhosted.org/packages/db/ed/1df62b44db2583375f6a8a5e2ca5432bbdc3edb477942b9b7c848c720055/libclang-18.1.1-py2.py3-none-macosx_11_0_arm64.whl.metadata
  Downloading libclang-18.1.1-py2.py3-none-macosx_11_0_arm64.whl.metadata (5.2 kB)
Collecting ml-dtypes~=0.3.1 (from tensorflow)
  Obtaining dependency information for ml-dtypes~=0.3.1 from https://files.pythonhosted.org/packages/62/0a/2b586fd10be7b8311068f4078623a73376fc49c8b3768be9965034062982/ml_dtypes-0.3.2-cp310-cp310-macosx_10_9_universal2.whl.metadata
  Downloading ml_dtypes-0.3.2-cp310-cp310-macosx_10_9_universal2.whl.metadata (20 kB)
Collecting opt-einsum>=2.3.2 (from tensorflow)
  Obtaining dependency information for opt-einsum>=2.3.2 from https://files.pythonhosted.org/packages/bc/19/404708a7e54ad2798907210462fd950c3442ea51acc8790f3da48d2bee8b/opt_einsum-3.3.0-py3-none-any.whl.metadata
  Downloading opt_einsum-3.3.0-py3-none-any.whl.metadata (6.5 kB)
Requirement already satisfied: packaging in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow) (23.1)
Collecting protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 (from tensorflow)
  Obtaining dependency information for protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 from https://files.pythonhosted.org/packages/f3/bf/26deba06a4c910a85f78245cac7698f67cedd7efe00d04f6b3e1b3506a59/protobuf-4.25.3-cp37-abi3-macosx_10_9_universal2.whl.metadata
  Downloading protobuf-4.25.3-cp37-abi3-macosx_10_9_universal2.whl.metadata (541 bytes)
Requirement already satisfied: requests<3,>=2.21.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow) (2.31.0)
Requirement already satisfied: setuptools in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow) (68.1.2)
Requirement already satisfied: six>=1.12.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow) (1.16.0)
Collecting termcolor>=1.1.0 (from tensorflow)
  Obtaining dependency information for termcolor>=1.1.0 from https://files.pythonhosted.org/packages/d9/5f/8c716e47b3a50cbd7c146f45881e11d9414def768b7cd9c5e6650ec2a80a/termcolor-2.4.0-py3-none-any.whl.metadata
  Downloading termcolor-2.4.0-py3-none-any.whl.metadata (6.1 kB)
Requirement already satisfied: typing-extensions>=3.6.6 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow) (4.8.0)
Collecting wrapt>=1.11.0 (from tensorflow)
  Obtaining dependency information for wrapt>=1.11.0 from https://files.pythonhosted.org/packages/32/12/e11adfde33444986135d8881b401e4de6cbb4cced046edc6b464e6ad7547/wrapt-1.16.0-cp310-cp310-macosx_11_0_arm64.whl.metadata
  Downloading wrapt-1.16.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (6.6 kB)
Collecting grpcio<2.0,>=1.24.3 (from tensorflow)
  Obtaining dependency information for grpcio<2.0,>=1.24.3 from https://files.pythonhosted.org/packages/cc/fb/09c2e42f37858f699b5f56e40f2c3a45fb24b1b7a9dbed3ae1ca7e5fbac9/grpcio-1.62.1-cp310-cp310-macosx_12_0_universal2.whl.metadata
  Downloading grpcio-1.62.1-cp310-cp310-macosx_12_0_universal2.whl.metadata (4.0 kB)
Collecting tensorboard<2.17,>=2.16 (from tensorflow)
  Obtaining dependency information for tensorboard<2.17,>=2.16 from https://files.pythonhosted.org/packages/3a/d0/b97889ffa769e2d1fdebb632084d5e8b53fc299d43a537acee7ec0c021a3/tensorboard-2.16.2-py3-none-any.whl.metadata
  Downloading tensorboard-2.16.2-py3-none-any.whl.metadata (1.6 kB)
Collecting keras>=3.0.0 (from tensorflow)
  Obtaining dependency information for keras>=3.0.0 from https://files.pythonhosted.org/packages/38/28/63b0e7851c36dcb1a10757d598c68cc1e48a669bdb63bfdd9a1b9b1c643f/keras-3.1.0-py3-none-any.whl.metadata
  Downloading keras-3.1.0-py3-none-any.whl.metadata (5.6 kB)
Collecting tensorflow-io-gcs-filesystem>=0.23.1 (from tensorflow)
  Obtaining dependency information for tensorflow-io-gcs-filesystem>=0.23.1 from https://files.pythonhosted.org/packages/c7/64/bb98ed6e6b797c134d66cb199e2d5b998cfcb9afff0312bc01665b3a6700/tensorflow_io_gcs_filesystem-0.36.0-cp310-cp310-macosx_12_0_arm64.whl.metadata
  Downloading tensorflow_io_gcs_filesystem-0.36.0-cp310-cp310-macosx_12_0_arm64.whl.metadata (14 kB)
Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow) (1.25.2)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from astunparse>=1.6.0->tensorflow) (0.41.2)
Collecting rich (from keras>=3.0.0->tensorflow)
  Obtaining dependency information for rich from https://files.pythonhosted.org/packages/87/67/a37f6214d0e9fe57f6ae54b2956d550ca8365857f42a1ce0392bb21d9410/rich-13.7.1-py3-none-any.whl.metadata
  Downloading rich-13.7.1-py3-none-any.whl.metadata (18 kB)
Collecting namex (from keras>=3.0.0->tensorflow)
  Obtaining dependency information for namex from https://files.pythonhosted.org/packages/cd/43/b971880e2eb45c0bee2093710ae8044764a89afe9620df34a231c6f0ecd2/namex-0.0.7-py3-none-any.whl.metadata
  Downloading namex-0.0.7-py3-none-any.whl.metadata (246 bytes)
Collecting optree (from keras>=3.0.0->tensorflow)
  Obtaining dependency information for optree from https://files.pythonhosted.org/packages/e3/f7/d626e2e0dbbeaa54ea9ee2375638ae0995bdaf7e5c4671212346a95d61f7/optree-0.10.0-cp310-cp310-macosx_11_0_arm64.whl.metadata
  Downloading optree-0.10.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (45 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.3/45.3 kB 183.1 kB/s eta 0:00:00
Requirement already satisfied: charset-normalizer<4,>=2 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (3.2.0)
Requirement already satisfied: idna<4,>=2.5 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (2.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (2023.7.22)
Collecting markdown>=2.6.8 (from tensorboard<2.17,>=2.16->tensorflow)
  Obtaining dependency information for markdown>=2.6.8 from https://files.pythonhosted.org/packages/fc/b3/0c0c994fe49cd661084f8d5dc06562af53818cc0abefaca35bdc894577c3/Markdown-3.6-py3-none-any.whl.metadata
  Downloading Markdown-3.6-py3-none-any.whl.metadata (7.0 kB)
Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard<2.17,>=2.16->tensorflow)
  Obtaining dependency information for tensorboard-data-server<0.8.0,>=0.7.0 from https://files.pythonhosted.org/packages/7a/13/e503968fefabd4c6b2650af21e110aa8466fe21432cd7c43a84577a89438/tensorboard_data_server-0.7.2-py3-none-any.whl.metadata
  Downloading tensorboard_data_server-0.7.2-py3-none-any.whl.metadata (1.1 kB)
Collecting werkzeug>=1.0.1 (from tensorboard<2.17,>=2.16->tensorflow)
  Obtaining dependency information for werkzeug>=1.0.1 from https://files.pythonhosted.org/packages/c3/fc/254c3e9b5feb89ff5b9076a23218dafbc99c96ac5941e900b71206e6313b/werkzeug-3.0.1-py3-none-any.whl.metadata
  Downloading werkzeug-3.0.1-py3-none-any.whl.metadata (4.1 kB)
Requirement already satisfied: MarkupSafe>=2.1.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard<2.17,>=2.16->tensorflow) (2.1.5)
Collecting markdown-it-py>=2.2.0 (from rich->keras>=3.0.0->tensorflow)
  Obtaining dependency information for markdown-it-py>=2.2.0 from https://files.pythonhosted.org/packages/42/d7/1ec15b46af6af88f19b8e5ffea08fa375d433c998b8a7639e76935c14f1f/markdown_it_py-3.0.0-py3-none-any.whl.metadata
  Downloading markdown_it_py-3.0.0-py3-none-any.whl.metadata (6.9 kB)
Collecting pygments<3.0.0,>=2.13.0 (from rich->keras>=3.0.0->tensorflow)
  Obtaining dependency information for pygments<3.0.0,>=2.13.0 from https://files.pythonhosted.org/packages/97/9c/372fef8377a6e340b1704768d20daaded98bf13282b5327beb2e2fe2c7ef/pygments-2.17.2-py3-none-any.whl.metadata
  Downloading pygments-2.17.2-py3-none-any.whl.metadata (2.6 kB)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich->keras>=3.0.0->tensorflow)
  Obtaining dependency information for mdurl~=0.1 from https://files.pythonhosted.org/packages/b3/38/89ba8ad64ae25be8de66a6d463314cf1eb366222074cfda9ee839c56a4b4/mdurl-0.1.2-py3-none-any.whl.metadata
  Downloading mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Downloading tensorflow-2.16.1-cp310-cp310-macosx_12_0_arm64.whl (227.0 MB)
   ━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 21.2/227.0 MB 139.7 kB/s eta 0:24:34
ERROR: Exception:
Traceback (most recent call last):
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_vendor/urllib3/response.py", line 438, in _error_catcher
    yield
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_vendor/urllib3/response.py", line 561, in read
    data = self._fp_read(amt) if not fp_closed else b""
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_vendor/urllib3/response.py", line 527, in _fp_read
    return self._fp.read(amt) if amt is not None else self._fp.read()
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_vendor/cachecontrol/filewrapper.py", line 90, in read
    data = self.__fp.read(amt)
  File "/Users/deangladish/miniforge3/lib/python3.10/http/client.py", line 466, in read
    s = self.fp.read(amt)
  File "/Users/deangladish/miniforge3/lib/python3.10/socket.py", line 705, in readinto
    return self._sock.recv_into(b)
  File "/Users/deangladish/miniforge3/lib/python3.10/ssl.py", line 1274, in recv_into
    return self.read(nbytes, buffer)
  File "/Users/deangladish/miniforge3/lib/python3.10/ssl.py", line 1130, in read
    return self._sslobj.read(len, buffer)
TimeoutError: The read operation timed out

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_internal/cli/base_command.py", line 180, in exc_logging_wrapper
    status = run_func(*args)
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_internal/cli/req_command.py", line 248, in wrapper
    return func(self, options, args)
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_internal/commands/install.py", line 377, in run
    requirement_set = resolver.resolve(
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/resolver.py", line 161, in resolve
    self.factory.preparer.prepare_linked_requirements_more(reqs)
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_internal/operations/prepare.py", line 565, in prepare_linked_requirements_more
    self._complete_partial_requirements(
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_internal/operations/prepare.py", line 479, in _complete_partial_requirements
    for link, (filepath, _) in batch_download:
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_internal/network/download.py", line 183, in __call__
    for chunk in chunks:
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_internal/cli/progress_bars.py", line 53, in _rich_progress_bar
    for chunk in iterable:
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_internal/network/utils.py", line 63, in response_chunks
    for chunk in response.raw.stream(
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_vendor/urllib3/response.py", line 622, in stream
    data = self.read(amt=amt, decode_content=decode_content)
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_vendor/urllib3/response.py", line 560, in read
    with self._error_catcher():
  File "/Users/deangladish/miniforge3/lib/python3.10/contextlib.py", line 153, in __exit__
    self.gen.throw(typ, value, traceback)
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/pip/_vendor/urllib3/response.py", line 443, in _error_catcher
    raise ReadTimeoutError(self._pool, None, "Read timed out.")
pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.
(base) ~/CS-7643-O01/Group_Project (main ✗) python practice.py

Traceback (most recent call last):
  File "/Users/deangladish/CS-7643-O01/Group_Project/practice.py", line 84, in <module>
    import tensorflow as tf
ModuleNotFoundError: No module named 'tensorflow'
(base) ~/CS-7643-O01/Group_Project (main ✗)
(base) ~/CS-7643-O01/Group_Project (main ✗)



********************************************************************************



(base) ~/CS-7643-O01/Group_Project (main ✗) pip --default-timeout=1000000 install tensorflow

Collecting tensorflow
  Obtaining dependency information for tensorflow from https://files.pythonhosted.org/packages/7d/01/bee34cf4d207cc5ae4f445c0e743691697cd89359a24a5fcdcfa8372f042/tensorflow-2.16.1-cp310-cp310-macosx_12_0_arm64.whl.metadata
  Using cached tensorflow-2.16.1-cp310-cp310-macosx_12_0_arm64.whl.metadata (4.1 kB)
Collecting absl-py>=1.0.0 (from tensorflow)
  Obtaining dependency information for absl-py>=1.0.0 from https://files.pythonhosted.org/packages/a2/ad/e0d3c824784ff121c03cc031f944bc7e139a8f1870ffd2845cc2dd76f6c4/absl_py-2.1.0-py3-none-any.whl.metadata
  Using cached absl_py-2.1.0-py3-none-any.whl.metadata (2.3 kB)
Collecting astunparse>=1.6.0 (from tensorflow)
  Obtaining dependency information for astunparse>=1.6.0 from https://files.pythonhosted.org/packages/2b/03/13dde6512ad7b4557eb792fbcf0c653af6076b81e5941d36ec61f7ce6028/astunparse-1.6.3-py2.py3-none-any.whl.metadata
  Using cached astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)
Collecting flatbuffers>=23.5.26 (from tensorflow)
  Obtaining dependency information for flatbuffers>=23.5.26 from https://files.pythonhosted.org/packages/bf/45/c961e3cb6ddad76b325c163d730562bb6deb1ace5acbed0306f5fbefb90e/flatbuffers-24.3.7-py2.py3-none-any.whl.metadata
  Using cached flatbuffers-24.3.7-py2.py3-none-any.whl.metadata (849 bytes)
Collecting gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 (from tensorflow)
  Obtaining dependency information for gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 from https://files.pythonhosted.org/packages/fa/39/5aae571e5a5f4de9c3445dae08a530498e5c53b0e74410eeeb0991c79047/gast-0.5.4-py3-none-any.whl.metadata
  Using cached gast-0.5.4-py3-none-any.whl.metadata (1.3 kB)
Collecting google-pasta>=0.1.1 (from tensorflow)
  Obtaining dependency information for google-pasta>=0.1.1 from https://files.pythonhosted.org/packages/a3/de/c648ef6835192e6e2cc03f40b19eeda4382c49b5bafb43d88b931c4c74ac/google_pasta-0.2.0-py3-none-any.whl.metadata
  Using cached google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)
Collecting h5py>=3.10.0 (from tensorflow)
  Obtaining dependency information for h5py>=3.10.0 from https://files.pythonhosted.org/packages/2c/8b/b173963891023310ba849c44509e61ada94fda87123e6ba4e91ec8401183/h5py-3.10.0-cp310-cp310-macosx_11_0_arm64.whl.metadata
  Using cached h5py-3.10.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (2.5 kB)
Collecting libclang>=13.0.0 (from tensorflow)
  Obtaining dependency information for libclang>=13.0.0 from https://files.pythonhosted.org/packages/db/ed/1df62b44db2583375f6a8a5e2ca5432bbdc3edb477942b9b7c848c720055/libclang-18.1.1-py2.py3-none-macosx_11_0_arm64.whl.metadata
  Using cached libclang-18.1.1-py2.py3-none-macosx_11_0_arm64.whl.metadata (5.2 kB)
Collecting ml-dtypes~=0.3.1 (from tensorflow)
  Obtaining dependency information for ml-dtypes~=0.3.1 from https://files.pythonhosted.org/packages/62/0a/2b586fd10be7b8311068f4078623a73376fc49c8b3768be9965034062982/ml_dtypes-0.3.2-cp310-cp310-macosx_10_9_universal2.whl.metadata
  Using cached ml_dtypes-0.3.2-cp310-cp310-macosx_10_9_universal2.whl.metadata (20 kB)
Collecting opt-einsum>=2.3.2 (from tensorflow)
  Obtaining dependency information for opt-einsum>=2.3.2 from https://files.pythonhosted.org/packages/bc/19/404708a7e54ad2798907210462fd950c3442ea51acc8790f3da48d2bee8b/opt_einsum-3.3.0-py3-none-any.whl.metadata
  Using cached opt_einsum-3.3.0-py3-none-any.whl.metadata (6.5 kB)
Requirement already satisfied: packaging in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow) (23.1)
Collecting protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 (from tensorflow)
  Obtaining dependency information for protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 from https://files.pythonhosted.org/packages/f3/bf/26deba06a4c910a85f78245cac7698f67cedd7efe00d04f6b3e1b3506a59/protobuf-4.25.3-cp37-abi3-macosx_10_9_universal2.whl.metadata
  Using cached protobuf-4.25.3-cp37-abi3-macosx_10_9_universal2.whl.metadata (541 bytes)
Requirement already satisfied: requests<3,>=2.21.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow) (2.31.0)
Requirement already satisfied: setuptools in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow) (68.1.2)
Requirement already satisfied: six>=1.12.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow) (1.16.0)
Collecting termcolor>=1.1.0 (from tensorflow)
  Obtaining dependency information for termcolor>=1.1.0 from https://files.pythonhosted.org/packages/d9/5f/8c716e47b3a50cbd7c146f45881e11d9414def768b7cd9c5e6650ec2a80a/termcolor-2.4.0-py3-none-any.whl.metadata
  Using cached termcolor-2.4.0-py3-none-any.whl.metadata (6.1 kB)
Requirement already satisfied: typing-extensions>=3.6.6 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow) (4.8.0)
Collecting wrapt>=1.11.0 (from tensorflow)
  Obtaining dependency information for wrapt>=1.11.0 from https://files.pythonhosted.org/packages/32/12/e11adfde33444986135d8881b401e4de6cbb4cced046edc6b464e6ad7547/wrapt-1.16.0-cp310-cp310-macosx_11_0_arm64.whl.metadata
  Using cached wrapt-1.16.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (6.6 kB)
Collecting grpcio<2.0,>=1.24.3 (from tensorflow)
  Obtaining dependency information for grpcio<2.0,>=1.24.3 from https://files.pythonhosted.org/packages/cc/fb/09c2e42f37858f699b5f56e40f2c3a45fb24b1b7a9dbed3ae1ca7e5fbac9/grpcio-1.62.1-cp310-cp310-macosx_12_0_universal2.whl.metadata
  Using cached grpcio-1.62.1-cp310-cp310-macosx_12_0_universal2.whl.metadata (4.0 kB)
Collecting tensorboard<2.17,>=2.16 (from tensorflow)
  Obtaining dependency information for tensorboard<2.17,>=2.16 from https://files.pythonhosted.org/packages/3a/d0/b97889ffa769e2d1fdebb632084d5e8b53fc299d43a537acee7ec0c021a3/tensorboard-2.16.2-py3-none-any.whl.metadata
  Using cached tensorboard-2.16.2-py3-none-any.whl.metadata (1.6 kB)
Collecting keras>=3.0.0 (from tensorflow)
  Obtaining dependency information for keras>=3.0.0 from https://files.pythonhosted.org/packages/38/28/63b0e7851c36dcb1a10757d598c68cc1e48a669bdb63bfdd9a1b9b1c643f/keras-3.1.0-py3-none-any.whl.metadata
  Using cached keras-3.1.0-py3-none-any.whl.metadata (5.6 kB)
Collecting tensorflow-io-gcs-filesystem>=0.23.1 (from tensorflow)
  Obtaining dependency information for tensorflow-io-gcs-filesystem>=0.23.1 from https://files.pythonhosted.org/packages/c7/64/bb98ed6e6b797c134d66cb199e2d5b998cfcb9afff0312bc01665b3a6700/tensorflow_io_gcs_filesystem-0.36.0-cp310-cp310-macosx_12_0_arm64.whl.metadata
  Using cached tensorflow_io_gcs_filesystem-0.36.0-cp310-cp310-macosx_12_0_arm64.whl.metadata (14 kB)
Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow) (1.25.2)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from astunparse>=1.6.0->tensorflow) (0.41.2)
Collecting rich (from keras>=3.0.0->tensorflow)
  Obtaining dependency information for rich from https://files.pythonhosted.org/packages/87/67/a37f6214d0e9fe57f6ae54b2956d550ca8365857f42a1ce0392bb21d9410/rich-13.7.1-py3-none-any.whl.metadata
  Using cached rich-13.7.1-py3-none-any.whl.metadata (18 kB)
Collecting namex (from keras>=3.0.0->tensorflow)
  Obtaining dependency information for namex from https://files.pythonhosted.org/packages/cd/43/b971880e2eb45c0bee2093710ae8044764a89afe9620df34a231c6f0ecd2/namex-0.0.7-py3-none-any.whl.metadata
  Using cached namex-0.0.7-py3-none-any.whl.metadata (246 bytes)
Collecting optree (from keras>=3.0.0->tensorflow)
  Obtaining dependency information for optree from https://files.pythonhosted.org/packages/e3/f7/d626e2e0dbbeaa54ea9ee2375638ae0995bdaf7e5c4671212346a95d61f7/optree-0.10.0-cp310-cp310-macosx_11_0_arm64.whl.metadata
  Using cached optree-0.10.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (45 kB)
Requirement already satisfied: charset-normalizer<4,>=2 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (3.2.0)
Requirement already satisfied: idna<4,>=2.5 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (2.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (2023.7.22)
Collecting markdown>=2.6.8 (from tensorboard<2.17,>=2.16->tensorflow)
  Obtaining dependency information for markdown>=2.6.8 from https://files.pythonhosted.org/packages/fc/b3/0c0c994fe49cd661084f8d5dc06562af53818cc0abefaca35bdc894577c3/Markdown-3.6-py3-none-any.whl.metadata
  Using cached Markdown-3.6-py3-none-any.whl.metadata (7.0 kB)
Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard<2.17,>=2.16->tensorflow)
  Obtaining dependency information for tensorboard-data-server<0.8.0,>=0.7.0 from https://files.pythonhosted.org/packages/7a/13/e503968fefabd4c6b2650af21e110aa8466fe21432cd7c43a84577a89438/tensorboard_data_server-0.7.2-py3-none-any.whl.metadata
  Using cached tensorboard_data_server-0.7.2-py3-none-any.whl.metadata (1.1 kB)
Collecting werkzeug>=1.0.1 (from tensorboard<2.17,>=2.16->tensorflow)
  Obtaining dependency information for werkzeug>=1.0.1 from https://files.pythonhosted.org/packages/c3/fc/254c3e9b5feb89ff5b9076a23218dafbc99c96ac5941e900b71206e6313b/werkzeug-3.0.1-py3-none-any.whl.metadata
  Using cached werkzeug-3.0.1-py3-none-any.whl.metadata (4.1 kB)
Requirement already satisfied: MarkupSafe>=2.1.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard<2.17,>=2.16->tensorflow) (2.1.5)
Collecting markdown-it-py>=2.2.0 (from rich->keras>=3.0.0->tensorflow)
  Obtaining dependency information for markdown-it-py>=2.2.0 from https://files.pythonhosted.org/packages/42/d7/1ec15b46af6af88f19b8e5ffea08fa375d433c998b8a7639e76935c14f1f/markdown_it_py-3.0.0-py3-none-any.whl.metadata
  Using cached markdown_it_py-3.0.0-py3-none-any.whl.metadata (6.9 kB)
Collecting pygments<3.0.0,>=2.13.0 (from rich->keras>=3.0.0->tensorflow)
  Obtaining dependency information for pygments<3.0.0,>=2.13.0 from https://files.pythonhosted.org/packages/97/9c/372fef8377a6e340b1704768d20daaded98bf13282b5327beb2e2fe2c7ef/pygments-2.17.2-py3-none-any.whl.metadata
  Using cached pygments-2.17.2-py3-none-any.whl.metadata (2.6 kB)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich->keras>=3.0.0->tensorflow)
  Obtaining dependency information for mdurl~=0.1 from https://files.pythonhosted.org/packages/b3/38/89ba8ad64ae25be8de66a6d463314cf1eb366222074cfda9ee839c56a4b4/mdurl-0.1.2-py3-none-any.whl.metadata
  Using cached mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Downloading tensorflow-2.16.1-cp310-cp310-macosx_12_0_arm64.whl (227.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 227.0/227.0 MB 158.3 kB/s eta 0:00:00
Downloading absl_py-2.1.0-py3-none-any.whl (133 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 133.7/133.7 kB 152.4 kB/s eta 0:00:00
Downloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
Downloading flatbuffers-24.3.7-py2.py3-none-any.whl (26 kB)
Using cached gast-0.5.4-py3-none-any.whl (19 kB)
Downloading google_pasta-0.2.0-py3-none-any.whl (57 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.5/57.5 kB 270.4 kB/s eta 0:00:00
Downloading grpcio-1.62.1-cp310-cp310-macosx_12_0_universal2.whl (10.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.0/10.0 MB 158.8 kB/s eta 0:00:00
Downloading h5py-3.10.0-cp310-cp310-macosx_11_0_arm64.whl (2.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.7/2.7 MB 164.1 kB/s eta 0:00:00
Downloading keras-3.1.0-py3-none-any.whl (1.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 156.5 kB/s eta 0:00:00
Downloading libclang-18.1.1-py2.py3-none-macosx_11_0_arm64.whl (26.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 26.4/26.4 MB 21.8 MB/s eta 0:00:00
Downloading ml_dtypes-0.3.2-cp310-cp310-macosx_10_9_universal2.whl (389 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 389.8/389.8 kB 96.9 kB/s eta 0:00:00
Downloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 65.5/65.5 kB 210.1 kB/s eta 0:00:00
Downloading protobuf-4.25.3-cp37-abi3-macosx_10_9_universal2.whl (394 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 394.2/394.2 kB 157.1 kB/s eta 0:00:00
Downloading tensorboard-2.16.2-py3-none-any.whl (5.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.5/5.5 MB 159.4 kB/s eta 0:00:00
Downloading tensorflow_io_gcs_filesystem-0.36.0-cp310-cp310-macosx_12_0_arm64.whl (3.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.4/3.4 MB 162.4 kB/s eta 0:00:00
Downloading termcolor-2.4.0-py3-none-any.whl (7.7 kB)
Downloading wrapt-1.16.0-cp310-cp310-macosx_11_0_arm64.whl (38 kB)
Downloading Markdown-3.6-py3-none-any.whl (105 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 105.4/105.4 kB 166.2 kB/s eta 0:00:00
Downloading tensorboard_data_server-0.7.2-py3-none-any.whl (2.4 kB)
Downloading werkzeug-3.0.1-py3-none-any.whl (226 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 226.7/226.7 kB 175.4 kB/s eta 0:00:00
Downloading namex-0.0.7-py3-none-any.whl (5.8 kB)
Downloading optree-0.10.0-cp310-cp310-macosx_11_0_arm64.whl (248 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 248.9/248.9 kB 168.1 kB/s eta 0:00:00
Downloading rich-13.7.1-py3-none-any.whl (240 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 240.7/240.7 kB 164.5 kB/s eta 0:00:00
Downloading markdown_it_py-3.0.0-py3-none-any.whl (87 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 87.5/87.5 kB 151.9 kB/s eta 0:00:00
Downloading pygments-2.17.2-py3-none-any.whl (1.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 172.5 kB/s eta 0:00:00
Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Installing collected packages: namex, libclang, flatbuffers, wrapt, werkzeug, termcolor, tensorflow-io-gcs-filesystem, tensorboard-data-server, pygments, protobuf, optree, opt-einsum, ml-dtypes, mdurl, markdown, h5py, grpcio, google-pasta, gast, astunparse, absl-py, tensorboard, markdown-it-py, rich, keras, tensorflow
Successfully installed absl-py-2.1.0 astunparse-1.6.3 flatbuffers-24.3.7 gast-0.5.4 google-pasta-0.2.0 grpcio-1.62.1 h5py-3.10.0 keras-3.1.0 libclang-18.1.1 markdown-3.6 markdown-it-py-3.0.0 mdurl-0.1.2 ml-dtypes-0.3.2 namex-0.0.7 opt-einsum-3.3.0 optree-0.10.0 protobuf-4.25.3 pygments-2.17.2 rich-13.7.1 tensorboard-2.16.2 tensorboard-data-server-0.7.2 tensorflow-2.16.1 tensorflow-io-gcs-filesystem-0.36.0 termcolor-2.4.0 werkzeug-3.0.1 wrapt-1.16.0
(base) ~/CS-7643-O01/Group_Project (main ✗)

########################
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/_pywrap_quantize_training.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/_pywrap_sanitizers.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/_pywrap_tfcompile.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/_pywrap_tfe.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/_pywrap_toco_api.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/autograph/impl/testing/pybind_for_testing.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/autograph/utils/type_registry.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/checkpoint/*
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/client/_pywrap_debug_events_writer.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/client/_pywrap_device_lib.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/client/_pywrap_events_writer.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/client/_pywrap_tf_session.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/compiler/xla/experimental/*
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/experimental/ops/distributed_save_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/experimental/ops/from_list.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/experimental/ops/pad_to_cardinality.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/experimental/service/_pywrap_server_lib.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/experimental/service/_pywrap_snapshot_utils.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/experimental/service/_pywrap_snapshot_utils.so
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/experimental/service/_pywrap_utils.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/batch_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/cache_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/choose_from_datasets_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/concatenate_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/counter_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/dataset_autograph.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/debug_mode.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/directed_interleave_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/filter_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/flat_map_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/from_generator_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/from_sparse_tensor_slices_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/from_tensor_slices_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/from_tensors_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/group_by_window_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/ignore_errors_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/interleave_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/iterator_autograph.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/load_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/map_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/padded_batch_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/prefetch_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/ragged_batch_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/random_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/range_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/rebatch_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/repeat_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/sample_from_datasets_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/save_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/scan_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/shard_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/shuffle_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/skip_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/snapshot_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/sparse_batch_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/take_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/take_while_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/test_mode.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/unbatch_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/unique_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/window_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/data/ops/zip_op.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/distribute/coordinator/fault_tolerance_test_base.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/distribute/coordinator/remote_value.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/distribute/experimental/dtensor_strategy_extended.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/distribute/experimental/dtensor_util.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/distribute/experimental/mirrored_strategy.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/distribute/experimental/multi_worker_mirrored_strategy.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/distribute/failure_handling/failure_handling_util.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/distribute/failure_handling/preemption_watcher.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/distribute/load_context.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/eager/polymorphic_function/*
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/eager/record.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/feature_column/feature_column_v2_types.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/flags_pybind.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/_dtypes.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/_op_def_library_pybind.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/_op_def_registry.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/_proto_comparators.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/_python_memory_checker_helper.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/_pywrap_python_api_dispatcher.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/_pywrap_python_op_gen.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/_test_metrics_util.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/byte_swap_tensor.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/constant_tensor_conversion.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/flexible_dtypes.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/lib_native_proto_caster.dylib
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/none_tensor.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/override_binary_operator.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/stack.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/strict_mode.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/tensor.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/tensor_conversion.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/type_spec_registry.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/type_utils.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/framework/weak_tensor.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/grappler/_pywrap_tf_cluster.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/grappler/_pywrap_tf_item.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/grappler/_pywrap_tf_optimizer.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/kernel_tests/nn_ops/depthwise_conv_op_base.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/lib/core/_pywrap_py_func.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/lib/io/_pywrap_file_io.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/lib/io/_pywrap_record_io.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/array_ops_stack.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/autograph_ops.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/cond.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/control_flow_assert.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/control_flow_case.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/control_flow_switch_case.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/gen_optional_ops.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/gen_sync_ops.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/gen_uniform_quant_ops.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/linalg/property_hint_util.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/linalg/slicing.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/lookup_grad.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/nn_fused_batch_norm_grad.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/nn_impl_distribute.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/parsing_grad.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/ragged/ragged_autograph.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/ragged/ragged_bincount_ops.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/ragged/ragged_embedding_ops.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/random_crop_ops.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/random_ops_util.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/ref_variable.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/resource_variables_toggle.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/shape_util.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/structured/structured_tensor_dynamic.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/tensor_getitem_override.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/tensor_math_operator_overrides.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/variable_v1.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/weak_tensor_ops.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/weak_tensor_test_util.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/ops/while_loop.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/platform/_pywrap_cpu_feature_guard.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/platform/_pywrap_cpu_feature_guard.so
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/platform/_pywrap_stacktrace_handler.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/platform/_pywrap_tf2.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/profiler/internal/_pywrap_profiler.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/profiler/internal/_pywrap_traceme.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/proto_exports.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/saved_model/fingerprinting.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/saved_model/fingerprinting_utils.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/saved_model/path_helpers.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/saved_model/pywrap_saved_model/__init__.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/saved_model/pywrap_saved_model/constants.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/saved_model/pywrap_saved_model/fingerprinting.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/saved_model/pywrap_saved_model/merger.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/saved_model/pywrap_saved_model/metrics.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/saved_model/tracing_utils.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/summary/tb_summary.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/tools/api/generator2/*
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/tpu/_pywrap_tpu_embedding.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/tpu/_pywrap_tpu_embedding.so
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/tpu/ops/gen_xla_ops.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/tpu/tpu_embedding_v3.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/tpu/tpu_embedding_v3_utils.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/tpu/tpu_replication.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/trackable/*
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/training/saving/trace_saveable_util.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/types/data.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/_pywrap_checkpoint_reader.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/_pywrap_determinism.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/_pywrap_kernel_registry.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/_pywrap_nest.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/_pywrap_stat_summarizer.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/_pywrap_tensor_float_32_execution.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/_pywrap_tfprof.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/_pywrap_transform_graph.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/_pywrap_util_port.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/_pywrap_utils.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/_tf_stack.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/custom_nest_protocol.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/deprecated_module.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/deprecated_module_new.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/fast_module_type.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/nest_util.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/pywrap_xla_ops.pyi
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/pywrap_xla_ops.so
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/tf_decorator_export.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/python/util/variable_utils.py
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/security/*
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/tools/pip_package/v2/*
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/tsl/*
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/tsl/framework/contraction/eigen_contraction_kernel.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/tsl/platform/ctstring.h
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/tsl/platform/ctstring_internal.h
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/tsl/platform/default/env_time.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/tsl/platform/default/integral_types.h
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/tsl/platform/dynamic_annotations.h
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/tsl/platform/env_time.h
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/tsl/platform/macros.h
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/tsl/platform/platform.h
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/cpu_function_runtime.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/executable_run_options.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_conv2d.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_conv3d.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_custom_call_status.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_fft.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_fork_join.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_fp16.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_key_value_sort.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_matmul_c128.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_matmul_c64.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_matmul_common.h
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_matmul_f16.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_matmul_f32.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_matmul_f64.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_matmul_s32.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_pow.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_single_threaded_conv2d.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_single_threaded_conv3d.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_single_threaded_fft.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_single_threaded_matmul_c128.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_single_threaded_matmul_c64.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_single_threaded_matmul_common.h
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_single_threaded_matmul_f16.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_single_threaded_matmul_f32.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_single_threaded_matmul_f64.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_single_threaded_matmul_s32.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/cpu/runtime_topk.cc
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow/xla_aot_runtime_src/xla/service/custom_call_status.cc
Proceed (Y/n)? y
  Successfully uninstalled tensorflow-2.16.1
Found existing installation: tensorflow-macos 2.16.1
Uninstalling tensorflow-macos-2.16.1:
  Would remove:
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_macos-2.16.1.dist-info/*
Proceed (Y/n)? y
  Successfully uninstalled tensorflow-macos-2.16.1
Found existing installation: tensorflow-metal 1.1.0
Uninstalling tensorflow-metal-1.1.0:
  Would remove:
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow-plugins/*
    /Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_metal-1.1.0.dist-info/*
Proceed (Y/n)? y
  Successfully uninstalled tensorflow-metal-1.1.0
Collecting tensorflow-macos
  Using cached tensorflow_macos-2.16.1-cp310-cp310-macosx_12_0_arm64.whl.metadata (3.5 kB)
Collecting tensorflow-metal
  Using cached tensorflow_metal-1.1.0-cp310-cp310-macosx_12_0_arm64.whl.metadata (1.2 kB)
Collecting tensorflow==2.16.1 (from tensorflow-macos)
  Using cached tensorflow-2.16.1-cp310-cp310-macosx_12_0_arm64.whl.metadata (4.1 kB)
Requirement already satisfied: absl-py>=1.0.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (1.4.0)
Requirement already satisfied: astunparse>=1.6.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (1.6.3)
Requirement already satisfied: flatbuffers>=23.5.26 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (24.3.25)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (0.4.0)
Requirement already satisfied: google-pasta>=0.1.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (0.2.0)
Requirement already satisfied: h5py>=3.10.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (3.10.0)
Requirement already satisfied: libclang>=13.0.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (18.1.1)
Requirement already satisfied: ml-dtypes~=0.3.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (0.3.2)
Requirement already satisfied: opt-einsum>=2.3.2 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (3.3.0)
Requirement already satisfied: packaging in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (24.0)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (4.25.3)
Requirement already satisfied: requests<3,>=2.21.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (2.31.0)
Requirement already satisfied: setuptools in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (69.2.0)
Requirement already satisfied: six>=1.12.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (2.4.0)
Requirement already satisfied: typing-extensions>=3.6.6 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (4.11.0)
Requirement already satisfied: wrapt>=1.11.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (1.16.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (1.62.1)
Requirement already satisfied: tensorboard<2.17,>=2.16 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (2.16.2)
Requirement already satisfied: keras>=3.0.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (3.1.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (0.36.0)
Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (1.26.4)
Requirement already satisfied: wheel~=0.35 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorflow-metal) (0.43.0)
Requirement already satisfied: rich in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos) (13.7.1)
Requirement already satisfied: namex in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos) (0.0.7)
Requirement already satisfied: optree in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos) (0.10.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow==2.16.1->tensorflow-macos) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow==2.16.1->tensorflow-macos) (3.6)
Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow==2.16.1->tensorflow-macos) (2.2.1)
Requirement already satisfied: certifi>=2017.4.17 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow==2.16.1->tensorflow-macos) (2024.2.2)
Requirement already satisfied: markdown>=2.6.8 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorboard<2.17,>=2.16->tensorflow==2.16.1->tensorflow-macos) (3.6)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorboard<2.17,>=2.16->tensorflow==2.16.1->tensorflow-macos) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from tensorboard<2.17,>=2.16->tensorflow==2.16.1->tensorflow-macos) (3.0.2)
Requirement already satisfied: MarkupSafe>=2.1.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard<2.17,>=2.16->tensorflow==2.16.1->tensorflow-macos) (2.1.5)
Requirement already satisfied: markdown-it-py>=2.2.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from rich->keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from rich->keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos) (2.17.2)
Requirement already satisfied: mdurl~=0.1 in /Users/deangladish/miniforge3/lib/python3.10/site-packages (from markdown-it-py>=2.2.0->rich->keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos) (0.1.2)
Using cached tensorflow_macos-2.16.1-cp310-cp310-macosx_12_0_arm64.whl (2.2 kB)
Using cached tensorflow-2.16.1-cp310-cp310-macosx_12_0_arm64.whl (227.0 MB)
Using cached tensorflow_metal-1.1.0-cp310-cp310-macosx_12_0_arm64.whl (1.4 MB)
Installing collected packages: tensorflow-metal, tensorflow, tensorflow-macos
Successfully installed tensorflow-2.16.1 tensorflow-macos-2.16.1 tensorflow-metal-1.1.0
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Convolutional_Neural_Network.py", line 112, in <module>
    from tensorflow_model_optimization.sparsity import keras as sparsity
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/__init__.py", line 86, in <module>
    from tensorflow_model_optimization.python.core.api import clustering
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/__init__.py", line 16, in <module>
    from tensorflow_model_optimization.python.core.api import clustering
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/clustering/__init__.py", line 16, in <module>
    from tensorflow_model_optimization.python.core.api.clustering import keras
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/clustering/keras/__init__.py", line 19, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras.cluster import cluster_scope
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/cluster.py", line 22, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras import cluster_wrapper
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/cluster_wrapper.py", line 23, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/clustering_centroids.py", line 22, in <module>
    from tensorflow_model_optimization.python.core.keras.compat import keras
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/keras/compat.py", line 41, in <module>
    keras = _get_keras_instance()
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/keras/compat.py", line 35, in _get_keras_instance
    import tf_keras as keras_internal  # pylint: disable=g-import-not-at-top,unused-import
ModuleNotFoundError: No module named 'tf_keras'
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) pip install --upgrade tensorflow tensorflow-model-optimization

Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: tensorflow in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (2.16.1)
Requirement already satisfied: tensorflow-model-optimization in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (0.8.0)
Requirement already satisfied: absl-py>=1.0.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (1.4.0)
Requirement already satisfied: astunparse>=1.6.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (1.6.3)
Requirement already satisfied: flatbuffers>=23.5.26 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (24.3.25)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (0.5.4)
Requirement already satisfied: google-pasta>=0.1.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (0.2.0)
Requirement already satisfied: h5py>=3.10.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (3.10.0)
Requirement already satisfied: libclang>=13.0.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (18.1.1)
Requirement already satisfied: ml-dtypes~=0.3.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (0.3.2)
Requirement already satisfied: opt-einsum>=2.3.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (3.3.0)
Requirement already satisfied: packaging in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow) (21.3)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (4.25.3)
Requirement already satisfied: requests<3,>=2.21.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow) (2.28.1)
Requirement already satisfied: setuptools in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow) (63.2.0)
Requirement already satisfied: six>=1.12.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow) (2.0.1)
Requirement already satisfied: typing-extensions>=3.6.6 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow) (4.5.0)
Requirement already satisfied: wrapt>=1.11.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (1.16.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (1.62.1)
Requirement already satisfied: tensorboard<2.17,>=2.16 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (2.16.2)
Requirement already satisfied: keras>=3.0.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (3.1.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (0.36.0)
Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (1.26.4)
Requirement already satisfied: dm-tree~=0.1.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (0.1.8)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from astunparse>=1.6.0->tensorflow) (0.37.1)
Requirement already satisfied: rich in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from keras>=3.0.0->tensorflow) (13.7.1)
Requirement already satisfied: namex in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from keras>=3.0.0->tensorflow) (0.0.7)
Requirement already satisfied: optree in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from keras>=3.0.0->tensorflow) (0.11.0)
Requirement already satisfied: charset-normalizer<3,>=2 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (2.1.1)
Requirement already satisfied: idna<4,>=2.5 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (3.4)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (1.26.12)
Requirement already satisfied: certifi>=2017.4.17 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (2022.9.24)
Requirement already satisfied: markdown>=2.6.8 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorboard<2.17,>=2.16->tensorflow) (3.6)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorboard<2.17,>=2.16->tensorflow) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorboard<2.17,>=2.16->tensorflow) (3.0.2)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from packaging->tensorflow) (3.1.1)
Requirement already satisfied: MarkupSafe>=2.1.1 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard<2.17,>=2.16->tensorflow) (2.1.1)
Requirement already satisfied: markdown-it-py>=2.2.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from rich->keras>=3.0.0->tensorflow) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from rich->keras>=3.0.0->tensorflow) (2.13.0)
Requirement already satisfied: mdurl~=0.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from markdown-it-py>=2.2.0->rich->keras>=3.0.0->tensorflow) (0.1.2)
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Convolutional_Neural_Network.py", line 114, in <module>
    from tensorflow_model_optimization.sparsity import keras as sparsity
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/__init__.py", line 86, in <module>
    from tensorflow_model_optimization.python.core.api import clustering
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/__init__.py", line 16, in <module>
    from tensorflow_model_optimization.python.core.api import clustering
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/clustering/__init__.py", line 16, in <module>
    from tensorflow_model_optimization.python.core.api.clustering import keras
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/clustering/keras/__init__.py", line 19, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras.cluster import cluster_scope
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/cluster.py", line 22, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras import cluster_wrapper
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/cluster_wrapper.py", line 23, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/clustering_centroids.py", line 22, in <module>
    from tensorflow_model_optimization.python.core.keras.compat import keras
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/keras/compat.py", line 41, in <module>
    keras = _get_keras_instance()
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/keras/compat.py", line 35, in _get_keras_instance
    import tf_keras as keras_internal  # pylint: disable=g-import-not-at-top,unused-import
ModuleNotFoundError: No module named 'tf_keras'
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) pip install --upgrade tensorflow
pip install --upgrade tensorflow-model-optimization

Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: tensorflow in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (2.16.1)
Requirement already satisfied: absl-py>=1.0.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (1.4.0)
Requirement already satisfied: astunparse>=1.6.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (1.6.3)
Requirement already satisfied: flatbuffers>=23.5.26 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (24.3.25)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (0.5.4)
Requirement already satisfied: google-pasta>=0.1.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (0.2.0)
Requirement already satisfied: h5py>=3.10.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (3.10.0)
Requirement already satisfied: libclang>=13.0.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (18.1.1)
Requirement already satisfied: ml-dtypes~=0.3.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (0.3.2)
Requirement already satisfied: opt-einsum>=2.3.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (3.3.0)
Requirement already satisfied: packaging in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow) (21.3)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (4.25.3)
Requirement already satisfied: requests<3,>=2.21.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow) (2.28.1)
Requirement already satisfied: setuptools in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow) (63.2.0)
Requirement already satisfied: six>=1.12.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow) (2.0.1)
Requirement already satisfied: typing-extensions>=3.6.6 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow) (4.5.0)
Requirement already satisfied: wrapt>=1.11.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (1.16.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (1.62.1)
Requirement already satisfied: tensorboard<2.17,>=2.16 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (2.16.2)
Requirement already satisfied: keras>=3.0.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (3.1.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (0.36.0)
Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow) (1.26.4)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from astunparse>=1.6.0->tensorflow) (0.37.1)
Requirement already satisfied: rich in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from keras>=3.0.0->tensorflow) (13.7.1)
Requirement already satisfied: namex in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from keras>=3.0.0->tensorflow) (0.0.7)
Requirement already satisfied: optree in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from keras>=3.0.0->tensorflow) (0.11.0)
Requirement already satisfied: charset-normalizer<3,>=2 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (2.1.1)
Requirement already satisfied: idna<4,>=2.5 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (3.4)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (1.26.12)
Requirement already satisfied: certifi>=2017.4.17 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (2022.9.24)
Requirement already satisfied: markdown>=2.6.8 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorboard<2.17,>=2.16->tensorflow) (3.6)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorboard<2.17,>=2.16->tensorflow) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorboard<2.17,>=2.16->tensorflow) (3.0.2)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from packaging->tensorflow) (3.1.1)
Requirement already satisfied: MarkupSafe>=2.1.1 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard<2.17,>=2.16->tensorflow) (2.1.1)
Requirement already satisfied: markdown-it-py>=2.2.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from rich->keras>=3.0.0->tensorflow) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from rich->keras>=3.0.0->tensorflow) (2.13.0)
Requirement already satisfied: mdurl~=0.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from markdown-it-py>=2.2.0->rich->keras>=3.0.0->tensorflow) (0.1.2)
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: tensorflow-model-optimization in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (0.8.0)
Requirement already satisfied: absl-py~=1.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.4.0)
Requirement already satisfied: dm-tree~=0.1.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (0.1.8)
Requirement already satisfied: numpy~=1.23 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.26.4)
Requirement already satisfied: six~=1.14 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.16.0)
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Convolutional_Neural_Network.py", line 114, in <module>
    from tensorflow_model_optimization.sparsity import keras as sparsity
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/__init__.py", line 86, in <module>
    from tensorflow_model_optimization.python.core.api import clustering
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/__init__.py", line 16, in <module>
    from tensorflow_model_optimization.python.core.api import clustering
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/clustering/__init__.py", line 16, in <module>
    from tensorflow_model_optimization.python.core.api.clustering import keras
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/clustering/keras/__init__.py", line 19, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras.cluster import cluster_scope
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/cluster.py", line 22, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras import cluster_wrapper
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/cluster_wrapper.py", line 23, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/clustering_centroids.py", line 22, in <module>
    from tensorflow_model_optimization.python.core.keras.compat import keras
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/keras/compat.py", line 41, in <module>
    keras = _get_keras_instance()
  File "/Users/deangladish/miniforge3/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/keras/compat.py", line 35, in _get_keras_instance
    import tf_keras as keras_internal  # pylint: disable=g-import-not-at-top,unused-import
ModuleNotFoundError: No module named 'tf_keras'
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) pip uninstall tensorflow tensorflow-model-optimization

Found existing installation: tensorflow 2.16.1
Uninstalling tensorflow-2.16.1:
  Would remove:
    /Users/deangladish/Library/Python/3.10/bin/import_pb_to_tensorboard
    /Users/deangladish/Library/Python/3.10/bin/saved_model_cli
    /Users/deangladish/Library/Python/3.10/bin/tensorboard
    /Users/deangladish/Library/Python/3.10/bin/tf_upgrade_v2
    /Users/deangladish/Library/Python/3.10/bin/tflite_convert
    /Users/deangladish/Library/Python/3.10/bin/toco
    /Users/deangladish/Library/Python/3.10/bin/toco_from_protos
    /Users/deangladish/Library/Python/3.10/lib/python/site-packages/tensorflow-2.16.1.dist-info/*
    /Users/deangladish/Library/Python/3.10/lib/python/site-packages/tensorflow/*
Proceed (Y/n)? y
  Successfully uninstalled tensorflow-2.16.1
Found existing installation: tensorflow-model-optimization 0.8.0
Uninstalling tensorflow-model-optimization-0.8.0:
  Would remove:
    /Users/deangladish/Library/Python/3.10/lib/python/site-packages/tensorflow_model_optimization-0.8.0.dist-info/*
    /Users/deangladish/Library/Python/3.10/lib/python/site-packages/tensorflow_model_optimization/*
Proceed (Y/n)? y
  Successfully uninstalled tensorflow-model-optimization-0.8.0
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) brew uninstall python

Warning: Cask python was renamed to homebrew/core/python.
Error: No such keg: /usr/local/Cellar/python
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) brew update

==> Updating Homebrew...
fatal: couldn't find remote ref refs/heads/master
Error: Fetching /usr/local/Homebrew/Library/Taps/heroku/homebrew-brew failed!
==> Updated Homebrew from 4.2.15 (4fa7264a52) to 4.2.16 (0476c2e5e4).
Updated 4 taps (homebrew/cask-versions, homebrew/bundle, homebrew/core and homebrew/cask).
==> New Formulae
dissent               jtbl                  manim                 msieve                overarch              rage                  valkey
gitu                  libscfg               mantra                navidrome             policy_sentry         redict                vfox
==> New Casks
arctic                         clearvpn                       halloy                         phoenix-code                   starnet2
capcut                         fujifilm-x-raw-studio          paragon-extfs11                requestly                      viable
==> Outdated Formulae
autoconf         gettext          gmp              isl              libidn2          libunistring     openssl@1.1      readline         sqlite
ca-certificates  gh               heroku           jenv             libmpc           mpdecimal        openssl@3        ruby             wget
gcc              git-lfs          heroku-node      libffi           libomp           mpfr             python@3.7       ruby-install     xz
==> Outdated Casks
chatgpt                                                                       java6

You have 27 outdated formulae and 2 outdated casks installed.
You can upgrade them with brew upgrade
or list them with brew outdated.
Error: Some taps failed to update!
The following taps can not read their remote branches:
  heroku/brew
This is happening because the remote branch was renamed or deleted.
Reset taps to point to the correct remote branches by running `brew tap --repair`

The 4.2.16 changelog can be found at:
  https://github.com/Homebrew/brew/releases/tag/4.2.16
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) brew list | grep python

python@3.7
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) brew uninstall python@3.7
Uninstalling /usr/local/Cellar/python@3.7/3.7.14... (4,833 files, 76.7MB)
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) brew install python

==> Downloading https://ghcr.io/v2/homebrew/core/python/3.12/manifests/3.12.2_1-1
################################################################################################################################################### 100.0%
==> Fetching dependencies for python@3.12: mpdecimal, ca-certificates, openssl@3, readline, sqlite and xz
==> Downloading https://ghcr.io/v2/homebrew/core/mpdecimal/manifests/4.0.0-1
################################################################################################################################################### 100.0%
==> Fetching mpdecimal
==> Downloading https://ghcr.io/v2/homebrew/core/mpdecimal/blobs/sha256:bb1729bd410275aab1bd276f99fb22678b6ad53de2c9c474fdda854ed0ebaebd
################################################################################################################################################### 100.0%
==> Downloading https://ghcr.io/v2/homebrew/core/ca-certificates/manifests/2024-03-11
################################################################################################################################################### 100.0%
==> Fetching ca-certificates
==> Downloading https://ghcr.io/v2/homebrew/core/ca-certificates/blobs/sha256:cab828953672906e00a8f25db751977b8dc4115f021f8dfe82b644ade03dacdb
################################################################################################################################################### 100.0%
==> Downloading https://ghcr.io/v2/homebrew/core/openssl/3/manifests/3.2.1-1
################################################################################################################################################### 100.0%
==> Fetching openssl@3
==> Downloading https://ghcr.io/v2/homebrew/core/openssl/3/blobs/sha256:f3cd46e866f40f134ee02ca264456e69730c721f577af6bd6927fdb90f2807e0
################################################################################################################################################### 100.0%
==> Downloading https://ghcr.io/v2/homebrew/core/readline/manifests/8.2.10
################################################################################################################################################### 100.0%
==> Fetching readline
==> Downloading https://ghcr.io/v2/homebrew/core/readline/blobs/sha256:952e2975dffc98bd35673c86474dbb91fadc8d993c0720e4f085597f7a484af9
################################################################################################################################################### 100.0%
==> Downloading https://ghcr.io/v2/homebrew/core/sqlite/manifests/3.45.2
################################################################################################################################################### 100.0%
==> Fetching sqlite
==> Downloading https://ghcr.io/v2/homebrew/core/sqlite/blobs/sha256:b528fc258961192ff8fd5abd9b2ea18d2b24508b41aee7c5647c913ff93ff599
################################################################################################################################################### 100.0%
==> Downloading https://ghcr.io/v2/homebrew/core/xz/manifests/5.4.6
################################################################################################################################################### 100.0%
==> Fetching xz
==> Downloading https://ghcr.io/v2/homebrew/core/xz/blobs/sha256:8a3f7325f367f90a22f3c17c0bcc65af615de713a8598e973691e84f118b325c
################################################################################################################################################### 100.0%
==> Fetching python@3.12
==> Downloading https://ghcr.io/v2/homebrew/core/python/3.12/blobs/sha256:c3ba89dca54a1af743dbecc096e384ae8f192e7015dfba39ceb54bbfcac65273
################################################################################################################################################### 100.0%
==> Installing dependencies for python@3.12: mpdecimal, ca-certificates, openssl@3, readline, sqlite and xz
==> Installing python@3.12 dependency: mpdecimal
==> Downloading https://ghcr.io/v2/homebrew/core/mpdecimal/manifests/4.0.0-1
Already downloaded: /Users/deangladish/Library/Caches/Homebrew/downloads/7b63c3b34bee402290af49fac829a6682ab45ea5c9258b6fe03b590a03a4c4a9--mpdecimal-4.0.0-1.bottle_manifest.json
==> Pouring mpdecimal--4.0.0.ventura.bottle.1.tar.gz
🍺  /usr/local/Cellar/mpdecimal/4.0.0: 21 files, 612.7KB
==> Installing python@3.12 dependency: ca-certificates
==> Downloading https://ghcr.io/v2/homebrew/core/ca-certificates/manifests/2024-03-11
Already downloaded: /Users/deangladish/Library/Caches/Homebrew/downloads/c431e0186df2ccc2ea942b34a3c26c2cebebec8e07ad6abdae48447a52c5f506--ca-certificates-2024-03-11.bottle_manifest.json
==> Pouring ca-certificates--2024-03-11.all.bottle.tar.gz
==> Regenerating CA certificate bundle from keychain, this may take a while...
🍺  /usr/local/Cellar/ca-certificates/2024-03-11: 3 files, 229.7KB
==> Installing python@3.12 dependency: openssl@3
==> Downloading https://ghcr.io/v2/homebrew/core/openssl/3/manifests/3.2.1-1
Already downloaded: /Users/deangladish/Library/Caches/Homebrew/downloads/f7b6e249843882452d784a8cbc4e19231186230b9e485a2a284d5c1952a95ec2--openssl@3-3.2.1-1.bottle_manifest.json
==> Pouring openssl@3--3.2.1.ventura.bottle.1.tar.gz
🍺  /usr/local/Cellar/openssl@3/3.2.1: 6,874 files, 32.5MB
==> Installing python@3.12 dependency: readline
==> Downloading https://ghcr.io/v2/homebrew/core/readline/manifests/8.2.10
Already downloaded: /Users/deangladish/Library/Caches/Homebrew/downloads/4ddd52803319828799f1932d4c7fa8d11c667049b20a56341c0c19246a1be93b--readline-8.2.10.bottle_manifest.json
==> Pouring readline--8.2.10.ventura.bottle.tar.gz
🍺  /usr/local/Cellar/readline/8.2.10: 50 files, 1.7MB
==> Installing python@3.12 dependency: sqlite
==> Downloading https://ghcr.io/v2/homebrew/core/sqlite/manifests/3.45.2
Already downloaded: /Users/deangladish/Library/Caches/Homebrew/downloads/52aeccef7dfe87a5156de420a9e1f4b5b62f61b6c2b57633a5e6f04518b50edf--sqlite-3.45.2.bottle_manifest.json
==> Pouring sqlite--3.45.2.ventura.bottle.tar.gz
🍺  /usr/local/Cellar/sqlite/3.45.2: 11 files, 4.7MB
==> Installing python@3.12 dependency: xz
==> Downloading https://ghcr.io/v2/homebrew/core/xz/manifests/5.4.6
Already downloaded: /Users/deangladish/Library/Caches/Homebrew/downloads/b2cc4077807c100af6e0253f51d186f187ff55165638cbe3a4aa16d1c4762660--xz-5.4.6.bottle_manifest.json
==> Pouring xz--5.4.6.ventura.bottle.tar.gz
🍺  /usr/local/Cellar/xz/5.4.6: 163 files, 2.6MB
==> Installing python@3.12
==> Pouring python@3.12--3.12.2_1.ventura.bottle.1.tar.gz
Error: The `brew link` step did not complete successfully
The formula built, but is not symlinked into /usr/local
Could not symlink bin/2to3
Target /usr/local/bin/2to3
already exists. You may want to remove it:
  rm '/usr/local/bin/2to3'

To force the link and overwrite all conflicting files:
  brew link --overwrite python@3.12

To list all files that would be deleted:
  brew link --overwrite python@3.12 --dry-run

Possible conflicting files are:
/usr/local/bin/2to3 -> /Library/Frameworks/Python.framework/Versions/3.10/bin/2to3
/usr/local/bin/idle3 -> /Library/Frameworks/Python.framework/Versions/3.10/bin/idle3
/usr/local/bin/pydoc3 -> /Library/Frameworks/Python.framework/Versions/3.10/bin/pydoc3
/usr/local/bin/python3 -> /Library/Frameworks/Python.framework/Versions/3.10/bin/python3
/usr/local/bin/python3-config -> /Library/Frameworks/Python.framework/Versions/3.10/bin/python3-config
==> /usr/local/Cellar/python@3.12/3.12.2_1/bin/python3.12 -Im ensurepip
==> /usr/local/Cellar/python@3.12/3.12.2_1/bin/python3.12 -Im pip install -v --no-index --upgrade --isolated --target=/usr/local/lib/python3.12/site-packa
==> Caveats
Python has been installed as
  /usr/local/bin/python3

Unversioned symlinks `python`, `python-config`, `pip` etc. pointing to
`python3`, `python3-config`, `pip3` etc., respectively, have been installed into
  /usr/local/opt/python@3.12/libexec/bin

See: https://docs.brew.sh/Homebrew-and-Python
==> Summary
🍺  /usr/local/Cellar/python@3.12/3.12.2_1: 3,237 files, 63.6MB
==> Running `brew cleanup python@3.12`...
Disable this behaviour by setting HOMEBREW_NO_INSTALL_CLEANUP.
Hide these hints with HOMEBREW_NO_ENV_HINTS (see `man brew`).
==> Upgrading 4 dependents of upgraded formulae:
Disable this behaviour by setting HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK.
Hide these hints with HOMEBREW_NO_ENV_HINTS (see `man brew`).
openssl@1.1 1.1.1q -> 1.1.1w, ruby 3.1.2_1 -> 3.3.0, ruby-install 0.8.5 -> 0.9.3, wget 1.21.4 -> 1.24.5
Warning: openssl@1.1 has been deprecated because it is not supported upstream!
==> Downloading https://ghcr.io/v2/homebrew/core/openssl/1.1/manifests/1.1.1w
Already downloaded: /Users/deangladish/Library/Caches/Homebrew/downloads/c7860697af60391a780aafe02430f03bf392038b0688e0f8ab4a50dda098e042--openssl@1.1-1.1.1w.bottle_manifest.json
==> Downloading https://ghcr.io/v2/homebrew/core/ruby/manifests/3.3.0
Already downloaded: /Users/deangladish/Library/Caches/Homebrew/downloads/ad5286c3745392cf1071b19f938bf3343140a0cf750450ac22a1d613c23f4539--ruby-3.3.0.bottle_manifest.json
==> Downloading https://ghcr.io/v2/homebrew/core/ruby-install/manifests/0.9.3
################################################################################################################################################### 100.0%
==> Downloading https://ghcr.io/v2/homebrew/core/wget/manifests/1.24.5
################################################################################################################################################### 100.0%
==> Fetching dependencies for wget: libunistring, gettext and libidn2
==> Downloading https://ghcr.io/v2/homebrew/core/libunistring/manifests/1.2
################################################################################################################################################### 100.0%
==> Fetching libunistring
==> Downloading https://ghcr.io/v2/homebrew/core/libunistring/blobs/sha256:66091a34396e4e17fc78f31410bf5878091ee6887cec79995f3598093ee481ea
################################################################################################################################################### 100.0%
==> Downloading https://ghcr.io/v2/homebrew/core/gettext/manifests/0.22.5
################################################################################################################################################### 100.0%
==> Fetching gettext
==> Downloading https://ghcr.io/v2/homebrew/core/gettext/blobs/sha256:1a35820de97aa8d93019d64f7add5443bcf1c14f05bd249e670e7ca0f0fc6b2a
################################################################################################################################################### 100.0%
==> Downloading https://ghcr.io/v2/homebrew/core/libidn2/manifests/2.3.7
################################################################################################################################################### 100.0%
==> Fetching libidn2
==> Downloading https://ghcr.io/v2/homebrew/core/libidn2/blobs/sha256:4b4f5eadc82273fb3b0d384466dab53d9fdc7200cbfae1eb5b5bebfe359f4f1e
################################################################################################################################################### 100.0%
==> Fetching wget
==> Downloading https://ghcr.io/v2/homebrew/core/wget/blobs/sha256:1b7e2f76c90553543a5e25dadf031c6fcfe280f52bf27d89e04006f9d33fd20b
################################################################################################################################################### 100.0%
==> Upgrading wget
  1.21.4 -> 1.24.5
==> Installing dependencies for wget: libunistring, gettext and libidn2
==> Installing wget dependency: libunistring
==> Downloading https://ghcr.io/v2/homebrew/core/libunistring/manifests/1.2
Already downloaded: /Users/deangladish/Library/Caches/Homebrew/downloads/48ac60445a77a63996cf15f6414f68a620d544fb683031b14eb3aea95c3064f6--libunistring-1.2.bottle_manifest.json
==> Pouring libunistring--1.2.ventura.bottle.tar.gz
🍺  /usr/local/Cellar/libunistring/1.2: 59 files, 5.1MB
==> Installing wget dependency: gettext
==> Downloading https://ghcr.io/v2/homebrew/core/gettext/manifests/0.22.5
Already downloaded: /Users/deangladish/Library/Caches/Homebrew/downloads/447e45b77bb47ede0377f7eab1863825298ecaaaeed0bbd84aca3bd300b00508--gettext-0.22.5.bottle_manifest.json
==> Pouring gettext--0.22.5.ventura.bottle.tar.gz
🍺  /usr/local/Cellar/gettext/0.22.5: 2,043 files, 23.8MB
==> Installing wget dependency: libidn2
==> Downloading https://ghcr.io/v2/homebrew/core/libidn2/manifests/2.3.7
Already downloaded: /Users/deangladish/Library/Caches/Homebrew/downloads/45d1d4d2930c4782bf53e761a1c0166cd8a40f4193ac8c44e86f0b6708e80354--libidn2-2.3.7.bottle_manifest.json
==> Pouring libidn2--2.3.7.ventura.bottle.tar.gz
🍺  /usr/local/Cellar/libidn2/2.3.7: 80 files, 1MB
==> Installing wget
==> Pouring wget--1.24.5.ventura.bottle.tar.gz
🍺  /usr/local/Cellar/wget/1.24.5: 91 files, 4.5MB
==> Running `brew cleanup wget`...
Removing: /usr/local/Cellar/wget/1.21.4... (91 files, 4.4MB)
Removing: /Users/deangladish/Library/Caches/Homebrew/wget_bottle_manifest--1.21.4... (13.3KB)
==> Checking for dependents of upgraded formulae...
==> No broken dependents found!
==> Caveats
==> python@3.12
Python has been installed as
  /usr/local/bin/python3

Unversioned symlinks `python`, `python-config`, `pip` etc. pointing to
`python3`, `python3-config`, `pip3` etc., respectively, have been installed into
  /usr/local/opt/python@3.12/libexec/bin

See: https://docs.brew.sh/Homebrew-and-Python
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python3 --version

Python 3.10.6
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python3 -m venv myprojectenv

(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) source myprojectenv/bin/activate

(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) pip install --upgrade pip
pip install tensorflow-macos
pip install tensorflow-metal

Requirement already satisfied: pip in ./myprojectenv/lib/python3.10/site-packages (22.2.1)
Collecting pip
  Using cached pip-24.0-py3-none-any.whl (2.1 MB)
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 22.2.1
    Uninstalling pip-22.2.1:
      Successfully uninstalled pip-22.2.1
Successfully installed pip-24.0
Collecting tensorflow-macos
  Using cached tensorflow_macos-2.16.1-cp310-cp310-macosx_12_0_arm64.whl.metadata (3.5 kB)
Collecting tensorflow==2.16.1 (from tensorflow-macos)
  Using cached tensorflow-2.16.1-cp310-cp310-macosx_12_0_arm64.whl.metadata (4.1 kB)
Collecting absl-py>=1.0.0 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached absl_py-2.1.0-py3-none-any.whl.metadata (2.3 kB)
Collecting astunparse>=1.6.0 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)
Collecting flatbuffers>=23.5.26 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached flatbuffers-24.3.25-py2.py3-none-any.whl.metadata (850 bytes)
Collecting gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached gast-0.5.4-py3-none-any.whl.metadata (1.3 kB)
Collecting google-pasta>=0.1.1 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)
Collecting h5py>=3.10.0 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached h5py-3.10.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (2.5 kB)
Collecting libclang>=13.0.0 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached libclang-18.1.1-py2.py3-none-macosx_11_0_arm64.whl.metadata (5.2 kB)
Collecting ml-dtypes~=0.3.1 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached ml_dtypes-0.3.2-cp310-cp310-macosx_10_9_universal2.whl.metadata (20 kB)
Collecting opt-einsum>=2.3.2 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached opt_einsum-3.3.0-py3-none-any.whl.metadata (6.5 kB)
Collecting packaging (from tensorflow==2.16.1->tensorflow-macos)
  Using cached packaging-24.0-py3-none-any.whl.metadata (3.2 kB)
Collecting protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached protobuf-4.25.3-cp37-abi3-macosx_10_9_universal2.whl.metadata (541 bytes)
Collecting requests<3,>=2.21.0 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached requests-2.31.0-py3-none-any.whl.metadata (4.6 kB)
Requirement already satisfied: setuptools in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (63.2.0)
Collecting six>=1.12.0 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached six-1.16.0-py2.py3-none-any.whl.metadata (1.8 kB)
Collecting termcolor>=1.1.0 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached termcolor-2.4.0-py3-none-any.whl.metadata (6.1 kB)
Collecting typing-extensions>=3.6.6 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached typing_extensions-4.11.0-py3-none-any.whl.metadata (3.0 kB)
Collecting wrapt>=1.11.0 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached wrapt-1.16.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (6.6 kB)
Collecting grpcio<2.0,>=1.24.3 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached grpcio-1.62.1-cp310-cp310-macosx_12_0_universal2.whl.metadata (4.0 kB)
Collecting tensorboard<2.17,>=2.16 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached tensorboard-2.16.2-py3-none-any.whl.metadata (1.6 kB)
Collecting keras>=3.0.0 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached keras-3.1.1-py3-none-any.whl.metadata (5.6 kB)
Collecting tensorflow-io-gcs-filesystem>=0.23.1 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached tensorflow_io_gcs_filesystem-0.36.0-cp310-cp310-macosx_12_0_arm64.whl.metadata (14 kB)
Collecting numpy<2.0.0,>=1.23.5 (from tensorflow==2.16.1->tensorflow-macos)
  Using cached numpy-1.26.4-cp310-cp310-macosx_11_0_arm64.whl.metadata (61 kB)
Collecting wheel<1.0,>=0.23.0 (from astunparse>=1.6.0->tensorflow==2.16.1->tensorflow-macos)
  Using cached wheel-0.43.0-py3-none-any.whl.metadata (2.2 kB)
Collecting rich (from keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos)
  Using cached rich-13.7.1-py3-none-any.whl.metadata (18 kB)
Collecting namex (from keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos)
  Using cached namex-0.0.7-py3-none-any.whl.metadata (246 bytes)
Collecting optree (from keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos)
  Using cached optree-0.11.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (45 kB)
Collecting charset-normalizer<4,>=2 (from requests<3,>=2.21.0->tensorflow==2.16.1->tensorflow-macos)
  Using cached charset_normalizer-3.3.2-cp310-cp310-macosx_11_0_arm64.whl.metadata (33 kB)
Collecting idna<4,>=2.5 (from requests<3,>=2.21.0->tensorflow==2.16.1->tensorflow-macos)
  Using cached idna-3.6-py3-none-any.whl.metadata (9.9 kB)
Collecting urllib3<3,>=1.21.1 (from requests<3,>=2.21.0->tensorflow==2.16.1->tensorflow-macos)
  Using cached urllib3-2.2.1-py3-none-any.whl.metadata (6.4 kB)
Collecting certifi>=2017.4.17 (from requests<3,>=2.21.0->tensorflow==2.16.1->tensorflow-macos)
  Using cached certifi-2024.2.2-py3-none-any.whl.metadata (2.2 kB)
Collecting markdown>=2.6.8 (from tensorboard<2.17,>=2.16->tensorflow==2.16.1->tensorflow-macos)
  Using cached Markdown-3.6-py3-none-any.whl.metadata (7.0 kB)
Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard<2.17,>=2.16->tensorflow==2.16.1->tensorflow-macos)
  Using cached tensorboard_data_server-0.7.2-py3-none-any.whl.metadata (1.1 kB)
Collecting werkzeug>=1.0.1 (from tensorboard<2.17,>=2.16->tensorflow==2.16.1->tensorflow-macos)
  Using cached werkzeug-3.0.2-py3-none-any.whl.metadata (4.1 kB)
Collecting MarkupSafe>=2.1.1 (from werkzeug>=1.0.1->tensorboard<2.17,>=2.16->tensorflow==2.16.1->tensorflow-macos)
  Using cached MarkupSafe-2.1.5-cp310-cp310-macosx_10_9_universal2.whl.metadata (3.0 kB)
Collecting markdown-it-py>=2.2.0 (from rich->keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos)
  Using cached markdown_it_py-3.0.0-py3-none-any.whl.metadata (6.9 kB)
Collecting pygments<3.0.0,>=2.13.0 (from rich->keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos)
  Downloading pygments-2.17.2-py3-none-any.whl.metadata (2.6 kB)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich->keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos)
  Using cached mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Using cached tensorflow_macos-2.16.1-cp310-cp310-macosx_12_0_arm64.whl (2.2 kB)
Using cached tensorflow-2.16.1-cp310-cp310-macosx_12_0_arm64.whl (227.0 MB)
Using cached absl_py-2.1.0-py3-none-any.whl (133 kB)
Using cached astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
Using cached flatbuffers-24.3.25-py2.py3-none-any.whl (26 kB)
Using cached gast-0.5.4-py3-none-any.whl (19 kB)
Using cached google_pasta-0.2.0-py3-none-any.whl (57 kB)
Using cached grpcio-1.62.1-cp310-cp310-macosx_12_0_universal2.whl (10.0 MB)
Using cached h5py-3.10.0-cp310-cp310-macosx_11_0_arm64.whl (2.7 MB)
Using cached keras-3.1.1-py3-none-any.whl (1.1 MB)
Using cached libclang-18.1.1-py2.py3-none-macosx_11_0_arm64.whl (26.4 MB)
Using cached ml_dtypes-0.3.2-cp310-cp310-macosx_10_9_universal2.whl (389 kB)
Using cached numpy-1.26.4-cp310-cp310-macosx_11_0_arm64.whl (14.0 MB)
Using cached opt_einsum-3.3.0-py3-none-any.whl (65 kB)
Using cached protobuf-4.25.3-cp37-abi3-macosx_10_9_universal2.whl (394 kB)
Using cached requests-2.31.0-py3-none-any.whl (62 kB)
Using cached six-1.16.0-py2.py3-none-any.whl (11 kB)
Using cached tensorboard-2.16.2-py3-none-any.whl (5.5 MB)
Using cached tensorflow_io_gcs_filesystem-0.36.0-cp310-cp310-macosx_12_0_arm64.whl (3.4 MB)
Using cached termcolor-2.4.0-py3-none-any.whl (7.7 kB)
Using cached typing_extensions-4.11.0-py3-none-any.whl (34 kB)
Using cached wrapt-1.16.0-cp310-cp310-macosx_11_0_arm64.whl (38 kB)
Using cached packaging-24.0-py3-none-any.whl (53 kB)
Using cached certifi-2024.2.2-py3-none-any.whl (163 kB)
Using cached charset_normalizer-3.3.2-cp310-cp310-macosx_11_0_arm64.whl (120 kB)
Using cached idna-3.6-py3-none-any.whl (61 kB)
Using cached Markdown-3.6-py3-none-any.whl (105 kB)
Using cached tensorboard_data_server-0.7.2-py3-none-any.whl (2.4 kB)
Using cached urllib3-2.2.1-py3-none-any.whl (121 kB)
Using cached werkzeug-3.0.2-py3-none-any.whl (226 kB)
Using cached wheel-0.43.0-py3-none-any.whl (65 kB)
Using cached namex-0.0.7-py3-none-any.whl (5.8 kB)
Using cached optree-0.11.0-cp310-cp310-macosx_11_0_arm64.whl (273 kB)
Using cached rich-13.7.1-py3-none-any.whl (240 kB)
Using cached markdown_it_py-3.0.0-py3-none-any.whl (87 kB)
Using cached MarkupSafe-2.1.5-cp310-cp310-macosx_10_9_universal2.whl (18 kB)
Downloading pygments-2.17.2-py3-none-any.whl (1.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 789.4 kB/s eta 0:00:00
Using cached mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Installing collected packages: namex, libclang, flatbuffers, wrapt, wheel, urllib3, typing-extensions, termcolor, tensorflow-io-gcs-filesystem, tensorboard-data-server, six, pygments, protobuf, packaging, numpy, mdurl, MarkupSafe, markdown, idna, grpcio, gast, charset-normalizer, certifi, absl-py, werkzeug, requests, optree, opt-einsum, ml-dtypes, markdown-it-py, h5py, google-pasta, astunparse, tensorboard, rich, keras, tensorflow, tensorflow-macos
Successfully installed MarkupSafe-2.1.5 absl-py-2.1.0 astunparse-1.6.3 certifi-2024.2.2 charset-normalizer-3.3.2 flatbuffers-24.3.25 gast-0.5.4 google-pasta-0.2.0 grpcio-1.62.1 h5py-3.10.0 idna-3.6 keras-3.1.1 libclang-18.1.1 markdown-3.6 markdown-it-py-3.0.0 mdurl-0.1.2 ml-dtypes-0.3.2 namex-0.0.7 numpy-1.26.4 opt-einsum-3.3.0 optree-0.11.0 packaging-24.0 protobuf-4.25.3 pygments-2.17.2 requests-2.31.0 rich-13.7.1 six-1.16.0 tensorboard-2.16.2 tensorboard-data-server-0.7.2 tensorflow-2.16.1 tensorflow-io-gcs-filesystem-0.36.0 tensorflow-macos-2.16.1 termcolor-2.4.0 typing-extensions-4.11.0 urllib3-2.2.1 werkzeug-3.0.2 wheel-0.43.0 wrapt-1.16.0
Collecting tensorflow-metal
  Using cached tensorflow_metal-1.1.0-cp310-cp310-macosx_12_0_arm64.whl.metadata (1.2 kB)
Requirement already satisfied: wheel~=0.35 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow-metal) (0.43.0)
Requirement already satisfied: six>=1.15.0 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow-metal) (1.16.0)
Using cached tensorflow_metal-1.1.0-cp310-cp310-macosx_12_0_arm64.whl (1.4 MB)
Installing collected packages: tensorflow-metal
Successfully installed tensorflow-metal-1.1.0
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

2024-04-07 23:03:26.687822: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M2 Pro
2024-04-07 23:03:26.687845: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
2024-04-07 23:03:26.687850: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.33 GB
2024-04-07 23:03:26.687887: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2024-04-07 23:03:26.687901: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
tf.Tensor(-28.180496, shape=(), dtype=float32)
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Convolutional_Neural_Network.py", line 106, in <module>
    import librosa
ModuleNotFoundError: No module named 'librosa'
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) pip install librosa

Collecting librosa
  Using cached librosa-0.10.1-py3-none-any.whl.metadata (8.3 kB)
Collecting audioread>=2.1.9 (from librosa)
  Using cached audioread-3.0.1-py3-none-any.whl.metadata (8.4 kB)
Requirement already satisfied: numpy!=1.22.0,!=1.22.1,!=1.22.2,>=1.20.3 in ./myprojectenv/lib/python3.10/site-packages (from librosa) (1.26.4)
Collecting scipy>=1.2.0 (from librosa)
  Downloading scipy-1.13.0-cp310-cp310-macosx_12_0_arm64.whl.metadata (60 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.6/60.6 kB 966.0 kB/s eta 0:00:00
Collecting scikit-learn>=0.20.0 (from librosa)
  Using cached scikit_learn-1.4.1.post1-cp310-cp310-macosx_12_0_arm64.whl.metadata (11 kB)
Collecting joblib>=0.14 (from librosa)
  Using cached joblib-1.3.2-py3-none-any.whl.metadata (5.4 kB)
Collecting decorator>=4.3.0 (from librosa)
  Downloading decorator-5.1.1-py3-none-any.whl.metadata (4.0 kB)
Collecting numba>=0.51.0 (from librosa)
  Using cached numba-0.59.1-cp310-cp310-macosx_11_0_arm64.whl.metadata (2.7 kB)
Collecting soundfile>=0.12.1 (from librosa)
  Using cached soundfile-0.12.1-py2.py3-none-macosx_11_0_arm64.whl.metadata (14 kB)
Collecting pooch>=1.0 (from librosa)
  Using cached pooch-1.8.1-py3-none-any.whl.metadata (9.5 kB)
Collecting soxr>=0.3.2 (from librosa)
  Using cached soxr-0.3.7-cp310-cp310-macosx_11_0_arm64.whl.metadata (5.5 kB)
Requirement already satisfied: typing-extensions>=4.1.1 in ./myprojectenv/lib/python3.10/site-packages (from librosa) (4.11.0)
Collecting lazy-loader>=0.1 (from librosa)
  Downloading lazy_loader-0.4-py3-none-any.whl.metadata (7.6 kB)
Collecting msgpack>=1.0 (from librosa)
  Using cached msgpack-1.0.8-cp310-cp310-macosx_11_0_arm64.whl.metadata (9.1 kB)
Requirement already satisfied: packaging in ./myprojectenv/lib/python3.10/site-packages (from lazy-loader>=0.1->librosa) (24.0)
Collecting llvmlite<0.43,>=0.42.0dev0 (from numba>=0.51.0->librosa)
  Using cached llvmlite-0.42.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (4.8 kB)
Collecting platformdirs>=2.5.0 (from pooch>=1.0->librosa)
  Using cached platformdirs-4.2.0-py3-none-any.whl.metadata (11 kB)
Requirement already satisfied: requests>=2.19.0 in ./myprojectenv/lib/python3.10/site-packages (from pooch>=1.0->librosa) (2.31.0)
Collecting threadpoolctl>=2.0.0 (from scikit-learn>=0.20.0->librosa)
  Downloading threadpoolctl-3.4.0-py3-none-any.whl.metadata (13 kB)
Collecting cffi>=1.0 (from soundfile>=0.12.1->librosa)
  Downloading cffi-1.16.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (1.5 kB)
Collecting pycparser (from cffi>=1.0->soundfile>=0.12.1->librosa)
  Downloading pycparser-2.22-py3-none-any.whl.metadata (943 bytes)
Requirement already satisfied: charset-normalizer<4,>=2 in ./myprojectenv/lib/python3.10/site-packages (from requests>=2.19.0->pooch>=1.0->librosa) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in ./myprojectenv/lib/python3.10/site-packages (from requests>=2.19.0->pooch>=1.0->librosa) (3.6)
Requirement already satisfied: urllib3<3,>=1.21.1 in ./myprojectenv/lib/python3.10/site-packages (from requests>=2.19.0->pooch>=1.0->librosa) (2.2.1)
Requirement already satisfied: certifi>=2017.4.17 in ./myprojectenv/lib/python3.10/site-packages (from requests>=2.19.0->pooch>=1.0->librosa) (2024.2.2)
Using cached librosa-0.10.1-py3-none-any.whl (253 kB)
Using cached audioread-3.0.1-py3-none-any.whl (23 kB)
Downloading decorator-5.1.1-py3-none-any.whl (9.1 kB)
Using cached joblib-1.3.2-py3-none-any.whl (302 kB)
Downloading lazy_loader-0.4-py3-none-any.whl (12 kB)
Using cached msgpack-1.0.8-cp310-cp310-macosx_11_0_arm64.whl (84 kB)
Using cached numba-0.59.1-cp310-cp310-macosx_11_0_arm64.whl (2.6 MB)
Using cached pooch-1.8.1-py3-none-any.whl (62 kB)
Using cached scikit_learn-1.4.1.post1-cp310-cp310-macosx_12_0_arm64.whl (10.4 MB)
Downloading scipy-1.13.0-cp310-cp310-macosx_12_0_arm64.whl (30.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 30.3/30.3 MB 3.1 MB/s eta 0:00:00
Using cached soundfile-0.12.1-py2.py3-none-macosx_11_0_arm64.whl (1.1 MB)
Using cached soxr-0.3.7-cp310-cp310-macosx_11_0_arm64.whl (390 kB)
Downloading cffi-1.16.0-cp310-cp310-macosx_11_0_arm64.whl (176 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 176.8/176.8 kB 5.5 MB/s eta 0:00:00
Using cached llvmlite-0.42.0-cp310-cp310-macosx_11_0_arm64.whl (28.8 MB)
Using cached platformdirs-4.2.0-py3-none-any.whl (17 kB)
Downloading threadpoolctl-3.4.0-py3-none-any.whl (17 kB)
Downloading pycparser-2.22-py3-none-any.whl (117 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 117.6/117.6 kB 4.1 MB/s eta 0:00:00
Installing collected packages: threadpoolctl, soxr, scipy, pycparser, platformdirs, msgpack, llvmlite, lazy-loader, joblib, decorator, audioread, scikit-learn, pooch, numba, cffi, soundfile, librosa
Successfully installed audioread-3.0.1 cffi-1.16.0 decorator-5.1.1 joblib-1.3.2 lazy-loader-0.4 librosa-0.10.1 llvmlite-0.42.0 msgpack-1.0.8 numba-0.59.1 platformdirs-4.2.0 pooch-1.8.1 pycparser-2.22 scikit-learn-1.4.1.post1 scipy-1.13.0 soundfile-0.12.1 soxr-0.3.7 threadpoolctl-3.4.0
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Convolutional_Neural_Network.py", line 114, in <module>
    from tensorflow_model_optimization.sparsity import keras as sparsity
ModuleNotFoundError: No module named 'tensorflow_model_optimization'
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) pip install tensorflow-model-optimization

Collecting tensorflow-model-optimization
  Using cached tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl.metadata (904 bytes)
Collecting absl-py~=1.2 (from tensorflow-model-optimization)
  Using cached absl_py-1.4.0-py3-none-any.whl.metadata (2.3 kB)
Collecting dm-tree~=0.1.1 (from tensorflow-model-optimization)
  Using cached dm_tree-0.1.8-cp310-cp310-macosx_11_0_arm64.whl.metadata (1.9 kB)
Requirement already satisfied: numpy~=1.23 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow-model-optimization) (1.26.4)
Requirement already satisfied: six~=1.14 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow-model-optimization) (1.16.0)
Using cached tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl (242 kB)
Using cached absl_py-1.4.0-py3-none-any.whl (126 kB)
Using cached dm_tree-0.1.8-cp310-cp310-macosx_11_0_arm64.whl (110 kB)
Installing collected packages: dm-tree, absl-py, tensorflow-model-optimization
  Attempting uninstall: absl-py
    Found existing installation: absl-py 2.1.0
    Uninstalling absl-py-2.1.0:
      Successfully uninstalled absl-py-2.1.0
Successfully installed absl-py-1.4.0 dm-tree-0.1.8 tensorflow-model-optimization-0.8.0
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Convolutional_Neural_Network.py", line 114, in <module>
    from tensorflow_model_optimization.sparsity import keras as sparsity
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/__init__.py", line 86, in <module>
    from tensorflow_model_optimization.python.core.api import clustering
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/__init__.py", line 16, in <module>
    from tensorflow_model_optimization.python.core.api import clustering
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/clustering/__init__.py", line 16, in <module>
    from tensorflow_model_optimization.python.core.api.clustering import keras
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/clustering/keras/__init__.py", line 19, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras.cluster import cluster_scope
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/cluster.py", line 22, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras import cluster_wrapper
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/cluster_wrapper.py", line 23, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/clustering_centroids.py", line 22, in <module>
    from tensorflow_model_optimization.python.core.keras.compat import keras
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/keras/compat.py", line 41, in <module>
    keras = _get_keras_instance()
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/keras/compat.py", line 35, in _get_keras_instance
    import tf_keras as keras_internal  # pylint: disable=g-import-not-at-top,unused-import
ModuleNotFoundError: No module named 'tf_keras'
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)



#######################################
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Convolutional_Neural_Network.py", line 114, in <module>
    import tensorflow_model_optimization as tfmot
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/__init__.py", line 86, in <module>
    from tensorflow_model_optimization.python.core.api import clustering
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/__init__.py", line 16, in <module>
    from tensorflow_model_optimization.python.core.api import clustering
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/clustering/__init__.py", line 16, in <module>
    from tensorflow_model_optimization.python.core.api.clustering import keras
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/clustering/keras/__init__.py", line 19, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras.cluster import cluster_scope
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/cluster.py", line 22, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras import cluster_wrapper
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/cluster_wrapper.py", line 23, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/clustering_centroids.py", line 22, in <module>
    from tensorflow_model_optimization.python.core.keras.compat import keras
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/keras/compat.py", line 41, in <module>
    keras = _get_keras_instance()
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/keras/compat.py", line 35, in _get_keras_instance
    import tf_keras as keras_internal  # pylint: disable=g-import-not-at-top,unused-import
ModuleNotFoundError: No module named 'tf_keras'
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) xcode-select --install

xcode-select: error: command line tools are already installed, use "Software Update" in System Settings to install updates
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) brew install miniforge

==> Downloading https://formulae.brew.sh/api/formula.jws.json
#=O#- #
==> Downloading https://formulae.brew.sh/api/cask.jws.json
#################################################################################################### 100.0%
==> Caveats
Please run the following to setup your shell:
  conda init "$(basename "${SHELL}")"

==> Downloading https://github.com/conda-forge/miniforge/releases/download/24.1.2-0/Miniforge3-24.1.2-0-Mac
==> Downloading from https://objects.githubusercontent.com/github-production-release-asset-2e65be/221584272
#################################################################################################### 100.0%
==> Installing Cask miniforge
==> Running installer script 'Miniforge3-24.1.2-0-MacOSX-x86_64.sh'
PREFIX=/usr/local/Caskroom/miniforge/base
Unpacking payload ...
Extracting bzip2-1.0.8-h10d778d_5.conda
Extracting c-ares-1.27.0-h10d778d_0.conda
Extracting ca-certificates-2024.2.2-h8857fd0_0.conda
Extracting icu-73.2-hf5e326d_0.conda
Extracting libcxx-16.0.6-hd57cbcb_0.conda
Extracting libev-4.33-h10d778d_2.conda
Extracting libffi-3.4.2-h0d85af4_5.tar.bz2
Extracting libiconv-1.17-hd75f5a5_2.conda
Extracting libzlib-1.2.13-h8a1eda9_5.conda
Extracting lzo-2.10-haf1e3a3_1000.tar.bz2
Extracting ncurses-6.4.20240210-h73e2aa4_0.conda
Extracting pybind11-abi-4-hd8ed1ab_3.tar.bz2
Extracting python_abi-3.10-4_cp310.conda
Extracting reproc-14.2.4.post0-h10d778d_1.conda
Extracting tzdata-2024a-h0c530f3_0.conda
Extracting xz-5.2.6-h775f41a_0.tar.bz2
Extracting fmt-10.2.1-h7728843_0.conda
Extracting libedit-3.1.20191231-h0678c8f_2.tar.bz2
Extracting libsolv-0.7.28-h2d185b6_0.conda
Extracting libsqlite-3.45.2-h92b6c6a_0.conda
Extracting libxml2-2.12.6-hc0ae0f7_0.conda
Extracting lz4-c-1.9.4-hf0c8a7f_0.conda
Extracting openssl-3.2.1-hd75f5a5_1.conda
Extracting readline-8.2-h9e318b2_1.conda
Extracting reproc-cpp-14.2.4.post0-h93d8f39_1.conda
Extracting tk-8.6.13-h1abcd95_1.conda
Extracting yaml-cpp-0.8.0-he965462_0.conda
Extracting zstd-1.5.5-h829000d_0.conda
Extracting krb5-1.21.2-hb884880_0.conda
Extracting libarchive-3.7.2-hd35d340_1.conda
Extracting libnghttp2-1.58.0-h64cf6d3_1.conda
Extracting libssh2-1.11.0-hd019ec5_0.conda
Extracting python-3.10.14-h00d2728_0_cpython.conda
Extracting libcurl-8.6.0-h726d00d_0.conda
Extracting menuinst-2.0.2-py310h2ec42d9_0.conda
Extracting archspec-0.2.3-pyhd8ed1ab_0.conda
Extracting boltons-23.1.1-pyhd8ed1ab_0.conda
Extracting brotli-python-1.1.0-py310h9e9d8ca_1.conda
Extracting certifi-2024.2.2-pyhd8ed1ab_0.conda
Extracting charset-normalizer-3.3.2-pyhd8ed1ab_0.conda
Extracting colorama-0.4.6-pyhd8ed1ab_0.tar.bz2
Extracting distro-1.9.0-pyhd8ed1ab_0.conda
Extracting idna-3.6-pyhd8ed1ab_0.conda
Extracting jsonpointer-2.4-py310h2ec42d9_3.conda
Extracting libmamba-1.5.7-ha449628_0.conda
Extracting packaging-24.0-pyhd8ed1ab_0.conda
Extracting platformdirs-4.2.0-pyhd8ed1ab_0.conda
Extracting pluggy-1.4.0-pyhd8ed1ab_0.conda
Extracting pycosat-0.6.6-py310h6729b98_0.conda
Extracting pycparser-2.21-pyhd8ed1ab_0.tar.bz2
Extracting pysocks-1.7.1-pyha2e5f31_6.tar.bz2
Extracting ruamel.yaml.clib-0.2.8-py310hb372a2b_0.conda
Extracting setuptools-69.2.0-pyhd8ed1ab_0.conda
Extracting truststore-0.8.0-pyhd8ed1ab_0.conda
Extracting wheel-0.43.0-pyhd8ed1ab_0.conda
Extracting cffi-1.16.0-py310hdca579f_0.conda
Extracting jsonpatch-1.33-pyhd8ed1ab_0.conda
Extracting libmambapy-1.5.7-py310hd168405_0.conda
Extracting pip-24.0-pyhd8ed1ab_0.conda
Extracting ruamel.yaml-0.18.6-py310hb372a2b_0.conda
Extracting tqdm-4.66.2-pyhd8ed1ab_0.conda
Extracting urllib3-2.2.1-pyhd8ed1ab_0.conda
Extracting requests-2.31.0-pyhd8ed1ab_0.conda
Extracting zstandard-0.22.0-py310hd88f66e_0.conda
Extracting conda-package-streaming-0.9.0-pyhd8ed1ab_0.conda
Extracting conda-package-handling-2.2.0-pyh38be061_0.conda
Extracting conda-libmamba-solver-24.1.0-pyhd8ed1ab_0.conda
Extracting conda-24.1.2-py310h2ec42d9_0.conda
Extracting mamba-1.5.7-py310h6bde348_0.conda

Installing base environment...


                                           __
          __  ______ ___  ____ _____ ___  / /_  ____ _
         / / / / __ `__ \/ __ `/ __ `__ \/ __ \/ __ `/
        / /_/ / / / / / / /_/ / / / / / / /_/ / /_/ /
       / .___/_/ /_/ /_/\__,_/_/ /_/ /_/_.___/\__,_/
      /_/

Transaction

  Prefix: /usr/local/Caskroom/miniforge/base

  Updating specs:

   - conda-forge/osx-64::bzip2==1.0.8=h10d778d_5[md5=6097a6ca9ada32699b5fc4312dd6ef18]
   - conda-forge/osx-64::c-ares==1.27.0=h10d778d_0[md5=713dd57081dfe8535eb961b45ed26a0c]
   - conda-forge/osx-64::ca-certificates==2024.2.2=h8857fd0_0[md5=f2eacee8c33c43692f1ccfd33d0f50b1]
   - conda-forge/osx-64::icu==73.2=hf5e326d_0[md5=5cc301d759ec03f28328428e28f65591]
   - conda-forge/osx-64::libcxx==16.0.6=hd57cbcb_0[md5=7d6972792161077908b62971802f289a]
   - conda-forge/osx-64::libev==4.33=h10d778d_2[md5=899db79329439820b7e8f8de41bca902]
   - conda-forge/osx-64::libffi==3.4.2=h0d85af4_5[md5=ccb34fb14960ad8b125962d3d79b31a9]
   - conda-forge/osx-64::libiconv==1.17=hd75f5a5_2[md5=6c3628d047e151efba7cf08c5e54d1ca]
   - conda-forge/osx-64::libzlib==1.2.13=h8a1eda9_5[md5=4a3ad23f6e16f99c04e166767193d700]
   - conda-forge/osx-64::lzo==2.10=haf1e3a3_1000[md5=0b6bca372a95d6c602c7a922e928ce79]
   - conda-forge/osx-64::ncurses==6.4.20240210=h73e2aa4_0[md5=50f28c512e9ad78589e3eab34833f762]
   - conda-forge/noarch::pybind11-abi==4=hd8ed1ab_3[md5=878f923dd6acc8aeb47a75da6c4098be]
   - conda-forge/osx-64::python_abi==3.10=4_cp310[md5=b15c816c5a86abcc4d1458dd63aa4c65]
   - conda-forge/osx-64::reproc==14.2.4.post0=h10d778d_1[md5=d7c3258e871481be5bbaf28b4729e29f]
   - conda-forge/noarch::tzdata==2024a=h0c530f3_0[md5=161081fc7cec0bfda0d86d7cb595f8d8]
   - conda-forge/osx-64::xz==5.2.6=h775f41a_0[md5=a72f9d4ea13d55d745ff1ed594747f10]
   - conda-forge/osx-64::fmt==10.2.1=h7728843_0[md5=ab205d53bda43d03f5c5b993ccb406b3]
   - conda-forge/osx-64::libedit==3.1.20191231=h0678c8f_2[md5=6016a8a1d0e63cac3de2c352cd40208b]
   - conda-forge/osx-64::libsolv==0.7.28=h2d185b6_0[md5=a30cb23edd3ef8c8a7a8e83c1bee9295]
   - conda-forge/osx-64::libsqlite==3.45.2=h92b6c6a_0[md5=086f56e13a96a6cfb1bf640505ae6b70]
   - conda-forge/osx-64::libxml2==2.12.6=hc0ae0f7_0[md5=913ce3dbfa8677fba65c44647ef88594]
   - conda-forge/osx-64::lz4-c==1.9.4=hf0c8a7f_0[md5=aa04f7143228308662696ac24023f991]
   - conda-forge/osx-64::openssl==3.2.1=hd75f5a5_1[md5=570a6f04802df580be529f3a72d2bbf7]
   - conda-forge/osx-64::readline==8.2=h9e318b2_1[md5=f17f77f2acf4d344734bda76829ce14e]
   - conda-forge/osx-64::reproc-cpp==14.2.4.post0=h93d8f39_1[md5=a32e95ada0ee860c91e87266700970c3]
   - conda-forge/osx-64::tk==8.6.13=h1abcd95_1[md5=bf830ba5afc507c6232d4ef0fb1a882d]
   - conda-forge/osx-64::yaml-cpp==0.8.0=he965462_0[md5=1bb3addc859ed1338370da6e2996ef47]
   - conda-forge/osx-64::zstd==1.5.5=h829000d_0[md5=80abc41d0c48b82fe0f04e7f42f5cb7e]
   - conda-forge/osx-64::krb5==1.21.2=hb884880_0[md5=80505a68783f01dc8d7308c075261b2f]
   - conda-forge/osx-64::libarchive==3.7.2=hd35d340_1[md5=8c7b79b20a67287a87b39df8a8c8dcc4]
   - conda-forge/osx-64::libnghttp2==1.58.0=h64cf6d3_1[md5=faecc55c2a8155d9ff1c0ff9a0fef64f]
   - conda-forge/osx-64::libssh2==1.11.0=hd019ec5_0[md5=ca3a72efba692c59a90d4b9fc0dfe774]
   - conda-forge/osx-64::python==3.10.14=h00d2728_0_cpython[md5=0a1cddc4382c5c171e791c70740546dd]
   - conda-forge/osx-64::libcurl==8.6.0=h726d00d_0[md5=09569d6e3dc8bef57841f1fc69ea3ea6]
   - conda-forge/osx-64::menuinst==2.0.2=py310h2ec42d9_0[md5=695a6f4401b25418956a90fe6301f50d]
   - conda-forge/noarch::archspec==0.2.3=pyhd8ed1ab_0[md5=192278292e20704f663b9c766909d67b]
   - conda-forge/noarch::boltons==23.1.1=pyhd8ed1ab_0[md5=56febe65315cc388a5d20adf2b39a74d]
   - conda-forge/osx-64::brotli-python==1.1.0=py310h9e9d8ca_1[md5=2362e323293e7699cf1e621d502f86d6]
   - conda-forge/noarch::certifi==2024.2.2=pyhd8ed1ab_0[md5=0876280e409658fc6f9e75d035960333]
   - conda-forge/noarch::charset-normalizer==3.3.2=pyhd8ed1ab_0[md5=7f4a9e3fcff3f6356ae99244a014da6a]
   - conda-forge/noarch::colorama==0.4.6=pyhd8ed1ab_0[md5=3faab06a954c2a04039983f2c4a50d99]
   - conda-forge/noarch::distro==1.9.0=pyhd8ed1ab_0[md5=bbdb409974cd6cb30071b1d978302726]
   - conda-forge/noarch::idna==3.6=pyhd8ed1ab_0[md5=1a76f09108576397c41c0b0c5bd84134]
   - conda-forge/osx-64::jsonpointer==2.4=py310h2ec42d9_3[md5=ca02450dbc1c346a06fc454b36ddab32]
   - conda-forge/osx-64::libmamba==1.5.7=ha449628_0[md5=f5fd8194c6af19225f848a616f53baa7]
   - conda-forge/noarch::packaging==24.0=pyhd8ed1ab_0[md5=248f521b64ce055e7feae3105e7abeb8]
   - conda-forge/noarch::platformdirs==4.2.0=pyhd8ed1ab_0[md5=a0bc3eec34b0fab84be6b2da94e98e20]
   - conda-forge/noarch::pluggy==1.4.0=pyhd8ed1ab_0[md5=139e9feb65187e916162917bb2484976]
   - conda-forge/osx-64::pycosat==0.6.6=py310h6729b98_0[md5=89b601f80d076bf8053eea906293353c]
   - conda-forge/noarch::pycparser==2.21=pyhd8ed1ab_0[md5=076becd9e05608f8dc72757d5f3a91ff]
   - conda-forge/noarch::pysocks==1.7.1=pyha2e5f31_6[md5=2a7de29fb590ca14b5243c4c812c8025]
   - conda-forge/osx-64::ruamel.yaml.clib==0.2.8=py310hb372a2b_0[md5=a6254db88b5bf45d4870c3a63dc39e8d]
   - conda-forge/noarch::setuptools==69.2.0=pyhd8ed1ab_0[md5=da214ecd521a720a9d521c68047682dc]
   - conda-forge/noarch::truststore==0.8.0=pyhd8ed1ab_0[md5=08316d001eca8854392cf2837828ea11]
   - conda-forge/noarch::wheel==0.43.0=pyhd8ed1ab_0[md5=6e43a94e0ee67523b5e781f0faba5c45]
   - conda-forge/osx-64::cffi==1.16.0=py310hdca579f_0[md5=b9e6213f0eb91f40c009ce69139c1869]
   - conda-forge/noarch::jsonpatch==1.33=pyhd8ed1ab_0[md5=bfdb7c5c6ad1077c82a69a8642c87aff]
   - conda-forge/osx-64::libmambapy==1.5.7=py310hd168405_0[md5=f981f435165e3053d9aa29603958962b]
   - conda-forge/noarch::pip==24.0=pyhd8ed1ab_0[md5=f586ac1e56c8638b64f9c8122a7b8a67]
   - conda-forge/osx-64::ruamel.yaml==0.18.6=py310hb372a2b_0[md5=a6691c80f3bf62bc0df37b87caac6a70]
   - conda-forge/noarch::tqdm==4.66.2=pyhd8ed1ab_0[md5=2b8dfb969f984497f3f98409a9545776]
   - conda-forge/noarch::urllib3==2.2.1=pyhd8ed1ab_0[md5=08807a87fa7af10754d46f63b368e016]
   - conda-forge/noarch::requests==2.31.0=pyhd8ed1ab_0[md5=a30144e4156cdbb236f99ebb49828f8b]
   - conda-forge/osx-64::zstandard==0.22.0=py310hd88f66e_0[md5=88c991558201cae2b7e690c2e9d2e618]
   - conda-forge/noarch::conda-package-streaming==0.9.0=pyhd8ed1ab_0[md5=38253361efb303deead3eab39ae9269b]
   - conda-forge/noarch::conda-package-handling==2.2.0=pyh38be061_0[md5=8a3ae7f6318376aa08ea753367bb7dd6]
   - conda-forge/noarch::conda-libmamba-solver==24.1.0=pyhd8ed1ab_0[md5=304dc78ad6e52e0fd663df1d484c1531]
   - conda-forge/osx-64::conda==24.1.2=py310h2ec42d9_0[md5=c60e9dbe2f9a13f4b6d3521f46c02ef8]
   - conda-forge/osx-64::mamba==1.5.7=py310h6bde348_0[md5=2d0208c01f24601e7671487d81afc2c1]


  Package                         Version  Build               Channel           Size
───────────────────────────────────────────────────────────────────────────────────────
  Install:
───────────────────────────────────────────────────────────────────────────────────────

  + archspec                        0.2.3  pyhd8ed1ab_0        conda-forge     Cached
  + boltons                        23.1.1  pyhd8ed1ab_0        conda-forge     Cached
  + brotli-python                   1.1.0  py310h9e9d8ca_1     conda-forge     Cached
  + bzip2                           1.0.8  h10d778d_5          conda-forge     Cached
  + c-ares                         1.27.0  h10d778d_0          conda-forge     Cached
  + ca-certificates              2024.2.2  h8857fd0_0          conda-forge     Cached
  + certifi                      2024.2.2  pyhd8ed1ab_0        conda-forge     Cached
  + cffi                           1.16.0  py310hdca579f_0     conda-forge     Cached
  + charset-normalizer              3.3.2  pyhd8ed1ab_0        conda-forge     Cached
  + colorama                        0.4.6  pyhd8ed1ab_0        conda-forge     Cached
  + conda                          24.1.2  py310h2ec42d9_0     conda-forge     Cached
  + conda-libmamba-solver          24.1.0  pyhd8ed1ab_0        conda-forge     Cached
  + conda-package-handling          2.2.0  pyh38be061_0        conda-forge     Cached
  + conda-package-streaming         0.9.0  pyhd8ed1ab_0        conda-forge     Cached
  + distro                          1.9.0  pyhd8ed1ab_0        conda-forge     Cached
  + fmt                            10.2.1  h7728843_0          conda-forge     Cached
  + icu                              73.2  hf5e326d_0          conda-forge     Cached
  + idna                              3.6  pyhd8ed1ab_0        conda-forge     Cached
  + jsonpatch                        1.33  pyhd8ed1ab_0        conda-forge     Cached
  + jsonpointer                       2.4  py310h2ec42d9_3     conda-forge     Cached
  + krb5                           1.21.2  hb884880_0          conda-forge     Cached
  + libarchive                      3.7.2  hd35d340_1          conda-forge     Cached
  + libcurl                         8.6.0  h726d00d_0          conda-forge     Cached
  + libcxx                         16.0.6  hd57cbcb_0          conda-forge     Cached
  + libedit                  3.1.20191231  h0678c8f_2          conda-forge     Cached
  + libev                            4.33  h10d778d_2          conda-forge     Cached
  + libffi                          3.4.2  h0d85af4_5          conda-forge     Cached
  + libiconv                         1.17  hd75f5a5_2          conda-forge     Cached
  + libmamba                        1.5.7  ha449628_0          conda-forge     Cached
  + libmambapy                      1.5.7  py310hd168405_0     conda-forge     Cached
  + libnghttp2                     1.58.0  h64cf6d3_1          conda-forge     Cached
  + libsolv                        0.7.28  h2d185b6_0          conda-forge     Cached
  + libsqlite                      3.45.2  h92b6c6a_0          conda-forge     Cached
  + libssh2                        1.11.0  hd019ec5_0          conda-forge     Cached
  + libxml2                        2.12.6  hc0ae0f7_0          conda-forge     Cached
  + libzlib                        1.2.13  h8a1eda9_5          conda-forge     Cached
  + lz4-c                           1.9.4  hf0c8a7f_0          conda-forge     Cached
  + lzo                              2.10  haf1e3a3_1000       conda-forge     Cached
  + mamba                           1.5.7  py310h6bde348_0     conda-forge     Cached
  + menuinst                        2.0.2  py310h2ec42d9_0     conda-forge     Cached
  + ncurses                  6.4.20240210  h73e2aa4_0          conda-forge     Cached
  + openssl                         3.2.1  hd75f5a5_1          conda-forge     Cached
  + packaging                        24.0  pyhd8ed1ab_0        conda-forge     Cached
  + pip                              24.0  pyhd8ed1ab_0        conda-forge     Cached
  + platformdirs                    4.2.0  pyhd8ed1ab_0        conda-forge     Cached
  + pluggy                          1.4.0  pyhd8ed1ab_0        conda-forge     Cached
  + pybind11-abi                        4  hd8ed1ab_3          conda-forge     Cached
  + pycosat                         0.6.6  py310h6729b98_0     conda-forge     Cached
  + pycparser                        2.21  pyhd8ed1ab_0        conda-forge     Cached
  + pysocks                         1.7.1  pyha2e5f31_6        conda-forge     Cached
  + python                        3.10.14  h00d2728_0_cpython  conda-forge     Cached
  + python_abi                       3.10  4_cp310             conda-forge     Cached
  + readline                          8.2  h9e318b2_1          conda-forge     Cached
  + reproc                   14.2.4.post0  h10d778d_1          conda-forge     Cached
  + reproc-cpp               14.2.4.post0  h93d8f39_1          conda-forge     Cached
  + requests                       2.31.0  pyhd8ed1ab_0        conda-forge     Cached
  + ruamel.yaml                    0.18.6  py310hb372a2b_0     conda-forge     Cached
  + ruamel.yaml.clib                0.2.8  py310hb372a2b_0     conda-forge     Cached
  + setuptools                     69.2.0  pyhd8ed1ab_0        conda-forge     Cached
  + tk                             8.6.13  h1abcd95_1          conda-forge     Cached
  + tqdm                           4.66.2  pyhd8ed1ab_0        conda-forge     Cached
  + truststore                      0.8.0  pyhd8ed1ab_0        conda-forge     Cached
  + tzdata                          2024a  h0c530f3_0          conda-forge     Cached
  + urllib3                         2.2.1  pyhd8ed1ab_0        conda-forge     Cached
  + wheel                          0.43.0  pyhd8ed1ab_0        conda-forge     Cached
  + xz                              5.2.6  h775f41a_0          conda-forge     Cached
  + yaml-cpp                        0.8.0  he965462_0          conda-forge     Cached
  + zstandard                      0.22.0  py310hd88f66e_0     conda-forge     Cached
  + zstd                            1.5.5  h829000d_0          conda-forge     Cached

  Summary:

  Install: 69 packages

  Total download: 0 B

───────────────────────────────────────────────────────────────────────────────────────



Transaction starting
Linking bzip2-1.0.8-h10d778d_5
Linking c-ares-1.27.0-h10d778d_0
Linking ca-certificates-2024.2.2-h8857fd0_0
Linking icu-73.2-hf5e326d_0
Linking libcxx-16.0.6-hd57cbcb_0
Linking libev-4.33-h10d778d_2
Linking libffi-3.4.2-h0d85af4_5
Linking libiconv-1.17-hd75f5a5_2
Linking libzlib-1.2.13-h8a1eda9_5
Linking lzo-2.10-haf1e3a3_1000
Linking ncurses-6.4.20240210-h73e2aa4_0
Linking pybind11-abi-4-hd8ed1ab_3
Linking python_abi-3.10-4_cp310
Linking reproc-14.2.4.post0-h10d778d_1
Linking tzdata-2024a-h0c530f3_0
Linking xz-5.2.6-h775f41a_0
Linking fmt-10.2.1-h7728843_0
Linking libedit-3.1.20191231-h0678c8f_2
Linking libsolv-0.7.28-h2d185b6_0
Linking libsqlite-3.45.2-h92b6c6a_0
Linking libxml2-2.12.6-hc0ae0f7_0
Linking lz4-c-1.9.4-hf0c8a7f_0
Linking openssl-3.2.1-hd75f5a5_1
Linking readline-8.2-h9e318b2_1
Linking reproc-cpp-14.2.4.post0-h93d8f39_1
Linking tk-8.6.13-h1abcd95_1
Linking yaml-cpp-0.8.0-he965462_0
Linking zstd-1.5.5-h829000d_0
Linking krb5-1.21.2-hb884880_0
Linking libarchive-3.7.2-hd35d340_1
Linking libnghttp2-1.58.0-h64cf6d3_1
Linking libssh2-1.11.0-hd019ec5_0
Linking python-3.10.14-h00d2728_0_cpython
Linking libcurl-8.6.0-h726d00d_0
Linking menuinst-2.0.2-py310h2ec42d9_0
Linking archspec-0.2.3-pyhd8ed1ab_0
Linking boltons-23.1.1-pyhd8ed1ab_0
Linking brotli-python-1.1.0-py310h9e9d8ca_1
Linking certifi-2024.2.2-pyhd8ed1ab_0
Linking charset-normalizer-3.3.2-pyhd8ed1ab_0
Linking colorama-0.4.6-pyhd8ed1ab_0
Linking distro-1.9.0-pyhd8ed1ab_0
Linking idna-3.6-pyhd8ed1ab_0
Linking jsonpointer-2.4-py310h2ec42d9_3
Linking libmamba-1.5.7-ha449628_0
Linking packaging-24.0-pyhd8ed1ab_0
Linking platformdirs-4.2.0-pyhd8ed1ab_0
Linking pluggy-1.4.0-pyhd8ed1ab_0
Linking pycosat-0.6.6-py310h6729b98_0
Linking pycparser-2.21-pyhd8ed1ab_0
Linking pysocks-1.7.1-pyha2e5f31_6
Linking ruamel.yaml.clib-0.2.8-py310hb372a2b_0
Linking setuptools-69.2.0-pyhd8ed1ab_0
Linking truststore-0.8.0-pyhd8ed1ab_0
Linking wheel-0.43.0-pyhd8ed1ab_0
Linking cffi-1.16.0-py310hdca579f_0
Linking jsonpatch-1.33-pyhd8ed1ab_0
Linking libmambapy-1.5.7-py310hd168405_0
Linking pip-24.0-pyhd8ed1ab_0
Linking ruamel.yaml-0.18.6-py310hb372a2b_0
Linking tqdm-4.66.2-pyhd8ed1ab_0
Linking urllib3-2.2.1-pyhd8ed1ab_0
Linking requests-2.31.0-pyhd8ed1ab_0
Linking zstandard-0.22.0-py310hd88f66e_0
Linking conda-package-streaming-0.9.0-pyhd8ed1ab_0
Linking conda-package-handling-2.2.0-pyh38be061_0
Linking conda-libmamba-solver-24.1.0-pyhd8ed1ab_0
Linking conda-24.1.2-py310h2ec42d9_0
Linking mamba-1.5.7-py310h6bde348_0
Transaction finished
installation finished.
==> Linking Binary 'conda' to '/usr/local/bin/conda'
==> Linking Binary 'mamba' to '/usr/local/bin/mamba'
🍺  miniforge was successfully installed!
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda create --name tensyflow python=3.8
conda activate tensyflow
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 23.3.1
  latest version: 24.3.0

Please update conda by running

    $ conda update -n base -c conda-forge conda

Or to minimize the number of packages updated during conda update use

     conda install conda=24.3.0



## Package Plan ##

  environment location: /Users/deangladish/miniforge3/envs/tensyflow

  added / updated specs:
    - python=3.8


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    python-3.8.19              |h2469fbe_0_cpython        11.2 MB  conda-forge
    ------------------------------------------------------------
                                           Total:        11.2 MB

The following NEW packages will be INSTALLED:

  bzip2              conda-forge/osx-arm64::bzip2-1.0.8-h93a5062_5
  ca-certificates    conda-forge/osx-arm64::ca-certificates-2024.2.2-hf0a4a13_0
  libffi             conda-forge/osx-arm64::libffi-3.4.2-h3422bc3_5
  libsqlite          conda-forge/osx-arm64::libsqlite-3.45.2-h091b4b1_0
  libzlib            conda-forge/osx-arm64::libzlib-1.2.13-h53f4e23_5
  ncurses            conda-forge/osx-arm64::ncurses-6.4.20240210-h078ce10_0
  openssl            conda-forge/osx-arm64::openssl-3.2.1-h0d3ecfb_1
  pip                conda-forge/noarch::pip-24.0-pyhd8ed1ab_0
  python             conda-forge/osx-arm64::python-3.8.19-h2469fbe_0_cpython
  readline           conda-forge/osx-arm64::readline-8.2-h92ec313_1
  setuptools         conda-forge/noarch::setuptools-69.2.0-pyhd8ed1ab_0
  tk                 conda-forge/osx-arm64::tk-8.6.13-h5083fa2_1
  wheel              conda-forge/noarch::wheel-0.43.0-pyhd8ed1ab_1
  xz                 conda-forge/osx-arm64::xz-5.2.6-h57fd34a_0


Proceed ([y]/n)? y


Downloading and Extracting Packages

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate tensyflow
#
# To deactivate an active environment, use
#
#     $ conda deactivate

(tensyflow) (myprojectenv) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda install -c apple tensorflow-deps
pip install tensorflow-macos # or pip3



####################################
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Convolutional_Neural_Network.py", line 114, in <module>
    import tensorflow_model_optimization as tfmot
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/__init__.py", line 86, in <module>
    from tensorflow_model_optimization.python.core.api import clustering
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/__init__.py", line 16, in <module>
    from tensorflow_model_optimization.python.core.api import clustering
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/clustering/__init__.py", line 16, in <module>
    from tensorflow_model_optimization.python.core.api.clustering import keras
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/api/clustering/keras/__init__.py", line 19, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras.cluster import cluster_scope
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/cluster.py", line 22, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras import cluster_wrapper
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/cluster_wrapper.py", line 23, in <module>
    from tensorflow_model_optimization.python.core.clustering.keras import clustering_centroids
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/clustering/keras/clustering_centroids.py", line 22, in <module>
    from tensorflow_model_optimization.python.core.keras.compat import keras
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/keras/compat.py", line 41, in <module>
    keras = _get_keras_instance()
  File "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/lib/python3.10/site-packages/tensorflow_model_optimization/python/core/keras/compat.py", line 35, in _get_keras_instance
    import tf_keras as keras_internal  # pylint: disable=g-import-not-at-top,unused-import
ModuleNotFoundError: No module named 'tf_keras'
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) xcode-select --install

xcode-select: error: command line tools are already installed, use "Software Update" in System Settings to install updates
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) brew install miniforge

==> Downloading https://formulae.brew.sh/api/formula.jws.json
#=O#- #
==> Downloading https://formulae.brew.sh/api/cask.jws.json
#################################################################################################### 100.0%
==> Caveats
Please run the following to setup your shell:
  conda init "$(basename "${SHELL}")"

==> Downloading https://github.com/conda-forge/miniforge/releases/download/24.1.2-0/Miniforge3-24.1.2-0-Mac
==> Downloading from https://objects.githubusercontent.com/github-production-release-asset-2e65be/221584272
#################################################################################################### 100.0%
==> Installing Cask miniforge
==> Running installer script 'Miniforge3-24.1.2-0-MacOSX-x86_64.sh'
PREFIX=/usr/local/Caskroom/miniforge/base
Unpacking payload ...
Extracting bzip2-1.0.8-h10d778d_5.conda
Extracting c-ares-1.27.0-h10d778d_0.conda
Extracting ca-certificates-2024.2.2-h8857fd0_0.conda
Extracting icu-73.2-hf5e326d_0.conda
Extracting libcxx-16.0.6-hd57cbcb_0.conda
Extracting libev-4.33-h10d778d_2.conda
Extracting libffi-3.4.2-h0d85af4_5.tar.bz2
Extracting libiconv-1.17-hd75f5a5_2.conda
Extracting libzlib-1.2.13-h8a1eda9_5.conda
Extracting lzo-2.10-haf1e3a3_1000.tar.bz2
Extracting ncurses-6.4.20240210-h73e2aa4_0.conda
Extracting pybind11-abi-4-hd8ed1ab_3.tar.bz2
Extracting python_abi-3.10-4_cp310.conda
Extracting reproc-14.2.4.post0-h10d778d_1.conda
Extracting tzdata-2024a-h0c530f3_0.conda
Extracting xz-5.2.6-h775f41a_0.tar.bz2
Extracting fmt-10.2.1-h7728843_0.conda
Extracting libedit-3.1.20191231-h0678c8f_2.tar.bz2
Extracting libsolv-0.7.28-h2d185b6_0.conda
Extracting libsqlite-3.45.2-h92b6c6a_0.conda
Extracting libxml2-2.12.6-hc0ae0f7_0.conda
Extracting lz4-c-1.9.4-hf0c8a7f_0.conda
Extracting openssl-3.2.1-hd75f5a5_1.conda
Extracting readline-8.2-h9e318b2_1.conda
Extracting reproc-cpp-14.2.4.post0-h93d8f39_1.conda
Extracting tk-8.6.13-h1abcd95_1.conda
Extracting yaml-cpp-0.8.0-he965462_0.conda
Extracting zstd-1.5.5-h829000d_0.conda
Extracting krb5-1.21.2-hb884880_0.conda
Extracting libarchive-3.7.2-hd35d340_1.conda
Extracting libnghttp2-1.58.0-h64cf6d3_1.conda
Extracting libssh2-1.11.0-hd019ec5_0.conda
Extracting python-3.10.14-h00d2728_0_cpython.conda
Extracting libcurl-8.6.0-h726d00d_0.conda
Extracting menuinst-2.0.2-py310h2ec42d9_0.conda
Extracting archspec-0.2.3-pyhd8ed1ab_0.conda
Extracting boltons-23.1.1-pyhd8ed1ab_0.conda
Extracting brotli-python-1.1.0-py310h9e9d8ca_1.conda
Extracting certifi-2024.2.2-pyhd8ed1ab_0.conda
Extracting charset-normalizer-3.3.2-pyhd8ed1ab_0.conda
Extracting colorama-0.4.6-pyhd8ed1ab_0.tar.bz2
Extracting distro-1.9.0-pyhd8ed1ab_0.conda
Extracting idna-3.6-pyhd8ed1ab_0.conda
Extracting jsonpointer-2.4-py310h2ec42d9_3.conda
Extracting libmamba-1.5.7-ha449628_0.conda
Extracting packaging-24.0-pyhd8ed1ab_0.conda
Extracting platformdirs-4.2.0-pyhd8ed1ab_0.conda
Extracting pluggy-1.4.0-pyhd8ed1ab_0.conda
Extracting pycosat-0.6.6-py310h6729b98_0.conda
Extracting pycparser-2.21-pyhd8ed1ab_0.tar.bz2
Extracting pysocks-1.7.1-pyha2e5f31_6.tar.bz2
Extracting ruamel.yaml.clib-0.2.8-py310hb372a2b_0.conda
Extracting setuptools-69.2.0-pyhd8ed1ab_0.conda
Extracting truststore-0.8.0-pyhd8ed1ab_0.conda
Extracting wheel-0.43.0-pyhd8ed1ab_0.conda
Extracting cffi-1.16.0-py310hdca579f_0.conda
Extracting jsonpatch-1.33-pyhd8ed1ab_0.conda
Extracting libmambapy-1.5.7-py310hd168405_0.conda
Extracting pip-24.0-pyhd8ed1ab_0.conda
Extracting ruamel.yaml-0.18.6-py310hb372a2b_0.conda
Extracting tqdm-4.66.2-pyhd8ed1ab_0.conda
Extracting urllib3-2.2.1-pyhd8ed1ab_0.conda
Extracting requests-2.31.0-pyhd8ed1ab_0.conda
Extracting zstandard-0.22.0-py310hd88f66e_0.conda
Extracting conda-package-streaming-0.9.0-pyhd8ed1ab_0.conda
Extracting conda-package-handling-2.2.0-pyh38be061_0.conda
Extracting conda-libmamba-solver-24.1.0-pyhd8ed1ab_0.conda
Extracting conda-24.1.2-py310h2ec42d9_0.conda
Extracting mamba-1.5.7-py310h6bde348_0.conda

Installing base environment...


                                           __
          __  ______ ___  ____ _____ ___  / /_  ____ _
         / / / / __ `__ \/ __ `/ __ `__ \/ __ \/ __ `/
        / /_/ / / / / / / /_/ / / / / / / /_/ / /_/ /
       / .___/_/ /_/ /_/\__,_/_/ /_/ /_/_.___/\__,_/
      /_/

Transaction

  Prefix: /usr/local/Caskroom/miniforge/base

  Updating specs:

   - conda-forge/osx-64::bzip2==1.0.8=h10d778d_5[md5=6097a6ca9ada32699b5fc4312dd6ef18]
   - conda-forge/osx-64::c-ares==1.27.0=h10d778d_0[md5=713dd57081dfe8535eb961b45ed26a0c]
   - conda-forge/osx-64::ca-certificates==2024.2.2=h8857fd0_0[md5=f2eacee8c33c43692f1ccfd33d0f50b1]
   - conda-forge/osx-64::icu==73.2=hf5e326d_0[md5=5cc301d759ec03f28328428e28f65591]
   - conda-forge/osx-64::libcxx==16.0.6=hd57cbcb_0[md5=7d6972792161077908b62971802f289a]
   - conda-forge/osx-64::libev==4.33=h10d778d_2[md5=899db79329439820b7e8f8de41bca902]
   - conda-forge/osx-64::libffi==3.4.2=h0d85af4_5[md5=ccb34fb14960ad8b125962d3d79b31a9]
   - conda-forge/osx-64::libiconv==1.17=hd75f5a5_2[md5=6c3628d047e151efba7cf08c5e54d1ca]
   - conda-forge/osx-64::libzlib==1.2.13=h8a1eda9_5[md5=4a3ad23f6e16f99c04e166767193d700]
   - conda-forge/osx-64::lzo==2.10=haf1e3a3_1000[md5=0b6bca372a95d6c602c7a922e928ce79]
   - conda-forge/osx-64::ncurses==6.4.20240210=h73e2aa4_0[md5=50f28c512e9ad78589e3eab34833f762]
   - conda-forge/noarch::pybind11-abi==4=hd8ed1ab_3[md5=878f923dd6acc8aeb47a75da6c4098be]
   - conda-forge/osx-64::python_abi==3.10=4_cp310[md5=b15c816c5a86abcc4d1458dd63aa4c65]
   - conda-forge/osx-64::reproc==14.2.4.post0=h10d778d_1[md5=d7c3258e871481be5bbaf28b4729e29f]
   - conda-forge/noarch::tzdata==2024a=h0c530f3_0[md5=161081fc7cec0bfda0d86d7cb595f8d8]
   - conda-forge/osx-64::xz==5.2.6=h775f41a_0[md5=a72f9d4ea13d55d745ff1ed594747f10]
   - conda-forge/osx-64::fmt==10.2.1=h7728843_0[md5=ab205d53bda43d03f5c5b993ccb406b3]
   - conda-forge/osx-64::libedit==3.1.20191231=h0678c8f_2[md5=6016a8a1d0e63cac3de2c352cd40208b]
   - conda-forge/osx-64::libsolv==0.7.28=h2d185b6_0[md5=a30cb23edd3ef8c8a7a8e83c1bee9295]
   - conda-forge/osx-64::libsqlite==3.45.2=h92b6c6a_0[md5=086f56e13a96a6cfb1bf640505ae6b70]
   - conda-forge/osx-64::libxml2==2.12.6=hc0ae0f7_0[md5=913ce3dbfa8677fba65c44647ef88594]
   - conda-forge/osx-64::lz4-c==1.9.4=hf0c8a7f_0[md5=aa04f7143228308662696ac24023f991]
   - conda-forge/osx-64::openssl==3.2.1=hd75f5a5_1[md5=570a6f04802df580be529f3a72d2bbf7]
   - conda-forge/osx-64::readline==8.2=h9e318b2_1[md5=f17f77f2acf4d344734bda76829ce14e]
   - conda-forge/osx-64::reproc-cpp==14.2.4.post0=h93d8f39_1[md5=a32e95ada0ee860c91e87266700970c3]
   - conda-forge/osx-64::tk==8.6.13=h1abcd95_1[md5=bf830ba5afc507c6232d4ef0fb1a882d]
   - conda-forge/osx-64::yaml-cpp==0.8.0=he965462_0[md5=1bb3addc859ed1338370da6e2996ef47]
   - conda-forge/osx-64::zstd==1.5.5=h829000d_0[md5=80abc41d0c48b82fe0f04e7f42f5cb7e]
   - conda-forge/osx-64::krb5==1.21.2=hb884880_0[md5=80505a68783f01dc8d7308c075261b2f]
   - conda-forge/osx-64::libarchive==3.7.2=hd35d340_1[md5=8c7b79b20a67287a87b39df8a8c8dcc4]
   - conda-forge/osx-64::libnghttp2==1.58.0=h64cf6d3_1[md5=faecc55c2a8155d9ff1c0ff9a0fef64f]
   - conda-forge/osx-64::libssh2==1.11.0=hd019ec5_0[md5=ca3a72efba692c59a90d4b9fc0dfe774]
   - conda-forge/osx-64::python==3.10.14=h00d2728_0_cpython[md5=0a1cddc4382c5c171e791c70740546dd]
   - conda-forge/osx-64::libcurl==8.6.0=h726d00d_0[md5=09569d6e3dc8bef57841f1fc69ea3ea6]
   - conda-forge/osx-64::menuinst==2.0.2=py310h2ec42d9_0[md5=695a6f4401b25418956a90fe6301f50d]
   - conda-forge/noarch::archspec==0.2.3=pyhd8ed1ab_0[md5=192278292e20704f663b9c766909d67b]
   - conda-forge/noarch::boltons==23.1.1=pyhd8ed1ab_0[md5=56febe65315cc388a5d20adf2b39a74d]
   - conda-forge/osx-64::brotli-python==1.1.0=py310h9e9d8ca_1[md5=2362e323293e7699cf1e621d502f86d6]
   - conda-forge/noarch::certifi==2024.2.2=pyhd8ed1ab_0[md5=0876280e409658fc6f9e75d035960333]
   - conda-forge/noarch::charset-normalizer==3.3.2=pyhd8ed1ab_0[md5=7f4a9e3fcff3f6356ae99244a014da6a]
   - conda-forge/noarch::colorama==0.4.6=pyhd8ed1ab_0[md5=3faab06a954c2a04039983f2c4a50d99]
   - conda-forge/noarch::distro==1.9.0=pyhd8ed1ab_0[md5=bbdb409974cd6cb30071b1d978302726]
   - conda-forge/noarch::idna==3.6=pyhd8ed1ab_0[md5=1a76f09108576397c41c0b0c5bd84134]
   - conda-forge/osx-64::jsonpointer==2.4=py310h2ec42d9_3[md5=ca02450dbc1c346a06fc454b36ddab32]
   - conda-forge/osx-64::libmamba==1.5.7=ha449628_0[md5=f5fd8194c6af19225f848a616f53baa7]
   - conda-forge/noarch::packaging==24.0=pyhd8ed1ab_0[md5=248f521b64ce055e7feae3105e7abeb8]
   - conda-forge/noarch::platformdirs==4.2.0=pyhd8ed1ab_0[md5=a0bc3eec34b0fab84be6b2da94e98e20]
   - conda-forge/noarch::pluggy==1.4.0=pyhd8ed1ab_0[md5=139e9feb65187e916162917bb2484976]
   - conda-forge/osx-64::pycosat==0.6.6=py310h6729b98_0[md5=89b601f80d076bf8053eea906293353c]
   - conda-forge/noarch::pycparser==2.21=pyhd8ed1ab_0[md5=076becd9e05608f8dc72757d5f3a91ff]
   - conda-forge/noarch::pysocks==1.7.1=pyha2e5f31_6[md5=2a7de29fb590ca14b5243c4c812c8025]
   - conda-forge/osx-64::ruamel.yaml.clib==0.2.8=py310hb372a2b_0[md5=a6254db88b5bf45d4870c3a63dc39e8d]
   - conda-forge/noarch::setuptools==69.2.0=pyhd8ed1ab_0[md5=da214ecd521a720a9d521c68047682dc]
   - conda-forge/noarch::truststore==0.8.0=pyhd8ed1ab_0[md5=08316d001eca8854392cf2837828ea11]
   - conda-forge/noarch::wheel==0.43.0=pyhd8ed1ab_0[md5=6e43a94e0ee67523b5e781f0faba5c45]
   - conda-forge/osx-64::cffi==1.16.0=py310hdca579f_0[md5=b9e6213f0eb91f40c009ce69139c1869]
   - conda-forge/noarch::jsonpatch==1.33=pyhd8ed1ab_0[md5=bfdb7c5c6ad1077c82a69a8642c87aff]
   - conda-forge/osx-64::libmambapy==1.5.7=py310hd168405_0[md5=f981f435165e3053d9aa29603958962b]
   - conda-forge/noarch::pip==24.0=pyhd8ed1ab_0[md5=f586ac1e56c8638b64f9c8122a7b8a67]
   - conda-forge/osx-64::ruamel.yaml==0.18.6=py310hb372a2b_0[md5=a6691c80f3bf62bc0df37b87caac6a70]
   - conda-forge/noarch::tqdm==4.66.2=pyhd8ed1ab_0[md5=2b8dfb969f984497f3f98409a9545776]
   - conda-forge/noarch::urllib3==2.2.1=pyhd8ed1ab_0[md5=08807a87fa7af10754d46f63b368e016]
   - conda-forge/noarch::requests==2.31.0=pyhd8ed1ab_0[md5=a30144e4156cdbb236f99ebb49828f8b]
   - conda-forge/osx-64::zstandard==0.22.0=py310hd88f66e_0[md5=88c991558201cae2b7e690c2e9d2e618]
   - conda-forge/noarch::conda-package-streaming==0.9.0=pyhd8ed1ab_0[md5=38253361efb303deead3eab39ae9269b]
   - conda-forge/noarch::conda-package-handling==2.2.0=pyh38be061_0[md5=8a3ae7f6318376aa08ea753367bb7dd6]
   - conda-forge/noarch::conda-libmamba-solver==24.1.0=pyhd8ed1ab_0[md5=304dc78ad6e52e0fd663df1d484c1531]
   - conda-forge/osx-64::conda==24.1.2=py310h2ec42d9_0[md5=c60e9dbe2f9a13f4b6d3521f46c02ef8]
   - conda-forge/osx-64::mamba==1.5.7=py310h6bde348_0[md5=2d0208c01f24601e7671487d81afc2c1]


  Package                         Version  Build               Channel           Size
───────────────────────────────────────────────────────────────────────────────────────
  Install:
───────────────────────────────────────────────────────────────────────────────────────

  + archspec                        0.2.3  pyhd8ed1ab_0        conda-forge     Cached
  + boltons                        23.1.1  pyhd8ed1ab_0        conda-forge     Cached
  + brotli-python                   1.1.0  py310h9e9d8ca_1     conda-forge     Cached
  + bzip2                           1.0.8  h10d778d_5          conda-forge     Cached
  + c-ares                         1.27.0  h10d778d_0          conda-forge     Cached
  + ca-certificates              2024.2.2  h8857fd0_0          conda-forge     Cached
  + certifi                      2024.2.2  pyhd8ed1ab_0        conda-forge     Cached
  + cffi                           1.16.0  py310hdca579f_0     conda-forge     Cached
  + charset-normalizer              3.3.2  pyhd8ed1ab_0        conda-forge     Cached
  + colorama                        0.4.6  pyhd8ed1ab_0        conda-forge     Cached
  + conda                          24.1.2  py310h2ec42d9_0     conda-forge     Cached
  + conda-libmamba-solver          24.1.0  pyhd8ed1ab_0        conda-forge     Cached
  + conda-package-handling          2.2.0  pyh38be061_0        conda-forge     Cached
  + conda-package-streaming         0.9.0  pyhd8ed1ab_0        conda-forge     Cached
  + distro                          1.9.0  pyhd8ed1ab_0        conda-forge     Cached
  + fmt                            10.2.1  h7728843_0          conda-forge     Cached
  + icu                              73.2  hf5e326d_0          conda-forge     Cached
  + idna                              3.6  pyhd8ed1ab_0        conda-forge     Cached
  + jsonpatch                        1.33  pyhd8ed1ab_0        conda-forge     Cached
  + jsonpointer                       2.4  py310h2ec42d9_3     conda-forge     Cached
  + krb5                           1.21.2  hb884880_0          conda-forge     Cached
  + libarchive                      3.7.2  hd35d340_1          conda-forge     Cached
  + libcurl                         8.6.0  h726d00d_0          conda-forge     Cached
  + libcxx                         16.0.6  hd57cbcb_0          conda-forge     Cached
  + libedit                  3.1.20191231  h0678c8f_2          conda-forge     Cached
  + libev                            4.33  h10d778d_2          conda-forge     Cached
  + libffi                          3.4.2  h0d85af4_5          conda-forge     Cached
  + libiconv                         1.17  hd75f5a5_2          conda-forge     Cached
  + libmamba                        1.5.7  ha449628_0          conda-forge     Cached
  + libmambapy                      1.5.7  py310hd168405_0     conda-forge     Cached
  + libnghttp2                     1.58.0  h64cf6d3_1          conda-forge     Cached
  + libsolv                        0.7.28  h2d185b6_0          conda-forge     Cached
  + libsqlite                      3.45.2  h92b6c6a_0          conda-forge     Cached
  + libssh2                        1.11.0  hd019ec5_0          conda-forge     Cached
  + libxml2                        2.12.6  hc0ae0f7_0          conda-forge     Cached
  + libzlib                        1.2.13  h8a1eda9_5          conda-forge     Cached
  + lz4-c                           1.9.4  hf0c8a7f_0          conda-forge     Cached
  + lzo                              2.10  haf1e3a3_1000       conda-forge     Cached
  + mamba                           1.5.7  py310h6bde348_0     conda-forge     Cached
  + menuinst                        2.0.2  py310h2ec42d9_0     conda-forge     Cached
  + ncurses                  6.4.20240210  h73e2aa4_0          conda-forge     Cached
  + openssl                         3.2.1  hd75f5a5_1          conda-forge     Cached
  + packaging                        24.0  pyhd8ed1ab_0        conda-forge     Cached
  + pip                              24.0  pyhd8ed1ab_0        conda-forge     Cached
  + platformdirs                    4.2.0  pyhd8ed1ab_0        conda-forge     Cached
  + pluggy                          1.4.0  pyhd8ed1ab_0        conda-forge     Cached
  + pybind11-abi                        4  hd8ed1ab_3          conda-forge     Cached
  + pycosat                         0.6.6  py310h6729b98_0     conda-forge     Cached
  + pycparser                        2.21  pyhd8ed1ab_0        conda-forge     Cached
  + pysocks                         1.7.1  pyha2e5f31_6        conda-forge     Cached
  + python                        3.10.14  h00d2728_0_cpython  conda-forge     Cached
  + python_abi                       3.10  4_cp310             conda-forge     Cached
  + readline                          8.2  h9e318b2_1          conda-forge     Cached
  + reproc                   14.2.4.post0  h10d778d_1          conda-forge     Cached
  + reproc-cpp               14.2.4.post0  h93d8f39_1          conda-forge     Cached
  + requests                       2.31.0  pyhd8ed1ab_0        conda-forge     Cached
  + ruamel.yaml                    0.18.6  py310hb372a2b_0     conda-forge     Cached
  + ruamel.yaml.clib                0.2.8  py310hb372a2b_0     conda-forge     Cached
  + setuptools                     69.2.0  pyhd8ed1ab_0        conda-forge     Cached
  + tk                             8.6.13  h1abcd95_1          conda-forge     Cached
  + tqdm                           4.66.2  pyhd8ed1ab_0        conda-forge     Cached
  + truststore                      0.8.0  pyhd8ed1ab_0        conda-forge     Cached
  + tzdata                          2024a  h0c530f3_0          conda-forge     Cached
  + urllib3                         2.2.1  pyhd8ed1ab_0        conda-forge     Cached
  + wheel                          0.43.0  pyhd8ed1ab_0        conda-forge     Cached
  + xz                              5.2.6  h775f41a_0          conda-forge     Cached
  + yaml-cpp                        0.8.0  he965462_0          conda-forge     Cached
  + zstandard                      0.22.0  py310hd88f66e_0     conda-forge     Cached
  + zstd                            1.5.5  h829000d_0          conda-forge     Cached

  Summary:

  Install: 69 packages

  Total download: 0 B

───────────────────────────────────────────────────────────────────────────────────────



Transaction starting
Linking bzip2-1.0.8-h10d778d_5
Linking c-ares-1.27.0-h10d778d_0
Linking ca-certificates-2024.2.2-h8857fd0_0
Linking icu-73.2-hf5e326d_0
Linking libcxx-16.0.6-hd57cbcb_0
Linking libev-4.33-h10d778d_2
Linking libffi-3.4.2-h0d85af4_5
Linking libiconv-1.17-hd75f5a5_2
Linking libzlib-1.2.13-h8a1eda9_5
Linking lzo-2.10-haf1e3a3_1000
Linking ncurses-6.4.20240210-h73e2aa4_0
Linking pybind11-abi-4-hd8ed1ab_3
Linking python_abi-3.10-4_cp310
Linking reproc-14.2.4.post0-h10d778d_1
Linking tzdata-2024a-h0c530f3_0
Linking xz-5.2.6-h775f41a_0
Linking fmt-10.2.1-h7728843_0
Linking libedit-3.1.20191231-h0678c8f_2
Linking libsolv-0.7.28-h2d185b6_0
Linking libsqlite-3.45.2-h92b6c6a_0
Linking libxml2-2.12.6-hc0ae0f7_0
Linking lz4-c-1.9.4-hf0c8a7f_0
Linking openssl-3.2.1-hd75f5a5_1
Linking readline-8.2-h9e318b2_1
Linking reproc-cpp-14.2.4.post0-h93d8f39_1
Linking tk-8.6.13-h1abcd95_1
Linking yaml-cpp-0.8.0-he965462_0
Linking zstd-1.5.5-h829000d_0
Linking krb5-1.21.2-hb884880_0
Linking libarchive-3.7.2-hd35d340_1
Linking libnghttp2-1.58.0-h64cf6d3_1
Linking libssh2-1.11.0-hd019ec5_0
Linking python-3.10.14-h00d2728_0_cpython
Linking libcurl-8.6.0-h726d00d_0
Linking menuinst-2.0.2-py310h2ec42d9_0
Linking archspec-0.2.3-pyhd8ed1ab_0
Linking boltons-23.1.1-pyhd8ed1ab_0
Linking brotli-python-1.1.0-py310h9e9d8ca_1
Linking certifi-2024.2.2-pyhd8ed1ab_0
Linking charset-normalizer-3.3.2-pyhd8ed1ab_0
Linking colorama-0.4.6-pyhd8ed1ab_0
Linking distro-1.9.0-pyhd8ed1ab_0
Linking idna-3.6-pyhd8ed1ab_0
Linking jsonpointer-2.4-py310h2ec42d9_3
Linking libmamba-1.5.7-ha449628_0
Linking packaging-24.0-pyhd8ed1ab_0
Linking platformdirs-4.2.0-pyhd8ed1ab_0
Linking pluggy-1.4.0-pyhd8ed1ab_0
Linking pycosat-0.6.6-py310h6729b98_0
Linking pycparser-2.21-pyhd8ed1ab_0
Linking pysocks-1.7.1-pyha2e5f31_6
Linking ruamel.yaml.clib-0.2.8-py310hb372a2b_0
Linking setuptools-69.2.0-pyhd8ed1ab_0
Linking truststore-0.8.0-pyhd8ed1ab_0
Linking wheel-0.43.0-pyhd8ed1ab_0
Linking cffi-1.16.0-py310hdca579f_0
Linking jsonpatch-1.33-pyhd8ed1ab_0
Linking libmambapy-1.5.7-py310hd168405_0
Linking pip-24.0-pyhd8ed1ab_0
Linking ruamel.yaml-0.18.6-py310hb372a2b_0
Linking tqdm-4.66.2-pyhd8ed1ab_0
Linking urllib3-2.2.1-pyhd8ed1ab_0
Linking requests-2.31.0-pyhd8ed1ab_0
Linking zstandard-0.22.0-py310hd88f66e_0
Linking conda-package-streaming-0.9.0-pyhd8ed1ab_0
Linking conda-package-handling-2.2.0-pyh38be061_0
Linking conda-libmamba-solver-24.1.0-pyhd8ed1ab_0
Linking conda-24.1.2-py310h2ec42d9_0
Linking mamba-1.5.7-py310h6bde348_0
Transaction finished
installation finished.
==> Linking Binary 'conda' to '/usr/local/bin/conda'
==> Linking Binary 'mamba' to '/usr/local/bin/mamba'
🍺  miniforge was successfully installed!
(myprojectenv) (base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda create --name tensyflow python=3.8
conda activate tensyflow
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 23.3.1
  latest version: 24.3.0

Please update conda by running

    $ conda update -n base -c conda-forge conda

Or to minimize the number of packages updated during conda update use

     conda install conda=24.3.0



## Package Plan ##

  environment location: /Users/deangladish/miniforge3/envs/tensyflow

  added / updated specs:
    - python=3.8


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    python-3.8.19              |h2469fbe_0_cpython        11.2 MB  conda-forge
    ------------------------------------------------------------
                                           Total:        11.2 MB

The following NEW packages will be INSTALLED:

  bzip2              conda-forge/osx-arm64::bzip2-1.0.8-h93a5062_5
  ca-certificates    conda-forge/osx-arm64::ca-certificates-2024.2.2-hf0a4a13_0
  libffi             conda-forge/osx-arm64::libffi-3.4.2-h3422bc3_5
  libsqlite          conda-forge/osx-arm64::libsqlite-3.45.2-h091b4b1_0
  libzlib            conda-forge/osx-arm64::libzlib-1.2.13-h53f4e23_5
  ncurses            conda-forge/osx-arm64::ncurses-6.4.20240210-h078ce10_0
  openssl            conda-forge/osx-arm64::openssl-3.2.1-h0d3ecfb_1
  pip                conda-forge/noarch::pip-24.0-pyhd8ed1ab_0
  python             conda-forge/osx-arm64::python-3.8.19-h2469fbe_0_cpython
  readline           conda-forge/osx-arm64::readline-8.2-h92ec313_1
  setuptools         conda-forge/noarch::setuptools-69.2.0-pyhd8ed1ab_0
  tk                 conda-forge/osx-arm64::tk-8.6.13-h5083fa2_1
  wheel              conda-forge/noarch::wheel-0.43.0-pyhd8ed1ab_1
  xz                 conda-forge/osx-arm64::xz-5.2.6-h57fd34a_0


Proceed ([y]/n)? y


Downloading and Extracting Packages

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate tensyflow
#
# To deactivate an active environment, use
#
#     $ conda deactivate

(tensyflow) (myprojectenv) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda install -c apple tensorflow-deps
pip install tensorflow-macos # or pip3
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Solving environment: failed with repodata from current_repodata.json, will retry with next repodata source.
Collecting package metadata (repodata.json): \ WARNING conda.models.version:get_matcher(546): Using .* with relational operator is superfluous and deprecated and will be removed in a future version of conda. Your spec was 1.8.0.*, but conda is ignoring the .* and treating it as 1.8.0
WARNING conda.models.version:get_matcher(546): Using .* with relational operator is superfluous and deprecated and will be removed in a future version of conda. Your spec was 1.9.0.*, but conda is ignoring the .* and treating it as 1.9.0
done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 23.3.1
  latest version: 24.3.0

Please update conda by running

    $ conda update -n base -c conda-forge conda

Or to minimize the number of packages updated during conda update use

     conda install conda=24.3.0



## Package Plan ##

  environment location: /Users/deangladish/miniforge3/envs/tensyflow

  added / updated specs:
    - tensorflow-deps


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    grpcio-1.46.3              |   py38h1ef021a_0         2.3 MB  conda-forge
    h5py-3.6.0                 |nompi_py38hacf61ce_100         1.1 MB  conda-forge
    hdf5-1.12.1                |nompi_hd9dbc9e_104         3.2 MB  conda-forge
    libprotobuf-3.19.6         |       hb5ab8b9_0         1.8 MB  conda-forge
    numpy-1.23.2               |   py38h579d673_0         5.9 MB  conda-forge
    protobuf-3.19.6            |   py38h2b1e499_0         280 KB  conda-forge
    python_abi-3.8             |           4_cp38           6 KB  conda-forge
    tensorflow-deps-2.10.0     |                0           2 KB  apple
    ------------------------------------------------------------
                                           Total:        14.5 MB

The following NEW packages will be INSTALLED:

  c-ares             conda-forge/osx-arm64::c-ares-1.28.1-h93a5062_0
  cached-property    conda-forge/noarch::cached-property-1.5.2-hd8ed1ab_1
  cached_property    conda-forge/noarch::cached_property-1.5.2-pyha770c72_1
  grpcio             conda-forge/osx-arm64::grpcio-1.46.3-py38h1ef021a_0
  h5py               conda-forge/osx-arm64::h5py-3.6.0-nompi_py38hacf61ce_100
  hdf5               conda-forge/osx-arm64::hdf5-1.12.1-nompi_hd9dbc9e_104
  krb5               conda-forge/osx-arm64::krb5-1.21.2-h92f50d5_0
  libblas            conda-forge/osx-arm64::libblas-3.9.0-22_osxarm64_openblas
  libcblas           conda-forge/osx-arm64::libcblas-3.9.0-22_osxarm64_openblas
  libcurl            conda-forge/osx-arm64::libcurl-8.7.1-h2d989ff_0
  libcxx             conda-forge/osx-arm64::libcxx-16.0.6-h4653b0c_0
  libedit            conda-forge/osx-arm64::libedit-3.1.20191231-hc8eb9b7_2
  libev              conda-forge/osx-arm64::libev-4.33-h93a5062_2
  libgfortran        conda-forge/osx-arm64::libgfortran-5.0.0-13_2_0_hd922786_3
  libgfortran5       conda-forge/osx-arm64::libgfortran5-13.2.0-hf226fd6_3
  liblapack          conda-forge/osx-arm64::liblapack-3.9.0-22_osxarm64_openblas
  libnghttp2         conda-forge/osx-arm64::libnghttp2-1.58.0-ha4dd798_1
  libopenblas        conda-forge/osx-arm64::libopenblas-0.3.27-openmp_h6c19121_0
  libprotobuf        conda-forge/osx-arm64::libprotobuf-3.19.6-hb5ab8b9_0
  libssh2            conda-forge/osx-arm64::libssh2-1.11.0-h7a5bd25_0
  llvm-openmp        conda-forge/osx-arm64::llvm-openmp-18.1.2-hcd81f8e_0
  numpy              conda-forge/osx-arm64::numpy-1.23.2-py38h579d673_0
  protobuf           conda-forge/osx-arm64::protobuf-3.19.6-py38h2b1e499_0
  python_abi         conda-forge/osx-arm64::python_abi-3.8-4_cp38
  six                conda-forge/noarch::six-1.16.0-pyh6c4a22f_0
  tensorflow-deps    apple/osx-arm64::tensorflow-deps-2.10.0-0
  zlib               conda-forge/osx-arm64::zlib-1.2.13-h53f4e23_5
  zstd               conda-forge/osx-arm64::zstd-1.5.5-h4f39d0f_0


Proceed ([y]/n)? y


Downloading and Extracting Packages

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
Requirement already satisfied: tensorflow-macos in ./myprojectenv/lib/python3.10/site-packages (2.16.1)
Requirement already satisfied: tensorflow==2.16.1 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow-macos) (2.16.1)
Requirement already satisfied: absl-py>=1.0.0 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (1.4.0)
Requirement already satisfied: astunparse>=1.6.0 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (1.6.3)
Requirement already satisfied: flatbuffers>=23.5.26 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (24.3.25)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (0.5.4)
Requirement already satisfied: google-pasta>=0.1.1 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (0.2.0)
Requirement already satisfied: h5py>=3.10.0 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (3.10.0)
Requirement already satisfied: libclang>=13.0.0 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (18.1.1)
Requirement already satisfied: ml-dtypes~=0.3.1 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (0.3.2)
Requirement already satisfied: opt-einsum>=2.3.2 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (3.3.0)
Requirement already satisfied: packaging in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (24.0)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (4.25.3)
Requirement already satisfied: requests<3,>=2.21.0 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (2.31.0)
Requirement already satisfied: setuptools in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (63.2.0)
Requirement already satisfied: six>=1.12.0 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (2.4.0)
Requirement already satisfied: typing-extensions>=3.6.6 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (4.11.0)
Requirement already satisfied: wrapt>=1.11.0 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (1.16.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (1.62.1)
Requirement already satisfied: tensorboard<2.17,>=2.16 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (2.16.2)
Requirement already satisfied: keras>=3.0.0 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (3.1.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (0.36.0)
Requirement already satisfied: numpy<2.0.0,>=1.23.5 in ./myprojectenv/lib/python3.10/site-packages (from tensorflow==2.16.1->tensorflow-macos) (1.26.4)
Requirement already satisfied: wheel<1.0,>=0.23.0 in ./myprojectenv/lib/python3.10/site-packages (from astunparse>=1.6.0->tensorflow==2.16.1->tensorflow-macos) (0.43.0)
Requirement already satisfied: rich in ./myprojectenv/lib/python3.10/site-packages (from keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos) (13.7.1)
Requirement already satisfied: namex in ./myprojectenv/lib/python3.10/site-packages (from keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos) (0.0.7)
Requirement already satisfied: optree in ./myprojectenv/lib/python3.10/site-packages (from keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos) (0.11.0)
Requirement already satisfied: charset-normalizer<4,>=2 in ./myprojectenv/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow==2.16.1->tensorflow-macos) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in ./myprojectenv/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow==2.16.1->tensorflow-macos) (3.6)
Requirement already satisfied: urllib3<3,>=1.21.1 in ./myprojectenv/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow==2.16.1->tensorflow-macos) (2.2.1)
Requirement already satisfied: certifi>=2017.4.17 in ./myprojectenv/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow==2.16.1->tensorflow-macos) (2024.2.2)
Requirement already satisfied: markdown>=2.6.8 in ./myprojectenv/lib/python3.10/site-packages (from tensorboard<2.17,>=2.16->tensorflow==2.16.1->tensorflow-macos) (3.6)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in ./myprojectenv/lib/python3.10/site-packages (from tensorboard<2.17,>=2.16->tensorflow==2.16.1->tensorflow-macos) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in ./myprojectenv/lib/python3.10/site-packages (from tensorboard<2.17,>=2.16->tensorflow==2.16.1->tensorflow-macos) (3.0.2)
Requirement already satisfied: MarkupSafe>=2.1.1 in ./myprojectenv/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard<2.17,>=2.16->tensorflow==2.16.1->tensorflow-macos) (2.1.5)
Requirement already satisfied: markdown-it-py>=2.2.0 in ./myprojectenv/lib/python3.10/site-packages (from rich->keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in ./myprojectenv/lib/python3.10/site-packages (from rich->keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos) (2.17.2)
Requirement already satisfied: mdurl~=0.1 in ./myprojectenv/lib/python3.10/site-packages (from markdown-it-py>=2.2.0->rich->keras>=3.0.0->tensorflow==2.16.1->tensorflow-macos) (0.1.2)
(tensyflow) (myprojectenv) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)


conda init


(base) ~/CS-7643-O01/Group_Project (main ✗) conda activate tensyflow
(tensyflow) ~/CS-7643-O01/Group_Project (main ✗) python Conv
python: can't open file 'Conv': [Errno 2] No such file or directory
(tensyflow) ~/CS-7643-O01/Group_Project (main ✗) cd Data\ Zenodo
(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "Convolutional_Neural_Network.py", line 106, in <module>
    import librosa
ModuleNotFoundError: No module named 'librosa'
(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)









Install XCode CLI tools

xcode-select --install
Install Miniforge. I prefer installing using brew
brew install miniforge
Create a conda environment and activate it
conda create --name tensyflow python=3.8
conda activate tensyflow
Install Tensorflow-MacOS
conda install -c apple tensorflow-deps
pip install tensorflow-macos # or pip3
Share
Edit
https://stackoverflow.com/questions/71516530/installing-keras-tensorflow2-on-macbook-air-with-apple-m1-chip









h5py -> hdf5[version='>=1.14.3,<1.14.4.0a0'] -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.1.0|>=11.2.0|>=11.3.0|>=12.2.0|>=12.3.0|>=13.2.0']
matplotlib-base -> numpy[version='>=1.21,<2'] -> libgfortran5[version='>=11.1.0']
numba -> numpy[version='>=1.20.3,<2.0a0'] -> libgfortran5[version='>=11.1.0']
libopenblas -> libgfortran=5 -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|13.2.0|13.2.0|12.3.0',build='h76267eb_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|hf226fd6_1|ha3a6a3e_1']
libcblas -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.1.0']
libcblas -> libblas==3.9.0=22_osxarm64_accelerate -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|>=12.2.0|>=12.3.0|13.2.0|13.2.0|12.3.0',build='h76267eb_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|hf226fd6_1|ha3a6a3e_1']
libopenblas -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.3.0|>=12.3.0|>=11.2.0|>=11.1.0']
openblas -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.3.0|>=12.3.0']
liblapack -> libblas==3.9.0=22_osxarm64_accelerate -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|>=12.2.0|>=12.3.0|13.2.0|13.2.0|12.3.0',build='h76267eb_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|hf226fd6_1|ha3a6a3e_1']
liblapack -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.1.0']
blas-devel -> libblas==3.9.0=22_osxarm64_accelerate -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.3.0|>=12.2.0|>=12.3.0|>=11.1.0']
scikit-learn -> scipy -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.3.0|>=12.2.0|>=12.3.0|>=13.2.0|>=11.2.0|>=11.1.0']
liblapacke -> libblas==3.9.0=22_osxarm64_accelerate -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|>=12.2.0|>=12.3.0|13.2.0|13.2.0|12.3.0',build='h76267eb_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|hf226fd6_1|ha3a6a3e_1']
openblas -> libgfortran=5 -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|13.2.0|13.2.0|12.3.0|>=11.2.0|>=11.1.0',build='h76267eb_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|hf226fd6_1|ha3a6a3e_1']
blas -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=12.2.0|>=12.3.0|>=13.2.0']
scipy -> libgfortran=5 -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|13.2.0|13.2.0|12.3.0',build='h76267eb_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|hf226fd6_1|ha3a6a3e_1']
numpy -> libblas[version='>=3.9.0,<4.0a0'] -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|>=11.0.0.dev0|>=11.0.1.dev0|>=12.2.0|>=12.3.0|>=13.2.0|>=11.3.0|>=11.2.0|13.2.0|13.2.0|12.3.0',build='h76267eb_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|hf226fd6_1|ha3a6a3e_1']
blas -> libgfortran=5 -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|13.2.0|13.2.0|12.3.0|>=11.1.0',build='h76267eb_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|hf226fd6_1|ha3a6a3e_1']
scipy -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.3.0|>=12.2.0|>=12.3.0|>=13.2.0|>=11.2.0|>=11.1.0']
libblas -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=12.2.0|>=12.3.0|>=11.1.0']
hdf5 -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.3.0|>=12.2.0|>=12.3.0|>=13.2.0|>=11.1.0']
pysoundfile -> numpy -> libgfortran5[version='>=11.1.0']
libgfortran -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|13.2.0|13.2.0|12.3.0',build='h76267eb_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|hf226fd6_1|ha3a6a3e_1']
numpy -> libgfortran5[version='>=11.1.0']
liblapacke -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.1.0']
librosa -> numpy[version='>=1.20.3,!=1.22.0,!=1.22.1,!=1.22.2'] -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.1.0|>=12.3.0|>=13.2.0|>=12.2.0|>=11.3.0|>=11.2.0']
libblas -> libgfortran=5 -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|13.2.0|13.2.0|12.3.0|>=11.3.0|>=11.2.0',build='h76267eb_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|hf226fd6_1|ha3a6a3e_1']

Package libxml2 conflicts for:
ffmpeg -> libxml2[version='>=2.10.3,<3.0.0a0|>=2.10.4,<3.0.0a0|>=2.11.3,<3.0.0a0|>=2.11.4,<3.0.0a0|>=2.11.5,<3.0.0a0|>=2.11.6,<3.0.0a0|>=2.12.1,<3.0.0a0|>=2.12.2,<3.0.0a0|>=2.12.3,<3.0.0a0|>=2.12.4,<3.0a0|>=2.12.5,<3.0a0|>=2.12.6,<3.0a0|>=2.9.14,<3.0.0a0|>=2.9.13,<3.0.0a0|>=2.9.12,<3.0.0a0']
libflac -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
libidn2 -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
audioread -> ffmpeg -> libxml2[version='>=2.10.3,<3.0.0a0|>=2.10.4,<3.0.0a0|>=2.11.3,<3.0.0a0|>=2.11.4,<3.0.0a0|>=2.11.5,<3.0.0a0|>=2.11.6,<3.0.0a0|>=2.12.1,<3.0.0a0|>=2.12.2,<3.0.0a0|>=2.12.3,<3.0.0a0|>=2.12.4,<3.0a0|>=2.12.5,<3.0a0|>=2.12.6,<3.0a0|>=2.9.14,<3.0.0a0|>=2.9.13,<3.0.0a0|>=2.9.12,<3.0.0a0']
gettext -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
libglib -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
ffmpeg -> fontconfig[version='>=2.14.1,<3.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.14,<2.10.0a0']
cairo -> fontconfig[version='>=2.13.96,<3.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.12,<3.0.0a0|>=2.9.14,<2.10.0a0|>=2.9.10,<3.0.0a0|>=2.9.10,<2.10.0a0']
fontconfig -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<3.0.0a0|>=2.9.12,<3.0.0a0|>=2.9.14,<2.10.0a0|>=2.9.10,<2.10.0a0']
gnutls -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']

Package libopenblas conflicts for:
blas-devel -> libblas==3.9.0=22_osxarm64_openblas -> libopenblas[version='0.3.11|0.3.12|0.3.12|0.3.13|0.3.15|0.3.15|0.3.16|0.3.17|0.3.17|0.3.18|0.3.18|0.3.20|0.3.20|0.3.20|0.3.21|0.3.21|0.3.21|0.3.21|0.3.21|0.3.23|0.3.24|0.3.25|0.3.26|0.3.27|>=0.3.27,<0.3.28.0a0|>=0.3.27,<1.0a0|>=0.3.26,<0.3.27.0a0|>=0.3.26,<1.0a0|>=0.3.25,<0.3.26.0a0|>=0.3.25,<1.0a0|>=0.3.24,<0.3.25.0a0|>=0.3.24,<1.0a0|>=0.3.23,<0.3.24.0a0|>=0.3.23,<1.0a0|>=0.3.21,<0.3.22.0a0|>=0.3.21,<1.0a0|>=0.3.20,<0.3.21.0a0|>=0.3.20,<1.0a0|>=0.3.18,<0.3.19.0a0|>=0.3.18,<1.0a0|0.3.17|0.3.17|0.3.13|>=0.3.17,<0.3.18.0a0|>=0.3.17,<1.0a0|>=0.3.15,<0.3.16.0a0|>=0.3.15,<1.0a0|>=0.3.12,<0.3.13.0a0|>=0.3.12,<1.0a0',build='hcab743c_2|openmp_h74fab25_1|openmp_h74fab25_0|openmp_h2ecc587_1|openmp_hf330de4_1|openmp_hf330de4_0|openmp_hf330de4_0|openmp_h5dd58f0_1|openmp_h2209c59_0|openmp_h130de29_1|h269037a_0|openmp_hd76b1f2_0|openmp_h6c19121_0|openmp_hc731615_0|openmp_hc731615_3|openmp_hc731615_2|openmp_hcb59c3b_1|openmp_hcb59c3b_0|hea475bc_0|openmp_h5dd58f0_0|hea475bc_0|openmp_hf330de4_0|openmp_he0e8823_0|h9886b1c_1|h9886b1c_0']
numpy -> libblas[version='>=3.9.0,<4.0a0'] -> libopenblas[version='>=0.3.12,<0.3.13.0a0|>=0.3.15,<0.3.16.0a0|>=0.3.17,<0.3.18.0a0|>=0.3.18,<0.3.19.0a0|>=0.3.20,<0.3.21.0a0|>=0.3.21,<0.3.22.0a0|>=0.3.23,<0.3.24.0a0|>=0.3.24,<0.3.25.0a0|>=0.3.25,<0.3.26.0a0|>=0.3.26,<0.3.27.0a0|>=0.3.27,<0.3.28.0a0|>=0.3.27,<1.0a0|>=0.3.26,<1.0a0|>=0.3.25,<1.0a0|>=0.3.24,<1.0a0|>=0.3.23,<1.0a0|>=0.3.18,<1.0a0|>=0.3.15,<1.0a0|>=0.3.12,<1.0a0']
scipy -> libopenblas[version='>=0.3.17,<1.0a0|>=0.3.20,<1.0a0|>=0.3.21,<1.0a0']
liblapacke -> libblas==3.9.0=22_osxarm64_openblas -> libopenblas[version='>=0.3.12,<0.3.13.0a0|>=0.3.15,<0.3.16.0a0|>=0.3.17,<0.3.18.0a0|>=0.3.18,<0.3.19.0a0|>=0.3.20,<0.3.21.0a0|>=0.3.21,<0.3.22.0a0|>=0.3.23,<0.3.24.0a0|>=0.3.24,<0.3.25.0a0|>=0.3.25,<0.3.26.0a0|>=0.3.26,<0.3.27.0a0|>=0.3.27,<0.3.28.0a0|>=0.3.27,<1.0a0|>=0.3.26,<1.0a0|>=0.3.25,<1.0a0|>=0.3.24,<1.0a0|>=0.3.23,<1.0a0|>=0.3.21,<1.0a0|>=0.3.20,<1.0a0|>=0.3.18,<1.0a0|>=0.3.17,<1.0a0|>=0.3.15,<1.0a0|>=0.3.12,<1.0a0']
scipy -> libblas[version='>=3.9.0,<4.0a0'] -> libopenblas[version='>=0.3.12,<0.3.13.0a0|>=0.3.15,<0.3.16.0a0|>=0.3.17,<0.3.18.0a0|>=0.3.18,<0.3.19.0a0|>=0.3.20,<0.3.21.0a0|>=0.3.21,<0.3.22.0a0|>=0.3.23,<0.3.24.0a0|>=0.3.24,<0.3.25.0a0|>=0.3.25,<0.3.26.0a0|>=0.3.26,<0.3.27.0a0|>=0.3.27,<0.3.28.0a0|>=0.3.27,<1.0a0|>=0.3.26,<1.0a0|>=0.3.25,<1.0a0|>=0.3.24,<1.0a0|>=0.3.23,<1.0a0|>=0.3.18,<1.0a0|>=0.3.15,<1.0a0|>=0.3.12,<1.0a0']
scikit-learn -> numpy[version='>=1.23.5,<2.0a0'] -> libopenblas[version='>=0.3.17,<1.0a0|>=0.3.20,<1.0a0|>=0.3.21,<1.0a0']
soxr-python -> numpy[version='>=1.23.5,<2.0a0'] -> libopenblas[version='>=0.3.17,<1.0a0|>=0.3.20,<1.0a0|>=0.3.21,<1.0a0']
liblapack -> libblas==3.9.0=22_osxarm64_openblas -> libopenblas[version='>=0.3.12,<0.3.13.0a0|>=0.3.15,<0.3.16.0a0|>=0.3.17,<0.3.18.0a0|>=0.3.18,<0.3.19.0a0|>=0.3.20,<0.3.21.0a0|>=0.3.21,<0.3.22.0a0|>=0.3.23,<0.3.24.0a0|>=0.3.24,<0.3.25.0a0|>=0.3.25,<0.3.26.0a0|>=0.3.26,<0.3.27.0a0|>=0.3.27,<0.3.28.0a0|>=0.3.27,<1.0a0|>=0.3.26,<1.0a0|>=0.3.25,<1.0a0|>=0.3.24,<1.0a0|>=0.3.23,<1.0a0|>=0.3.21,<1.0a0|>=0.3.20,<1.0a0|>=0.3.18,<1.0a0|>=0.3.17,<1.0a0|>=0.3.15,<1.0a0|>=0.3.12,<1.0a0']
matplotlib-base -> numpy[version='>=1.21,<2'] -> libopenblas[version='>=0.3.17,<1.0a0|>=0.3.20,<1.0a0|>=0.3.21,<1.0a0']
numpy -> libopenblas[version='>=0.3.17,<1.0a0|>=0.3.20,<1.0a0|>=0.3.21,<1.0a0']
libblas -> libopenblas[version='>=0.3.12,<0.3.13.0a0|>=0.3.15,<0.3.16.0a0|>=0.3.17,<0.3.18.0a0|>=0.3.18,<0.3.19.0a0|>=0.3.20,<0.3.21.0a0|>=0.3.21,<0.3.22.0a0|>=0.3.23,<0.3.24.0a0|>=0.3.24,<0.3.25.0a0|>=0.3.25,<0.3.26.0a0|>=0.3.26,<0.3.27.0a0|>=0.3.27,<0.3.28.0a0|>=0.3.27,<1.0a0|>=0.3.26,<1.0a0|>=0.3.25,<1.0a0|>=0.3.24,<1.0a0|>=0.3.23,<1.0a0|>=0.3.21,<1.0a0|>=0.3.20,<1.0a0|>=0.3.18,<1.0a0|>=0.3.17,<1.0a0|>=0.3.15,<1.0a0|>=0.3.12,<1.0a0']
librosa -> numpy[version='>=1.20.3,!=1.22.0,!=1.22.1,!=1.22.2'] -> libopenblas[version='>=0.3.17,<1.0a0|>=0.3.20,<1.0a0|>=0.3.21,<1.0a0']
numba -> numpy[version='>=1.23.5,<2.0a0'] -> libopenblas[version='>=0.3.17,<1.0a0|>=0.3.20,<1.0a0|>=0.3.21,<1.0a0']
h5py -> numpy[version='>=1.22.4,<2.0a0'] -> libopenblas[version='>=0.3.17,<1.0a0|>=0.3.20,<1.0a0|>=0.3.21,<1.0a0']
pysoundfile -> numpy -> libopenblas[version='>=0.3.17,<1.0a0|>=0.3.20,<1.0a0|>=0.3.21,<1.0a0']
libcblas -> libblas==3.9.0=22_osxarm64_openblas -> libopenblas[version='>=0.3.12,<0.3.13.0a0|>=0.3.15,<0.3.16.0a0|>=0.3.17,<0.3.18.0a0|>=0.3.18,<0.3.19.0a0|>=0.3.20,<0.3.21.0a0|>=0.3.21,<0.3.22.0a0|>=0.3.23,<0.3.24.0a0|>=0.3.24,<0.3.25.0a0|>=0.3.25,<0.3.26.0a0|>=0.3.26,<0.3.27.0a0|>=0.3.27,<0.3.28.0a0|>=0.3.27,<1.0a0|>=0.3.26,<1.0a0|>=0.3.25,<1.0a0|>=0.3.24,<1.0a0|>=0.3.23,<1.0a0|>=0.3.21,<1.0a0|>=0.3.20,<1.0a0|>=0.3.18,<1.0a0|>=0.3.17,<1.0a0|>=0.3.15,<1.0a0|>=0.3.12,<1.0a0']
openblas -> libopenblas[version='0.3.11|0.3.12|0.3.12|0.3.13|0.3.15|0.3.15|0.3.16|0.3.17|0.3.17|0.3.18|0.3.20|0.3.20|0.3.21|0.3.21|0.3.21|0.3.21|0.3.23|0.3.24|0.3.25|0.3.26|0.3.27|0.3.21|0.3.20|0.3.18|0.3.17|0.3.17|0.3.13',build='hcab743c_2|h269037a_0|openmp_h74fab25_0|openmp_h2ecc587_1|openmp_he0e8823_0|openmp_hf330de4_1|openmp_h5dd58f0_1|openmp_h130de29_1|openmp_hd76b1f2_0|openmp_h6c19121_0|openmp_hc731615_0|openmp_hc731615_3|openmp_hc731615_2|openmp_hcb59c3b_1|openmp_hcb59c3b_0|openmp_h2209c59_0|openmp_h5dd58f0_0|openmp_hf330de4_0|openmp_hf330de4_0|openmp_hf330de4_0|openmp_h74fab25_1|hea475bc_0|hea475bc_0|h9886b1c_1|h9886b1c_0']
blas -> libblas==3.9.0=22_osxarm64_openblas -> libopenblas[version='>=0.3.12,<0.3.13.0a0|>=0.3.15,<0.3.16.0a0|>=0.3.17,<0.3.18.0a0|>=0.3.18,<0.3.19.0a0|>=0.3.20,<0.3.21.0a0|>=0.3.21,<0.3.22.0a0|>=0.3.23,<0.3.24.0a0|>=0.3.24,<0.3.25.0a0|>=0.3.25,<0.3.26.0a0|>=0.3.26,<0.3.27.0a0|>=0.3.27,<0.3.28.0a0|>=0.3.27,<1.0a0|>=0.3.26,<1.0a0|>=0.3.25,<1.0a0|>=0.3.24,<1.0a0|>=0.3.23,<1.0a0|>=0.3.21,<1.0a0|>=0.3.20,<1.0a0|>=0.3.18,<1.0a0|>=0.3.17,<1.0a0|>=0.3.15,<1.0a0|>=0.3.12,<1.0a0']

Package pcre conflicts for:
harfbuzz -> libglib[version='>=2.72.1,<3.0a0'] -> pcre[version='>=8.44,<9.0a0|>=8.45,<9.0a0']
cairo -> libglib[version='>=2.72.1,<3.0a0'] -> pcre[version='>=8.44,<9.0a0|>=8.45,<9.0a0']
libglib -> pcre[version='>=8.44,<9.0a0|>=8.45,<9.0a0']

Package libblas conflicts for:
scipy -> libcblas[version='>=3.9.0,<4.0a0'] -> libblas[version='3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0.*',build='1_openblas|2_openblas|3_openblas|5_openblas|6_openblas|8_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_accelerate|17_osxarm64_openblas|18_osxarm64_openblas|21_osxarm64_openblas|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|20_osxarm64_accelerate|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|18_osxarm64_accelerate|17_osxarm64_accelerate|16_osxarm64_accelerate|16_osxarm64_openblas|15_osxarm64_openblas|15_osxarm64_accelerate|14_osxarm64_openblas|14_osxarm64_accelerate|13_osxarm64_openblas|12_osxarm64_accelerate|12_osxarm64_openblas|7_openblas|4_openblas']
h5py -> numpy[version='>=1.22.4,<2.0a0'] -> libblas[version='>=3.9.0,<4.0a0']
blas -> libcblas==3.9.0=5_h880f123_netlib -> libblas=3.9.0
soxr-python -> numpy[version='>=1.23.5,<2.0a0'] -> libblas[version='>=3.9.0,<4.0a0']
librosa -> numpy[version='>=1.20.3,!=1.22.0,!=1.22.1,!=1.22.2'] -> libblas[version='>=3.9.0,<4.0a0']
libcblas -> libblas[version='3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0.*',build='1_openblas|2_openblas|3_openblas|5_openblas|6_openblas|8_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_accelerate|17_osxarm64_openblas|18_osxarm64_openblas|21_osxarm64_openblas|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|20_osxarm64_accelerate|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|18_osxarm64_accelerate|17_osxarm64_accelerate|16_osxarm64_accelerate|16_osxarm64_openblas|15_osxarm64_openblas|15_osxarm64_accelerate|14_osxarm64_openblas|14_osxarm64_accelerate|13_osxarm64_openblas|12_osxarm64_accelerate|12_osxarm64_openblas|7_openblas|4_openblas']
blas-devel -> blas==2.106=openblas -> libblas[version='3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0.*',build='1_openblas|6_openblas|5_openblas|4_openblas|3_openblas|2_openblas']
numpy -> libcblas[version='>=3.9.0,<4.0a0'] -> libblas[version='3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0.*',build='1_openblas|2_openblas|3_openblas|5_openblas|6_openblas|8_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_accelerate|17_osxarm64_openblas|18_osxarm64_openblas|21_osxarm64_openblas|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|20_osxarm64_accelerate|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|18_osxarm64_accelerate|17_osxarm64_accelerate|16_osxarm64_accelerate|16_osxarm64_openblas|15_osxarm64_openblas|15_osxarm64_accelerate|14_osxarm64_openblas|14_osxarm64_accelerate|13_osxarm64_openblas|12_osxarm64_accelerate|12_osxarm64_openblas|7_openblas|4_openblas']
scikit-learn -> numpy[version='>=1.23.5,<2.0a0'] -> libblas[version='3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0.*|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|>=3.9.0,<4.0a0',build='1_openblas|2_openblas|3_openblas|5_openblas|6_openblas|8_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_accelerate|17_osxarm64_openblas|18_osxarm64_openblas|21_osxarm64_openblas|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|20_osxarm64_accelerate|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|18_osxarm64_accelerate|17_osxarm64_accelerate|16_osxarm64_accelerate|16_osxarm64_openblas|15_osxarm64_openblas|15_osxarm64_accelerate|14_osxarm64_openblas|14_osxarm64_accelerate|13_osxarm64_openblas|12_osxarm64_accelerate|12_osxarm64_openblas|7_openblas|4_openblas']
liblapack -> libblas[version='3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0.*',build='1_openblas|2_openblas|3_openblas|5_openblas|6_openblas|8_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_accelerate|17_osxarm64_openblas|18_osxarm64_openblas|21_osxarm64_openblas|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|20_osxarm64_accelerate|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|18_osxarm64_accelerate|17_osxarm64_accelerate|16_osxarm64_accelerate|16_osxarm64_openblas|15_osxarm64_openblas|15_osxarm64_accelerate|14_osxarm64_openblas|14_osxarm64_accelerate|13_osxarm64_openblas|12_osxarm64_accelerate|12_osxarm64_openblas|7_openblas|4_openblas']
numpy -> libblas[version='>=3.9.0,<4.0a0']
liblapacke -> libblas[version='3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0|3.9.0.*',build='1_openblas|2_openblas|3_openblas|5_openblas|6_openblas|8_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_accelerate|17_osxarm64_openblas|18_osxarm64_openblas|21_osxarm64_openblas|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|20_osxarm64_accelerate|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|18_osxarm64_accelerate|17_osxarm64_accelerate|16_osxarm64_accelerate|16_osxarm64_openblas|15_osxarm64_openblas|15_osxarm64_accelerate|14_osxarm64_openblas|14_osxarm64_accelerate|13_osxarm64_openblas|12_osxarm64_accelerate|12_osxarm64_openblas|7_openblas|4_openblas']
scipy -> libblas[version='>=3.9.0,<4.0a0']
blas-devel -> libblas==3.9.0[build='1_h9886b1c_netlib|0_h2ec9a88_netlib|5_h880f123_netlib|8_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_accelerate|17_osxarm64_openblas|18_osxarm64_openblas|21_osxarm64_openblas|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|20_osxarm64_accelerate|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|18_osxarm64_accelerate|17_osxarm64_accelerate|16_osxarm64_accelerate|16_osxarm64_openblas|15_osxarm64_openblas|15_osxarm64_accelerate|14_osxarm64_openblas|14_osxarm64_accelerate|13_osxarm64_openblas|12_osxarm64_accelerate|12_osxarm64_openblas|7_openblas']
numba -> numpy[version='>=1.23.5,<2.0a0'] -> libblas[version='>=3.9.0,<4.0a0']
blas -> libblas==3.9.0[build='0_h9886b1c_netlib|1_h9886b1c_netlib|0_h2ec9a88_netlib|1_openblas|2_h2ec9a88_netlib|3_openblas|4_openblas|4_h880f123_netlib|5_openblas|5_h880f123_netlib|6_openblas|8_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_accelerate|17_osxarm64_openblas|18_osxarm64_openblas|21_osxarm64_openblas|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|20_osxarm64_accelerate|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|18_osxarm64_accelerate|17_osxarm64_accelerate|16_osxarm64_accelerate|16_osxarm64_openblas|15_osxarm64_openblas|15_osxarm64_accelerate|14_osxarm64_openblas|14_osxarm64_accelerate|13_osxarm64_openblas|12_osxarm64_accelerate|12_osxarm64_openblas|7_openblas|3_he9612bc_netlib|2_openblas|1_h2ec9a88_netlib']
matplotlib-base -> numpy[version='>=1.21,<2'] -> libblas[version='>=3.9.0,<4.0a0']
pysoundfile -> numpy -> libblas[version='>=3.9.0,<4.0a0']
tensorflow-deps -> numpy[version='>=1.23.2,<1.23.3'] -> libblas[version='>=3.9.0,<4.0a0']

Package pypy3.7 conflicts for:
python_abi -> pypy3.7=7.3
cffi -> pypy3.7[version='7.3.3.*|7.3.4.*|7.3.5.*|7.3.7.*']
pysoundfile -> cffi -> pypy3.7[version='7.3.3.*|7.3.4.*|7.3.5.*|7.3.7.*']
cffi -> python_abi==3.7[build=*_pypy37_pp73] -> pypy3.7=7.3

Package brotli conflicts for:
urllib3 -> brotli[version='>=1.0.9']
matplotlib-base -> fonttools[version='>=4.22.0'] -> brotli[version='>=1.0.1']
requests -> urllib3[version='>=1.21.1,<3'] -> brotli[version='>=1.0.9']

Package libintl-devel conflicts for:
gettext -> libintl-devel==0.22.5=h8fbad5d_2
cairo -> glib[version='>=2.69.1,<3.0a0'] -> libintl-devel
harfbuzz -> glib[version='>=2.69.1,<3.0a0'] -> libintl-devel
libidn2 -> gettext[version='>=0.21.1,<1.0a0'] -> libintl-devel==0.22.5=h8fbad5d_2
libflac -> gettext[version='>=0.21.1,<1.0a0'] -> libintl-devel==0.22.5=h8fbad5d_2
libglib -> gettext[version='>=0.21.1,<1.0a0'] -> libintl-devel==0.22.5=h8fbad5d_2
gnutls -> gettext[version='>=0.21.1,<1.0a0'] -> libintl-devel==0.22.5=h8fbad5d_2

Package liblapacke conflicts for:
scipy -> blas=[build=openblas] -> liblapacke==3.9.0[build='1_openblas|2_openblas|3_openblas|5_openblas|6_openblas|8_openblas|9_openblas|11_osxarm64_openblas|15_osxarm64_openblas|17_osxarm64_openblas|18_osxarm64_openblas|19_osxarm64_openblas|21_osxarm64_openblas|22_osxarm64_openblas|20_osxarm64_openblas|16_osxarm64_openblas|14_osxarm64_openblas|13_osxarm64_openblas|12_osxarm64_openblas|10_openblas|7_openblas|4_openblas']
blas-devel -> liblapacke==3.9.0[build='1_h9886b1c_netlib|0_h2ec9a88_netlib|5_h880f123_netlib|8_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_accelerate|17_osxarm64_openblas|18_osxarm64_openblas|21_osxarm64_openblas|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|20_osxarm64_accelerate|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|18_osxarm64_accelerate|17_osxarm64_accelerate|16_osxarm64_accelerate|16_osxarm64_openblas|15_osxarm64_openblas|15_osxarm64_accelerate|14_osxarm64_openblas|14_osxarm64_accelerate|13_osxarm64_openblas|12_osxarm64_accelerate|12_osxarm64_openblas|7_openblas']
blas-devel -> blas==2.106=openblas -> liblapacke==3.9.0[build='1_openblas|6_openblas|5_openblas|4_openblas|3_openblas|2_openblas']
blas -> liblapacke==3.9.0[build='0_h9886b1c_netlib|1_h9886b1c_netlib|0_h2ec9a88_netlib|1_openblas|2_h2ec9a88_netlib|3_openblas|4_openblas|4_h880f123_netlib|5_openblas|5_h880f123_netlib|6_openblas|8_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_accelerate|17_osxarm64_openblas|18_osxarm64_openblas|21_osxarm64_openblas|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|20_osxarm64_accelerate|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|18_osxarm64_accelerate|17_osxarm64_accelerate|16_osxarm64_accelerate|16_osxarm64_openblas|15_osxarm64_openblas|15_osxarm64_accelerate|14_osxarm64_openblas|14_osxarm64_accelerate|13_osxarm64_openblas|12_osxarm64_accelerate|12_osxarm64_openblas|7_openblas|3_he9612bc_netlib|2_openblas|1_h2ec9a88_netlib']
numpy -> blas=[build=openblas] -> liblapacke==3.9.0[build='1_openblas|2_openblas|3_openblas|5_openblas|6_openblas|8_openblas|9_openblas|11_osxarm64_openblas|15_osxarm64_openblas|17_osxarm64_openblas|18_osxarm64_openblas|19_osxarm64_openblas|21_osxarm64_openblas|22_osxarm64_openblas|20_osxarm64_openblas|16_osxarm64_openblas|14_osxarm64_openblas|13_osxarm64_openblas|12_osxarm64_openblas|10_openblas|7_openblas|4_openblas']

Package flit-core conflicts for:
importlib-metadata -> typing_extensions[version='>=3.6.4'] -> flit-core[version='>=3.6,<4']
typing_extensions -> flit-core[version='>=3.6,<4']
librosa -> typing_extensions[version='>=4.1.1'] -> flit-core[version='>=3.6,<4']

Package c-ares conflicts for:
grpcio -> c-ares[version='>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.19.1,<2.0a0']
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> c-ares[version='>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.19.1,<2.0a0']
libcurl -> libnghttp2[version='>=1.58.0,<2.0a0'] -> c-ares[version='>=1.16.1,<2.0a0|>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.20.1,<2.0a0|>=1.21.0,<2.0a0|>=1.23.0,<2.0a0|>=1.19.1,<2.0a0|>=1.7.5|>=1.19.0,<2.0a0']
libnghttp2 -> c-ares[version='>=1.16.1,<2.0a0|>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.20.1,<2.0a0|>=1.21.0,<2.0a0|>=1.23.0,<2.0a0|>=1.7.5|>=1.19.1,<2.0a0|>=1.19.0,<2.0a0']
grpcio -> libgrpc==1.62.1=h9c18a4f_0 -> c-ares[version='>=1.19.0,<2.0a0|>=1.20.1,<2.0a0|>=1.21.0,<2.0a0|>=1.22.1,<2.0a0|>=1.25.0,<2.0a0|>=1.26.0,<2.0a0|>=1.27.0,<2.0a0']

Package libllvm14 conflicts for:
librosa -> numba[version='>=0.51.0'] -> libllvm14[version='>=14.0.6,<14.1.0a0']
llvmlite -> libllvm14[version='>=14.0.6,<14.1.0a0']
numba -> libllvm14[version='>=14.0.6,<14.1.0a0']

Package olefile conflicts for:
matplotlib-base -> pillow[version='>=8'] -> olefile
pillow -> olefile

Package six conflicts for:
packaging -> six
python-dateutil -> six[version='>=1.5']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> six[version='1.15.0.*|>=1.10.0|>=1.12|>=1.15,<1.16|>=1.15.0']
cycler -> six
wheel -> packaging[version='>=20.2'] -> six
pooch -> packaging[version='>=20.0'] -> six
librosa -> six[version='>=1.3']
protobuf -> six
matplotlib-base -> cycler[version='>=0.10'] -> six[version='>=1.5']
urllib3 -> cryptography[version='>=1.3.4'] -> six[version='>=1.4.1|>=1.5.2']
librosa -> packaging[version='>=20.0'] -> six
grpcio -> six[version='>=1.5.2|>=1.6.0']
h5py -> six
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> six[version='>=1.5.2|>=1.6.0']
lazy_loader -> packaging -> six
zipp -> more-itertools -> six[version='>=1.0.0,<2.0.0']

Package libllvm11 conflicts for:
numba -> llvmlite[version='>=0.39.1,<0.40.0a0'] -> libllvm11[version='>=11.1.0,<11.2.0a0']
llvmlite -> libllvm11[version='>=11.1.0,<11.2.0a0']

Package certifi conflicts for:
numba -> setuptools -> certifi[version='>=2016.9.26']
grpcio -> setuptools -> certifi[version='>=2016.9.26']
urllib3 -> certifi
protobuf -> setuptools -> certifi[version='>=2016.9.26']
pooch -> requests[version='>=2.19.0'] -> certifi[version='>=2017.4.17']
librosa -> matplotlib-base[version='>=3.3.0'] -> certifi[version='>=2016.9.26|>=2020.06.20']
matplotlib-base -> certifi[version='>=2020.06.20']
setuptools -> certifi[version='>=2016.9.26']
requests -> certifi[version='>=2017.4.17']
pip -> setuptools -> certifi[version='>=2016.9.26']
requests -> urllib3[version='>=1.21.1,<3'] -> certifi
joblib -> setuptools -> certifi[version='>=2016.9.26']
matplotlib-base -> setuptools -> certifi[version='>=2016.9.26']
wheel -> setuptools -> certifi[version='>=2016.9.26']

Package more-itertools conflicts for:
zipp -> jaraco.itertools -> more-itertools[version='>=4.0.0']
zipp -> more-itertools

Package typing-extensions conflicts for:
platformdirs -> typing-extensions[version='>=4.4|>=4.5|>=4.6.3']
pooch -> platformdirs[version='>=2.5.0'] -> typing-extensions[version='>=4.4|>=4.5|>=4.6.3']

Package gmp conflicts for:
gnutls -> gmp[version='>=6.2.1,<7.0a0']
ffmpeg -> gmp[version='>=6.2.1,<7.0a0|>=6.3.0,<7.0a0']
audioread -> ffmpeg -> gmp[version='>=6.2.1,<7.0a0|>=6.3.0,<7.0a0']
nettle -> gmp[version='>=6.2.1,<7.0a0']

Package protobuf conflicts for:
tensorflow-deps -> protobuf[version='>=3.19.1,<3.20']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> protobuf[version='>=3.19.6|>=3.20.3,<5,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5|>=3.9.2|>=3.6.1|>=3.6.0|>=3.9.2,<3.20']

Package libvorbis conflicts for:
libsndfile -> libvorbis[version='>=1.3.7,<1.4.0a0']
pysoundfile -> libsndfile[version='>=1.2'] -> libvorbis[version='>=1.3.7,<1.4.0a0']

Package icu conflicts for:
libxml2 -> icu[version='69.*|>=70.1,<71.0a0|>=72.1,<73.0a0|>=73.2,<74.0a0|>=69.1,<70.0a0|>=68.1,<69.0a0|>=67.1,<68.0a0|>=73.1,<74.0a0']
cairo -> icu[version='>=67.1,<68.0a0|>=68.1,<69.0a0|>=69.1,<70.0a0|>=70.1,<71.0a0|>=72.1,<73.0a0|>=73.2,<74.0a0']
fontconfig -> icu[version='>=67.1,<68.0a0|>=68.1,<69.0a0']
ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0'] -> icu[version='69.*|>=68.1,<69.0a0|>=72.1,<73.0a0|>=73.2,<74.0a0|>=73.1,<74.0a0|>=70.1,<71.0a0|>=69.1,<70.0a0']
gettext -> libxml2[version='>=2.10.3,<2.11.0a0'] -> icu[version='69.*|>=68.1,<69.0a0|>=70.1,<71.0a0|>=72.1,<73.0a0|>=73.1,<74.0a0|>=69.1,<70.0a0|>=67.1,<68.0a0']
harfbuzz -> icu[version='>=67.1,<68.0a0|>=68.1,<69.0a0|>=68.2,<69.0a0|>=69.1,<70.0a0|>=70.1,<71.0a0|>=72.1,<73.0a0|>=73.2,<74.0a0|>=73.1,<74.0a0']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> icu[version='>=68.1,<69.0a0|>=69.1,<70.0a0|>=70.1,<71.0a0|>=72.1,<73.0a0|>=73.2,<74.0a0']
fontconfig -> libxml2[version='>=2.9.12,<3.0.0a0'] -> icu[version='69.*|>=70.1,<71.0a0|>=72.1,<73.0a0|>=73.2,<74.0a0|>=69.1,<70.0a0|>=73.1,<74.0a0']
libass -> harfbuzz[version='>=8.1.1,<9.0a0'] -> icu[version='>=72.1,<73.0a0|>=73.2,<74.0a0']

Package jpeg conflicts for:
pillow -> lcms2[version='>=2.12,<3.0a0'] -> jpeg[version='>=9b,<10a']
openjpeg -> libtiff[version='>=4.5.0,<4.6.0a0'] -> jpeg[version='>=9b,<10a|>=9d,<10a|>=9e,<10a']
libtiff -> jpeg[version='>=9b,<10a|>=9d,<10a|>=9e,<10a']
matplotlib-base -> pillow[version='>=8'] -> jpeg[version='>=9d,<10a|>=9e,<10a']
lcms2 -> jpeg[version='>=9b,<10a|>=9d,<10a|>=9e,<10a']
pillow -> jpeg[version='>=9d,<10a|>=9e,<10a']
tensorflow -> tensorflow-base==2.11.0=cpu_py39h072764a_0 -> jpeg[version='>=9d,<10a|>=9e,<10a']

Package cryptography conflicts for:
requests -> urllib3[version='>=1.21.1,<3'] -> cryptography[version='>=1.3.4']
urllib3 -> pyopenssl[version='>=0.14'] -> cryptography[version='>=2.8|>=3.2|>=3.3|>=35.0,<39|>=38.0.0,<39|>=38.0.0,<40|>=38.0.0,<41|>=38.0.0,<42,!=40.0.0,!=40.0.1|>=41.0.5,<42|>=41.0.5,<43|>=35.0']
urllib3 -> cryptography[version='>=1.3.4']

Package libasprintf conflicts for:
libidn2 -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
libflac -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
libasprintf-devel -> libasprintf==0.22.5=h8fbad5d_2
libglib -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
gnutls -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
gettext -> libasprintf==0.22.5=h8fbad5d_2

Package zipp conflicts for:
importlib-metadata -> zipp[version='>=0.5']
matplotlib-base -> importlib-resources[version='>=3.2.0'] -> zipp[version='>=0.4|>=3.1.0']
numba -> importlib-metadata -> zipp[version='>=0.5']
lazy_loader -> importlib-metadata -> zipp[version='>=0.5']

Package libintl conflicts for:
cairo -> libglib[version='>=2.78.0,<3.0a0'] -> libintl[version='>=0.22.5,<1.0a0']
libgettextpo-devel -> libintl==0.22.5=h8fbad5d_2
libidn2 -> gettext[version='>=0.21.1,<1.0a0'] -> libintl==0.22.5=h8fbad5d_2
harfbuzz -> libglib[version='>=2.78.1,<3.0a0'] -> libintl[version='>=0.22.5,<1.0a0']
gettext-tools -> libintl==0.22.5=h8fbad5d_2
libflac -> gettext[version='>=0.21.1,<1.0a0'] -> libintl==0.22.5=h8fbad5d_2
gettext -> libintl==0.22.5=h8fbad5d_2
libglib -> gettext[version='>=0.21.1,<1.0a0'] -> libintl==0.22.5=h8fbad5d_2
libglib -> libintl[version='>=0.22.5,<1.0a0']
libgettextpo -> libintl==0.22.5=h8fbad5d_2
libintl-devel -> libintl==0.22.5=h8fbad5d_2
gnutls -> gettext[version='>=0.21.1,<1.0a0'] -> libintl==0.22.5=h8fbad5d_2

Package mpich conflicts for:
hdf5 -> mpich[version='>=3.3,<5.0.0a0|>=3.4,<5.0.0a0|>=4.0,<4.1.0a0|>=4.0.2,<4.1.0a0|>=4.0.3,<4.1.0a0|>=4.0.3,<5.0a0|>=4.1.1,<5.0a0|>=4.1.2,<5.0a0']
h5py -> mpi4py[version='>=3.0'] -> mpich[version='>=3.3,<3.4.0a0|>=3.3,<5.0.0a0|>=3.4,<5.0.0a0|>=4.0.1,<5.0a0|>=4.2.0,<5.0a0|>=4.0.3,<4.1.0a0|>=4.0.2,<4.1.0a0|>=4.0,<4.1.0a0']
scipy -> fftw[version='>=3.3.9,<4.0a0'] -> mpich[version='>=3.4,<5.0.0a0|>=3.4.1,<5.0a0|>=3.4.2,<5.0a0|>=4.0.2,<5.0a0|>=4.0.3,<5.0a0|>=4.1.1,<5.0a0']
h5py -> mpich[version='>=3.3.2,<5.0.0a0|>=3.4.1,<5.0a0|>=3.4.2,<5.0a0|>=4.0.2,<5.0a0|>=4.0.3,<5.0a0|>=4.1.1,<5.0a0|>=4.1.2,<5.0a0']
tensorflow-deps -> h5py[version='>=3.6.0,<3.7'] -> mpich[version='>=3.4.2,<5.0a0']

Package aom conflicts for:
ffmpeg -> aom[version='>=3.2.0,<3.3.0a0|>=3.3.0,<3.4.0a0|>=3.4.0,<3.5.0a0|>=3.5.0,<3.6.0a0|>=3.6.1,<3.7.0a0|>=3.7.0,<3.8.0a0|>=3.7.1,<3.8.0a0|>=3.8.1,<3.9.0a0|>=3.8.2,<3.9.0a0']
audioread -> ffmpeg -> aom[version='>=3.2.0,<3.3.0a0|>=3.3.0,<3.4.0a0|>=3.4.0,<3.5.0a0|>=3.5.0,<3.6.0a0|>=3.6.1,<3.7.0a0|>=3.7.0,<3.8.0a0|>=3.7.1,<3.8.0a0|>=3.8.1,<3.9.0a0|>=3.8.2,<3.9.0a0']

Package x264 conflicts for:
audioread -> ffmpeg -> x264[version='>=1!152.20180806,<1!153|>=1!161.3030,<1!162|>=1!164.3095,<1!165']
ffmpeg -> x264[version='>=1!152.20180806,<1!153|>=1!161.3030,<1!162|>=1!164.3095,<1!165']

Package libopus conflicts for:
libsndfile -> libopus[version='>=1.3.1,<2.0a0']
pysoundfile -> libsndfile[version='>=1.2'] -> libopus[version='>=1.3.1,<2.0a0']
ffmpeg -> libopus[version='>=1.3,<2.0a0|>=1.3.1,<2.0a0']
audioread -> ffmpeg -> libopus[version='>=1.3,<2.0a0|>=1.3.1,<2.0a0']

Package cached-property conflicts for:
h5py -> cached-property
tensorflow-deps -> h5py[version='>=3.6.0,<3.7'] -> cached-property

Package libogg conflicts for:
libvorbis -> libogg[version='>=1.3.4,<1.4.0a0|>=1.3.5,<1.4.0a0|>=1.3.5,<2.0a0']
pysoundfile -> libsndfile[version='>=1.2'] -> libogg[version='>=1.3.4,<1.4.0a0']
libflac -> libogg[version='1.3.*|>=1.3.4,<1.4.0a0']
libsndfile -> libogg[version='>=1.3.4,<1.4.0a0']
libsndfile -> libflac[version='>=1.4.3,<1.5.0a0'] -> libogg[version='1.3.*|>=1.3.5,<1.4.0a0|>=1.3.5,<2.0a0']

Package gettext-tools conflicts for:
libflac -> gettext[version='>=0.21.1,<1.0a0'] -> gettext-tools==0.22.5=h8fbad5d_2
gnutls -> gettext[version='>=0.21.1,<1.0a0'] -> gettext-tools==0.22.5=h8fbad5d_2
libidn2 -> gettext[version='>=0.21.1,<1.0a0'] -> gettext-tools==0.22.5=h8fbad5d_2
gettext -> gettext-tools==0.22.5=h8fbad5d_2
libglib -> gettext[version='>=0.21.1,<1.0a0'] -> gettext-tools==0.22.5=h8fbad5d_2

Package libpng conflicts for:
pillow -> freetype[version='>=2.12.1,<3.0a0'] -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.39,<1.7.0a0|>=1.6.43,<1.7.0a0']
cairo -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.38,<1.7.0a0|>=1.6.39,<1.7.0a0']
fontconfig -> freetype[version='>=2.12.1,<3.0a0'] -> libpng[version='>=1.6.39,<1.7.0a0']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.38,<1.7.0a0|>=1.6.39,<1.7.0a0']
harfbuzz -> cairo[version='>=1.18.0,<2.0a0'] -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.39,<1.7.0a0|>=1.6.38,<1.7.0a0']
libass -> freetype[version='>=2.12.1,<3.0a0'] -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.39,<1.7.0a0']
freetype -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.39,<1.7.0a0']
fontconfig -> libpng[version='>=1.6.37,<1.7.0a0']
matplotlib-base -> freetype[version='>=2.12.1,<3.0a0'] -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.39,<1.7.0a0']
openjpeg -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.39,<1.7.0a0|>=1.6.43,<1.7.0a0']
ffmpeg -> freetype[version='>=2.12.1,<3.0a0'] -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.39,<1.7.0a0']

Package jbig conflicts for:
lcms2 -> libtiff[version='>=4.2.0,<4.5.0a0'] -> jbig
pillow -> libtiff[version='>=4.3.0,<4.5.0a0'] -> jbig
libtiff -> jbig
openjpeg -> libtiff[version='>=4.3.0,<4.5.0a0'] -> jbig

Package libprotobuf conflicts for:
grpcio -> libprotobuf[version='>=3.20.1,<3.21.0a0|>=3.21.10,<3.22.0a0|>=3.21.12,<3.22.0a0|>=3.21.9,<3.22.0a0|>=3.21.8,<3.22.0a0|>=3.21.5,<3.22.0a0|>=3.20.3,<3.21.0a0']
grpcio -> libgrpc==1.62.1=h9c18a4f_0 -> libprotobuf[version='>=3.20.3,<3.20.4.0a0|>=4.23.1,<4.24.0a0|>=4.23.2,<4.23.3.0a0|>=4.23.3,<4.23.4.0a0|>=4.23.4,<4.23.5.0a0|>=4.24.3,<4.24.4.0a0|>=4.24.4,<4.24.5.0a0|>=4.25.1,<4.25.2.0a0|>=4.25.2,<4.25.3.0a0|>=4.25.3,<4.25.4.0a0']
protobuf -> libprotobuf[version='3.13.0.1.*|3.14.0.*|3.15.0.*|3.15.1.*|3.15.2.*|3.15.3.*|3.15.4.*|3.15.5.*|3.15.6.*|3.15.7.*|3.15.8.*|3.16.0.*|3.17.0.*|3.17.1.*|3.17.2.*|3.18.0.*|3.18.1.*|3.18.3.*|3.19.1.*|3.19.2.*|3.19.3.*|3.19.4.*|3.19.6.*|3.20.0.*|3.20.1.*|3.20.2.*|3.20.3.*|3.21.1.*|3.21.10.*|3.21.11.*|3.21.12.*|>=4.22.5,<4.23.0a0|>=4.23.1,<4.24.0a0|>=4.23.2,<4.23.3.0a0|>=4.23.3,<4.23.4.0a0|>=4.23.4,<4.23.5.0a0|>=4.24.3,<4.24.4.0a0|>=4.24.4,<4.24.5.0a0|>=4.25.1,<4.25.2.0a0|>=4.25.2,<4.25.3.0a0|>=4.25.3,<4.25.4.0a0|>=3.21.12,<3.22.0a0|>=3.21.11,<3.22.0a0|>=3.21.10,<3.22.0a0|3.21.9.*|>=3.21.9,<3.22.0a0|3.21.8.*|>=3.21.8,<3.22.0a0|3.21.7.*|>=3.21.7,<3.22.0a0|3.21.6.*|>=3.21.6,<3.22.0a0|3.21.5.*|>=3.21.5,<3.22.0a0|3.21.4.*|>=3.21.4,<3.22.0a0|3.21.3.*|>=3.21.3,<3.22.0a0|3.21.2.*|>=3.21.2,<3.22.0a0|>=3.21.1,<3.22.0a0|>=3.20.3,<3.21.0a0|>=3.20.2,<3.21.0a0|>=3.20.1,<3.21.0a0|>=3.20.0,<3.21.0a0|>=3.19.6,<3.20.0a0|>=3.19.4,<3.20.0a0|>=3.19.3,<3.20.0a0|>=3.19.2,<3.20.0a0|>=3.19.1,<3.20.0a0|>=3.18.3,<3.19.0a0|>=3.18.1,<3.19.0a0|>=3.18.0,<3.19.0a0|>=3.17.2,<3.18.0a0|>=3.17.1,<3.18.0a0|>=3.17.0,<3.18.0a0|>=3.16.0,<3.17.0a0|>=3.15.8,<3.16.0a0|>=3.15.7,<3.16.0a0|>=3.15.6,<3.16.0a0|>=3.15.5,<3.16.0a0|>=3.15.4,<3.16.0a0|>=3.15.3,<3.16.0a0|>=3.15.2,<3.16.0a0|>=3.15.1,<3.16.0a0|>=3.15.0,<3.16.0a0|>=3.14.0,<3.15.0a0|>=3.13.0.1,<3.14.0a0']
ffmpeg -> libopenvino-onnx-frontend[version='>=2024.0.0,<2024.0.1.0a0'] -> libprotobuf[version='>=4.24.3,<4.24.4.0a0|>=4.24.4,<4.24.5.0a0|>=4.25.1,<4.25.2.0a0|>=4.25.2,<4.25.3.0a0|>=4.25.3,<4.25.4.0a0']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> libprotobuf[version='>=3.15.7,<3.16.0a0|>=3.15.8,<3.16.0a0|>=3.16.0,<3.17.0a0|>=3.18.1,<3.19.0a0|>=3.19.4,<3.20.0a0|>=3.20.1,<3.21.0a0|>=3.21.12,<3.22.0a0|>=4.24.4,<4.24.5.0a0|>=3.21.6,<3.22.0a0|>=3.20.3,<3.21.0a0']
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> libprotobuf[version='3.19.1.*|3.19.2.*|3.19.3.*|3.19.4.*|3.19.6.*|>=3.20.1,<3.21.0a0|>=3.21.10,<3.22.0a0|>=3.21.12,<3.22.0a0|>=3.21.9,<3.22.0a0|>=3.21.8,<3.22.0a0|>=3.21.5,<3.22.0a0|>=3.20.3,<3.21.0a0|>=3.19.6,<3.20.0a0|>=3.19.4,<3.20.0a0|>=3.19.3,<3.20.0a0|>=3.19.2,<3.20.0a0|>=3.19.1,<3.20.0a0']

Package openblas conflicts for:
blas -> blas-devel==3.9.0=22_osxarm64_openblas -> openblas[version='0.3.18.*|0.3.20.*|0.3.21.*|0.3.23.*|0.3.24.*|0.3.25.*|0.3.26.*|0.3.27.*']
blas-devel -> openblas[version='0.3.18.*|0.3.20.*|0.3.21.*|0.3.23.*|0.3.24.*|0.3.25.*|0.3.26.*|0.3.27.*']

Package openmpi conflicts for:
h5py -> mpi4py[version='>=3.0'] -> openmpi[version='>=4.0,<5.0.0a0|>=4.1,<4.2.0a0|>=4.1.3,<5.0a0|>=4.1.4,<4.2.0a0']
h5py -> openmpi[version='>=4.0.5,<5.0.0a0|>=4.1.0,<5.0a0|>=4.1.1,<5.0a0|>=4.1.2,<5.0a0|>=4.1.4,<5.0a0|>=4.1.5,<5.0a0|>=4.1.6,<5.0a0']

Package libev conflicts for:
libnghttp2 -> libev[version='>=4.11|>=4.33,<4.34.0a0|>=4.33,<5.0a0']
libcurl -> libnghttp2[version='>=1.58.0,<2.0a0'] -> libev[version='>=4.11|>=4.33,<4.34.0a0|>=4.33,<5.0a0']

Package idna conflicts for:
urllib3 -> idna[version='>=2.0.0']
urllib3 -> cryptography[version='>=1.3.4'] -> idna
requests -> urllib3[version='>=1.21.1,<3'] -> idna[version='>=2.0.0']
pooch -> requests[version='>=2.19.0'] -> idna[version='>=2.5,<3|>=2.5,<4']
requests -> idna[version='>=2.5,<3|>=2.5,<4']

Package ffmpeg conflicts for:
librosa -> audioread[version='>=2.1.9'] -> ffmpeg
audioread -> ffmpeg

Package openh264 conflicts for:
ffmpeg -> openh264[version='>=1.8.0,<1.9.0a0|>=2.1.1,<2.2.0a0|>=2.2.0,<2.3.0a0|>=2.3.0,<2.3.1.0a0|>=2.3.1,<2.3.2.0a0|>=2.4.0,<2.4.1.0a0|>=2.4.1,<2.4.2.0a0']
audioread -> ffmpeg -> openh264[version='>=1.8.0,<1.9.0a0|>=2.1.1,<2.2.0a0|>=2.2.0,<2.3.0a0|>=2.3.0,<2.3.1.0a0|>=2.3.1,<2.3.2.0a0|>=2.4.0,<2.4.1.0a0|>=2.4.1,<2.4.2.0a0']

Package libtasn1 conflicts for:
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> libtasn1[version='>=4.16.0,<5.0a0|>=4.18.0,<5.0a0|>=4.19.0,<5.0a0']
gnutls -> libtasn1[version='>=4.16.0,<5.0a0|>=4.18.0,<5.0a0|>=4.19.0,<5.0a0']
p11-kit -> libtasn1[version='>=4.18.0,<5.0a0|>=4.19.0,<5.0a0']

Package jaraco.itertools conflicts for:
importlib-metadata -> zipp[version='>=0.5'] -> jaraco.itertools
zipp -> jaraco.itertools

Package libtiff conflicts for:
pillow -> libtiff[version='>=4.1.0,<4.4.0a0|>=4.2.0,<4.4.0a0|>=4.3.0,<4.4.0a0|>=4.3.0,<4.5.0a0|>=4.4.0,<4.5.0a0|>=4.5.0,<4.6.0a0|>=4.5.1,<4.6.0a0|>=4.6.0,<4.7.0a0|>=4.2.0,<5.0a0']
lcms2 -> libtiff[version='>=4.1.0,<4.5.0a0|>=4.2.0,<4.5.0a0|>=4.4.0,<4.5.0a0|>=4.5.0,<4.6.0a0|>=4.6.0,<4.7.0a0|>=4.2.0,<5.0a0']
openjpeg -> libtiff[version='>=4.1.0,<4.5.0a0|>=4.2.0,<4.5.0a0|>=4.3.0,<4.5.0a0|>=4.4.0,<4.5.0a0|>=4.5.0,<4.6.0a0|>=4.6.0,<4.7.0a0|>=4.2.0,<5.0a0']
matplotlib-base -> pillow[version='>=8'] -> libtiff[version='>=4.1.0,<4.4.0a0|>=4.2.0,<4.4.0a0|>=4.3.0,<4.4.0a0|>=4.3.0,<4.5.0a0|>=4.4.0,<4.5.0a0|>=4.5.0,<4.6.0a0|>=4.5.1,<4.6.0a0|>=4.6.0,<4.7.0a0|>=4.2.0,<5.0a0']
pillow -> lcms2[version='>=2.12,<3.0a0'] -> libtiff[version='>=4.1.0,<4.5.0a0|>=4.2.0,<4.5.0a0']

Package fontconfig conflicts for:
ffmpeg -> fontconfig[version='>=2.13.96,<3.0a0|>=2.14.0,<3.0a0|>=2.14.1,<3.0a0|>=2.14.2,<3.0a0']
harfbuzz -> cairo[version='>=1.18.0,<2.0a0'] -> fontconfig[version='>=2.13.1,<2.13.96.0a0|>=2.13.96,<3.0a0|>=2.14.2,<3.0a0|>=2.14.1,<3.0a0|>=2.13.1,<3.0a0']
cairo -> fontconfig[version='>=2.13.1,<2.13.96.0a0|>=2.13.96,<3.0a0|>=2.14.2,<3.0a0|>=2.14.1,<3.0a0|>=2.13.1,<3.0a0']
audioread -> ffmpeg -> fontconfig[version='>=2.13.96,<3.0a0|>=2.14.0,<3.0a0|>=2.14.1,<3.0a0|>=2.14.2,<3.0a0']
libass -> fontconfig[version='>=2.14.2,<3.0a0']

Package svt-av1 conflicts for:
audioread -> ffmpeg -> svt-av1[version='<1.0.0a0|>=1.1.0,<1.1.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0|>=1.3.0,<1.3.1.0a0|>=1.4.0,<1.4.1.0a0|>=1.4.1,<1.4.2.0a0|>=1.5.0,<1.5.1.0a0|>=1.6.0,<1.6.1.0a0|>=1.7.0,<1.7.1.0a0|>=1.8.0,<1.8.1.0a0|>=2.0.0,<2.0.1.0a0']
ffmpeg -> svt-av1[version='<1.0.0a0|>=1.1.0,<1.1.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0|>=1.3.0,<1.3.1.0a0|>=1.4.0,<1.4.1.0a0|>=1.4.1,<1.4.2.0a0|>=1.5.0,<1.5.1.0a0|>=1.6.0,<1.6.1.0a0|>=1.7.0,<1.7.1.0a0|>=1.8.0,<1.8.1.0a0|>=2.0.0,<2.0.1.0a0']

Package libbrotlidec conflicts for:
brotli-bin -> libbrotlidec[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_1|hb547adb_0|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']
brotli -> libbrotlidec[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_1|hb547adb_0|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']
urllib3 -> brotli[version='>=1.0.9'] -> libbrotlidec[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_1|hb547adb_0|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']

Package krb5 conflicts for:
libcurl -> krb5[version='>=1.17.1,<1.18.0a0|>=1.19.1,<1.20.0a0|>=1.19.2,<1.20.0a0|>=1.19.3,<1.20.0a0|>=1.20.1,<1.21.0a0|>=1.21.1,<1.22.0a0|>=1.21.2,<1.22.0a0|>=1.19.4,<1.20.0a0']
hdf5 -> libcurl[version='>=8.4.0,<9.0a0'] -> krb5[version='>=1.17.1,<1.18.0a0|>=1.19.1,<1.20.0a0|>=1.19.2,<1.20.0a0|>=1.19.4,<1.20.0a0|>=1.20.1,<1.21.0a0|>=1.21.2,<1.22.0a0|>=1.21.1,<1.22.0a0|>=1.19.3,<1.20.0a0']

Package pcre2 conflicts for:
libglib -> pcre2[version='>=10.37,<10.38.0a0|>=10.40,<10.41.0a0|>=10.42,<10.43.0a0|>=10.43,<10.44.0a0']
harfbuzz -> libglib[version='>=2.78.1,<3.0a0'] -> pcre2[version='>=10.37,<10.38.0a0|>=10.40,<10.41.0a0|>=10.42,<10.43.0a0|>=10.43,<10.44.0a0']
cairo -> libglib[version='>=2.78.0,<3.0a0'] -> pcre2[version='>=10.37,<10.38.0a0|>=10.40,<10.41.0a0|>=10.42,<10.43.0a0|>=10.43,<10.44.0a0']

Package libdeflate conflicts for:
libtiff -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']
lcms2 -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']
openjpeg -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']
pillow -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']

Package charset-normalizer conflicts for:
requests -> charset-normalizer[version='>=2,<3|>=2,<4|>=2.0.0,<2.1|>=2.0.0,<2.0.1|>=2.0.0,<3|>=2.0.0,<2.1.0']
pooch -> requests[version='>=2.19.0'] -> charset-normalizer[version='>=2,<3|>=2,<4|>=2.0.0,<2.1|>=2.0.0,<2.0.1|>=2.0.0,<3|>=2.0.0,<2.1.0']

Package lame conflicts for:
audioread -> ffmpeg -> lame[version='>=3.100,<3.101.0a0']
ffmpeg -> lame[version='>=3.100,<3.101.0a0']
pysoundfile -> libsndfile[version='>=1.2'] -> lame[version='>=3.100,<3.101.0a0']
libsndfile -> lame[version='>=3.100,<3.101.0a0']

Package giflib conflicts for:
pillow -> libwebp -> giflib[version='>=5.2.1,<5.3.0a0']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> giflib[version='>=5.2.1,<5.3.0a0']

Package kiwisolver conflicts for:
librosa -> matplotlib-base[version='>=3.3.0'] -> kiwisolver[version='>=1.0.1|>=1.3.1']
matplotlib-base -> kiwisolver[version='>=1.0.1|>=1.3.1']

Package pyparsing conflicts for:
librosa -> matplotlib-base[version='>=3.3.0'] -> pyparsing[version='<3,>=2.0.2|>=2.0.2,!=3.0.5|>=2.0.3,!=2.0.4,!=2.1.2,!=2.1.6|>=2.2.1|>=2.3.1|>=2.3.1,<3.1|>=2.0.2,<3|>=2.0.2']
matplotlib-base -> pyparsing[version='>=2.0.3,!=2.0.4,!=2.1.2,!=2.1.6|>=2.2.1|>=2.3.1|>=2.3.1,<3.1']
matplotlib-base -> packaging[version='>=20.0'] -> pyparsing[version='<3,>=2.0.2|>=2.0.2,!=3.0.5|>=2.0.2,<3|>=2.0.2']
packaging -> pyparsing[version='<3,>=2.0.2|>=2.0.2,!=3.0.5|>=2.0.2,<3|>=2.0.2']
pooch -> packaging[version='>=20.0'] -> pyparsing[version='<3,>=2.0.2|>=2.0.2,!=3.0.5|>=2.0.2,<3|>=2.0.2']
wheel -> packaging[version='>=20.2'] -> pyparsing[version='<3,>=2.0.2|>=2.0.2,!=3.0.5|>=2.0.2,<3|>=2.0.2']
lazy_loader -> packaging -> pyparsing[version='<3,>=2.0.2|>=2.0.2,!=3.0.5|>=2.0.2,<3|>=2.0.2']

Package wheel conflicts for:
tensorflow -> tensorflow-base==2.6.2=cpu_py39hd39b1ba_2 -> wheel[version='>=0.26|>=0.35|>=0.35,<1|>=0.35,<0.36']
python=3.8 -> pip -> wheel
pip -> wheel

Package fonts-conda-ecosystem conflicts for:
harfbuzz -> cairo[version='>=1.18.0,<2.0a0'] -> fonts-conda-ecosystem
audioread -> ffmpeg -> fonts-conda-ecosystem
cairo -> fonts-conda-ecosystem
ffmpeg -> fonts-conda-ecosystem
libass -> fonts-conda-ecosystem

Package libxcb conflicts for:
matplotlib-base -> pillow[version='>=8'] -> libxcb[version='>=1.13,<1.14.0a0|>=1.15,<1.16.0a0']
pillow -> libxcb[version='>=1.13,<1.14.0a0|>=1.15,<1.16.0a0']

Package openjpeg conflicts for:
pillow -> openjpeg[version='>=2.3.0,<3.0a0|>=2.4.0,<3.0.0a0|>=2.5.0,<2.6.0a0|>=2.5.0,<3.0a0|>=2.5.2,<3.0a0|>=2.4.0,<3.0a0']
matplotlib-base -> pillow[version='>=8'] -> openjpeg[version='>=2.3.0,<3.0a0|>=2.4.0,<3.0.0a0|>=2.5.0,<2.6.0a0|>=2.5.0,<3.0a0|>=2.5.2,<3.0a0|>=2.4.0,<3.0a0']

Package harfbuzz conflicts for:
ffmpeg -> libass[version='>=0.17.1,<0.17.2.0a0'] -> harfbuzz[version='>=7.2.0,<8.0a0|>=8.1.1,<9.0a0']
libass -> harfbuzz[version='>=7.2.0,<8.0a0|>=8.1.1,<9.0a0']
ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0']
audioread -> ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0']

Package gtest conflicts for:
grpcio -> abseil-cpp[version='>=20230802.0,<20230802.1.0a0'] -> gtest[version='>=1.14.0,<1.14.1.0a0']
libprotobuf -> gtest[version='>=1.14.0,<1.14.1.0a0']
protobuf -> libprotobuf[version='>=4.23.4,<4.23.5.0a0'] -> gtest[version='>=1.14.0,<1.14.1.0a0']

Package tbb conflicts for:
numba -> tbb[version='>=2021.3.0|>=2021.5.0|>=2021.8.0']
librosa -> numba[version='>=0.51.0'] -> tbb[version='>=2021.3.0|>=2021.5.0|>=2021.8.0']
ffmpeg -> libopenvino[version='>=2024.0.0,<2024.0.1.0a0'] -> tbb[version='>=2021.11.0|>=2021.5.0']

Package soxr conflicts for:
soxr-python -> soxr[version='>=0.1.3,<0.1.4.0a0']
librosa -> soxr-python[version='>=0.3.2'] -> soxr[version='>=0.1.3,<0.1.4.0a0']

Package x265 conflicts for:
audioread -> ffmpeg -> x265[version='>=3.5,<3.6.0a0']
ffmpeg -> x265[version='>=3.5,<3.6.0a0']

Package requests conflicts for:
librosa -> pooch[version='>=1.0'] -> requests[version='>=2.19.0']
scipy -> pooch -> requests[version='>=2.19.0']
tensorflow -> tensorboard=2.12 -> requests[version='>=2.21.0,<3']
pooch -> requests[version='>=2.19.0']

Package platformdirs conflicts for:
librosa -> pooch[version='>=1.0'] -> platformdirs[version='>=2.5.0']
scipy -> pooch -> platformdirs[version='>=2.5.0']
pooch -> platformdirs[version='>=2.5.0']

Package libwebp-base conflicts for:
libtiff -> libwebp-base[version='>=1.2.3,<2.0a0|>=1.2.4,<2.0a0|>=1.3.0,<2.0a0|>=1.3.1,<2.0a0|>=1.3.2,<2.0a0']
lcms2 -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libwebp-base[version='>=1.2.3,<2.0a0|>=1.2.4,<2.0a0|>=1.3.0,<2.0a0|>=1.3.1,<2.0a0|>=1.3.2,<2.0a0']
matplotlib-base -> pillow[version='>=8'] -> libwebp-base[version='>=1.2.0,<1.3.0a0|>=1.2.2,<2.0a0|>=1.2.4,<2.0a0|>=1.3.0,<2.0a0|>=1.3.1,<2.0a0|>=1.3.2,<2.0a0']
pillow -> libwebp-base[version='>=1.2.0,<1.3.0a0|>=1.2.2,<2.0a0|>=1.2.4,<2.0a0|>=1.3.0,<2.0a0|>=1.3.1,<2.0a0|>=1.3.2,<2.0a0']
openjpeg -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libwebp-base[version='>=1.2.3,<2.0a0|>=1.2.4,<2.0a0|>=1.3.0,<2.0a0|>=1.3.1,<2.0a0|>=1.3.2,<2.0a0']
pillow -> libtiff[version='>=4.4.0,<4.5.0a0'] -> libwebp-base[version='1.1.0.*|1.2.0.*|1.2.1.*|1.2.2.*|1.2.3.*|1.2.4.*|1.3.0.*|1.3.1.*|1.3.2.*|>=1.2.3,<2.0a0']

Package h5py conflicts for:
tensorflow-deps -> h5py[version='>=3.6.0,<3.7']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> h5py[version='2.10.*|>=2.9.0|>=3.1.0,<3.2|>=3.1.0']

Package lcms2 conflicts for:
pillow -> lcms2[version='>=2.11,<3.0a0|>=2.12,<3.0a0|>=2.14,<3.0a0|>=2.15,<3.0a0|>=2.16,<3.0a0']
matplotlib-base -> pillow[version='>=8'] -> lcms2[version='>=2.11,<3.0a0|>=2.12,<3.0a0|>=2.14,<3.0a0|>=2.15,<3.0a0|>=2.16,<3.0a0']

Package cycler conflicts for:
matplotlib-base -> cycler[version='>=0.10']
librosa -> matplotlib-base[version='>=3.3.0'] -> cycler[version='>=0.10']

Package brotli-bin conflicts for:
brotli -> brotli-bin[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_1|hb547adb_0|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']
urllib3 -> brotli[version='>=1.0.9'] -> brotli-bin[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_1|hb547adb_0|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']

Package mpi4py conflicts for:
h5py -> mpi4py[version='>=3.0']
tensorflow-deps -> h5py[version='>=3.6.0,<3.7'] -> mpi4py[version='>=3.0']

Package libidn2 conflicts for:
gnutls -> libidn2[version='>=2,<3.0a0']
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> libidn2[version='>=2,<3.0a0']

Package importlib-metadata conflicts for:
librosa -> lazy_loader[version='>=0.1'] -> importlib-metadata
numba -> importlib-metadata
lazy_loader -> importlib-metadata
numba -> importlib_metadata -> importlib-metadata[version='>=1.1.3,<1.1.4.0a0|>=1.5.0,<1.5.1.0a0|>=1.5.2,<1.5.3.0a0|>=1.6.0,<1.6.1.0a0|>=1.6.1,<1.6.2.0a0|>=1.7.0,<1.7.1.0a0|>=2.0.0,<2.0.1.0a0|>=3.0.0,<3.0.1.0a0|>=3.1.0,<3.1.1.0a0|>=3.1.1,<3.1.2.0a0|>=3.10.0,<3.10.1.0a0|>=3.10.1,<3.10.2.0a0|>=4.0.1,<4.0.2.0a0|>=4.10.0,<4.10.1.0a0|>=4.10.1,<4.10.2.0a0|>=4.11.0,<4.11.1.0a0|>=4.11.1,<4.11.2.0a0|>=4.11.2,<4.11.3.0a0|>=4.11.3,<4.11.4.0a0|>=4.11.4,<4.11.5.0a0|>=4.13.0,<4.13.1.0a0|>=5.0.0,<5.0.1.0a0|>=5.1.0,<5.1.1.0a0|>=5.2.0,<5.2.1.0a0|>=6.0.0,<6.0.1.0a0|>=6.1.0,<6.1.1.0a0|>=6.10.0,<6.10.1.0a0|>=7.0.0,<7.0.1.0a0|>=7.0.1,<7.0.2.0a0|>=7.0.2,<7.0.3.0a0|>=7.1.0,<7.1.1.0a0|>=6.9.0,<6.9.1.0a0|>=6.8.0,<6.8.1.0a0|>=6.7.0,<6.7.1.0a0|>=6.6.0,<6.6.1.0a0|>=6.5.1,<6.5.2.0a0|>=6.5.0,<6.5.1.0a0|>=6.4.1,<6.4.2.0a0|>=6.4.0,<6.4.1.0a0|>=6.3.0,<6.3.1.0a0|>=6.2.1,<6.2.2.0a0|>=6.2.0,<6.2.1.0a0|>=4.9.0,<4.9.1.0a0|>=4.8.3,<4.8.4.0a0|>=4.8.2,<4.8.3.0a0|>=4.8.1,<4.8.2.0a0|>=4.8.0,<4.8.1.0a0|>=4.7.1,<4.7.2.0a0|>=4.7.0,<4.7.1.0a0|>=4.6.4,<4.6.5.0a0|>=4.6.3,<4.6.4.0a0|>=4.6.2,<4.6.3.0a0|>=4.6.1,<4.6.2.0a0|>=4.6.0,<4.6.1.0a0|>=4.5.0,<4.5.1.0a0|>=4.4.0,<4.4.1.0a0|>=4.3.1,<4.3.2.0a0|>=4.3.0,<4.3.1.0a0|>=4.2.0,<4.2.1.0a0|>=3.9.1,<3.9.2.0a0|>=3.9.0,<3.9.1.0a0|>=3.8.1,<3.8.2.0a0|>=3.8.0,<3.8.1.0a0|>=3.7.3,<3.7.4.0a0|>=3.7.2,<3.7.3.0a0|>=3.7.0,<3.7.1.0a0|>=3.6.0,<3.6.1.0a0|>=3.4.0,<3.4.1.0a0|>=3.3.0,<3.3.1.0a0']

Package grpcio conflicts for:
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> grpcio[version='1.36.*|1.37.*|1.39.*|1.40.*|1.42.*|1.43.*|1.45.*|1.46.*|1.47.*|1.51.*|1.54.*|1.59.*|>=1.8.6|>=1.48.2|>=1.24.3,<2.0|>=1.24.3']

Package libglib conflicts for:
libass -> harfbuzz[version='>=8.1.1,<9.0a0'] -> libglib[version='>=2.76.2,<3.0a0|>=2.76.4,<3.0a0|>=2.78.0,<3.0a0|>=2.78.1,<3.0a0']
cairo -> glib[version='>=2.69.1,<3.0a0'] -> libglib[version='2.70.0|2.70.0|2.70.1|2.70.2|2.70.2|2.70.2|2.70.2|2.70.2|2.72.1|2.74.0|2.74.1|2.74.1|2.76.1|2.76.2|2.76.3|2.76.4|2.78.0|2.78.1|2.78.1|2.78.2|2.78.3|2.78.4|2.78.4|2.78.4|2.80.0',build='h67e64d8_0|h67e64d8_1|h67e64d8_1|h67e64d8_2|ha1047ec_0|h4646484_1|h4646484_0|h24e9cb9_0|hd9b11f9_0|hb438215_1|hb438215_0|hb438215_0|hfc324ee_4|hfc324ee_0|hfc324ee_1|hfc324ee_2|hfc324ee_3|hfc324ee_3|h1635a5e_0|h24e9cb9_0|h24e9cb9_0|h24e9cb9_0|h14ed1c1_0|h14ed1c1_0|h67e64d8_4|h67e64d8_3|h67e64d8_0|h67e64d8_0']
ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0'] -> libglib[version='>=2.78.1,<3.0a0']
harfbuzz -> libglib[version='>=2.66.2,<3.0a0|>=2.66.4,<3.0a0|>=2.66.7,<3.0a0|>=2.68.0,<3.0a0|>=2.68.1,<3.0a0|>=2.68.3,<3.0a0|>=2.68.4,<3.0a0|>=2.70.0,<3.0a0|>=2.70.1,<3.0a0|>=2.70.2,<3.0a0|>=2.72.1,<3.0a0|>=2.74.0,<3.0a0|>=2.74.1,<3.0a0|>=2.76.2,<3.0a0|>=2.76.4,<3.0a0|>=2.78.0,<3.0a0|>=2.78.1,<3.0a0']
cairo -> libglib[version='>=2.66.2,<3.0a0|>=2.66.4,<3.0a0|>=2.68.0,<3.0a0|>=2.70.0,<3.0a0|>=2.70.2,<3.0a0|>=2.72.1,<3.0a0|>=2.74.1,<3.0a0|>=2.76.2,<3.0a0|>=2.76.4,<3.0a0|>=2.78.0,<3.0a0']
harfbuzz -> glib[version='>=2.69.1,<3.0a0'] -> libglib[version='2.70.0|2.70.0|2.70.1|2.70.2|2.70.2|2.70.2|2.70.2|2.70.2|2.72.1|2.74.0|2.74.1|2.74.1|2.76.1|2.76.2|2.76.3|2.76.4|2.78.0|2.78.1|2.78.1|2.78.2|2.78.3|2.78.4|2.78.4|2.78.4|2.80.0',build='h67e64d8_0|h67e64d8_1|h67e64d8_1|h67e64d8_2|ha1047ec_0|h4646484_1|h4646484_0|h24e9cb9_0|hd9b11f9_0|hb438215_1|hb438215_0|hb438215_0|hfc324ee_4|hfc324ee_0|hfc324ee_1|hfc324ee_2|hfc324ee_3|hfc324ee_3|h1635a5e_0|h24e9cb9_0|h24e9cb9_0|h24e9cb9_0|h14ed1c1_0|h14ed1c1_0|h67e64d8_4|h67e64d8_3|h67e64d8_0|h67e64d8_0']

Package font-ttf-inconsolata conflicts for:
fonts-conda-forge -> font-ttf-inconsolata
fonts-conda-ecosystem -> fonts-conda-forge -> font-ttf-inconsolata

Package cached_property conflicts for:
cached-property -> cached_property[version='>=1.5.2,<1.5.3.0a0']
h5py -> cached-property -> cached_property[version='>=1.5.2,<1.5.3.0a0']

Package urllib3 conflicts for:
pooch -> requests[version='>=2.19.0'] -> urllib3[version='>=1.21.1,<1.26,!=1.25.0,!=1.25.1|>=1.21.1,<1.27|>=1.21.1,<3|>=1.21.1,<2']
requests -> urllib3[version='>=1.21.1,<1.26,!=1.25.0,!=1.25.1|>=1.21.1,<1.27|>=1.21.1,<3|>=1.21.1,<2']

Package pixman conflicts for:
harfbuzz -> cairo[version='>=1.18.0,<2.0a0'] -> pixman[version='>=0.40.0,<1.0a0|>=0.42.2,<1.0a0']
cairo -> pixman[version='>=0.40.0,<1.0a0|>=0.42.2,<1.0a0']

Package cairo conflicts for:
libass -> harfbuzz[version='>=8.1.1,<9.0a0'] -> cairo[version='>=1.16.0,<2.0a0|>=1.18.0,<2.0a0']
harfbuzz -> cairo[version='>=1.16.0,<2.0.0a0|>=1.16.0,<2.0a0|>=1.18.0,<2.0a0']
ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0'] -> cairo[version='>=1.18.0,<2.0a0']

Package brotli-python conflicts for:
urllib3 -> brotli-python[version='>=1.0.9']
requests -> urllib3[version='>=1.21.1,<3'] -> brotli-python[version='>=1.0.9']

Package libaec conflicts for:
h5py -> hdf5[version='>=1.14.3,<1.14.4.0a0'] -> libaec[version='>=1.0.6,<2.0a0|>=1.1.2,<2.0a0']
hdf5 -> libaec[version='>=1.0.6,<2.0a0|>=1.1.2,<2.0a0']

Package scipy conflicts for:
librosa -> scipy[version='>=0.14.0|>=1.0.0|>=1.2.0']
tensorflow -> tensorflow-base==2.12.0=eigen_py310h0a52ebb_0 -> scipy[version='>=1.7.3,<2']
scikit-learn -> scipy[version='1.10.0.*|>=1.3.2,<1.10.0|>=1.3.2|>=1.5.0|>=1.3.2,<=1.9.3|>=1.1.0|>=0.19.1']
librosa -> scikit-learn[version='>=0.20.0'] -> scipy[version='1.10.0.*|>=1.3.2,<1.10.0|>=1.3.2|>=1.5.0|>=1.3.2,<=1.9.3|>=1.1.0|>=0.19.1|>=0.13']

Package dav1d conflicts for:
ffmpeg -> dav1d[version='>=1.0.0,<1.0.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0']
audioread -> ffmpeg -> dav1d[version='>=1.0.0,<1.0.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0']

Package pooch conflicts for:
librosa -> scipy[version='>=1.2.0'] -> pooch
scikit-learn -> scipy -> pooch
librosa -> pooch[version='>=1.0|>=1.0,<1.7']
scipy -> pooch

Package appdirs conflicts for:
librosa -> pooch[version='>=1.0'] -> appdirs[version='>=1.3.0']
scipy -> pooch -> appdirs[version='>=1.3.0']
pooch -> appdirs[version='>=1.3.0']

Package mpg123 conflicts for:
pysoundfile -> libsndfile[version='>=1.2'] -> mpg123[version='>=1.30.2,<1.31.0a0|>=1.31.1,<1.32.0a0|>=1.31.3,<1.32.0a0|>=1.32.1,<1.33.0a0']
libsndfile -> mpg123[version='>=1.30.2,<1.31.0a0|>=1.31.1,<1.32.0a0|>=1.31.3,<1.32.0a0|>=1.32.1,<1.33.0a0']

Package pthread-stubs conflicts for:
libxcb -> pthread-stubs
pillow -> libxcb[version='>=1.15,<1.16.0a0'] -> pthread-stubs

Package re2 conflicts for:
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> re2[version='>=2022.4.1,<2022.4.2.0a0|>=2022.6.1,<2022.6.2.0a0|>=2023.2.1,<2023.2.2.0a0']
grpcio -> libgrpc==1.62.1=h9c18a4f_0 -> re2[version='>=2023.2.2,<2023.2.3.0a0|>=2023.3.2,<2023.3.3.0a0']
grpcio -> re2[version='>=2022.4.1,<2022.4.2.0a0|>=2022.6.1,<2022.6.2.0a0|>=2023.2.1,<2023.2.2.0a0']

Package font-ttf-dejavu-sans-mono conflicts for:
fonts-conda-ecosystem -> fonts-conda-forge -> font-ttf-dejavu-sans-mono
fonts-conda-forge -> font-ttf-dejavu-sans-mono

Package numba conflicts for:
librosa -> resampy[version='>=0.2.2'] -> numba[version='>=0.32|>=0.47|>=0.53']
librosa -> numba[version='>=0.38.0|>=0.43.0|>=0.45.1|>=0.51.0']

Package pysocks conflicts for:
requests -> urllib3[version='>=1.21.1,<3'] -> pysocks[version='>=1.5.6,<2.0,!=1.5.7']
urllib3 -> pysocks[version='>=1.5.6,<2.0,!=1.5.7']

Package font-ttf-source-code-pro conflicts for:
fonts-conda-forge -> font-ttf-source-code-pro
fonts-conda-ecosystem -> fonts-conda-forge -> font-ttf-source-code-pro

Package lz4-c conflicts for:
zstd -> lz4-c[version='>=1.9.2,<1.9.3.0a0|>=1.9.3,<1.10.0a0|>=1.9.3,<1.9.4.0a0|>=1.9.4,<1.10.0a0']
libcurl -> zstd[version='>=1.5.5,<1.6.0a0'] -> lz4-c[version='>=1.9.3,<1.10.0a0|>=1.9.4,<1.10.0a0|>=1.9.3,<1.9.4.0a0']
libtiff -> zstd[version='>=1.5.5,<1.6.0a0'] -> lz4-c[version='>=1.9.2,<1.9.3.0a0|>=1.9.3,<1.10.0a0|>=1.9.4,<1.10.0a0|>=1.9.3,<1.9.4.0a0']

Package libopenvino-tensorflow-frontend conflicts for:
audioread -> ffmpeg -> libopenvino-tensorflow-frontend[version='>=2023.2.0,<2023.2.1.0a0|>=2023.3.0,<2023.3.1.0a0|>=2024.0.0,<2024.0.1.0a0']
ffmpeg -> libopenvino-tensorflow-frontend[version='>=2023.2.0,<2023.2.1.0a0|>=2023.3.0,<2023.3.1.0a0|>=2024.0.0,<2024.0.1.0a0']

Package libbrotlienc conflicts for:
brotli-bin -> libbrotlienc[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_1|hb547adb_0|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']
urllib3 -> brotli[version='>=1.0.9'] -> libbrotlienc[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_1|hb547adb_0|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']
brotli -> libbrotlienc[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_1|hb547adb_0|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']

Package win_inet_pton conflicts for:
urllib3 -> pysocks[version='>=1.5.6,<2.0,!=1.5.7'] -> win_inet_pton
pysocks -> win_inet_pton

Package libflac conflicts for:
libsndfile -> libflac[version='>=1.3.3,<1.4.0a0|>=1.4.1,<1.5.0a0|>=1.4.2,<1.5.0a0|>=1.4.3,<1.5.0a0']
pysoundfile -> libsndfile[version='>=1.2'] -> libflac[version='>=1.3.3,<1.4.0a0|>=1.4.1,<1.5.0a0|>=1.4.2,<1.5.0a0|>=1.4.3,<1.5.0a0']

Package p11-kit conflicts for:
gnutls -> p11-kit[version='>=0.23.21,<0.24.0a0|>=0.24.1,<0.25.0a0']
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> p11-kit[version='>=0.23.21,<0.24.0a0|>=0.24.1,<0.25.0a0']

Package python-dateutil conflicts for:
matplotlib-base -> python-dateutil[version='>=2.1|>=2.7']
librosa -> matplotlib-base[version='>=3.3.0'] -> python-dateutil[version='>=2.1|>=2.7']

Package xorg-libxdmcp conflicts for:
libxcb -> xorg-libxdmcp
pillow -> libxcb[version='>=1.15,<1.16.0a0'] -> xorg-libxdmcp

Package libnghttp2 conflicts for:
hdf5 -> libcurl[version='>=8.4.0,<9.0a0'] -> libnghttp2[version='>=1.41.0,<2.0a0|>=1.43.0,<2.0a0|>=1.46.0,<2.0a0|>=1.46.0|>=1.47.0,<2.0a0|>=1.51.0,<2.0a0|>=1.52.0|>=1.52.0,<2.0a0|>=1.58.0,<2.0a0|>=1.57.0|>=1.57.0,<2.0a0']
libcurl -> libnghttp2[version='>=1.41.0,<2.0a0|>=1.43.0,<2.0a0|>=1.47.0,<2.0a0|>=1.51.0,<2.0a0|>=1.52.0,<2.0a0|>=1.58.0,<2.0a0|>=1.57.0|>=1.57.0,<2.0a0|>=1.52.0|>=1.46.0|>=1.46.0,<2.0a0']

Package libass conflicts for:
ffmpeg -> libass[version='>=0.17.1,<0.17.2.0a0']
audioread -> ffmpeg -> libass[version='>=0.17.1,<0.17.2.0a0']

Package threadpoolctl conflicts for:
scikit-learn -> threadpoolctl[version='>=2.0.0']
librosa -> scikit-learn[version='>=0.20.0'] -> threadpoolctl[version='>=2.0.0']

Package hdf5 conflicts for:
tensorflow-deps -> h5py[version='>=3.6.0,<3.7'] -> hdf5[version='>=1.12.1,<1.12.2.0a0',build='mpi_openmpi_*|mpi_mpich_*']
h5py -> hdf5[version='>=1.10.6,<1.10.7.0a0|>=1.10.6,<1.10.7.0a0|>=1.10.6,<1.10.7.0a0|>=1.12.1,<1.12.2.0a0|>=1.12.1,<1.12.2.0a0|>=1.12.1,<1.12.2.0a0|>=1.12.2,<1.12.3.0a0|>=1.12.2,<1.12.3.0a0|>=1.12.2,<1.12.3.0a0|>=1.14.0,<1.14.1.0a0|>=1.14.0,<1.14.1.0a0|>=1.14.0,<1.14.1.0a0|>=1.14.1,<1.14.2.0a0|>=1.14.2,<1.14.4.0a0|>=1.14.2,<1.14.4.0a0|>=1.14.2,<1.14.4.0a0|>=1.14.3,<1.14.4.0a0|>=1.14.3,<1.14.4.0a0|>=1.14.3,<1.14.4.0a0|>=1.14.1,<1.14.2.0a0|>=1.14.1,<1.14.2.0a0',build='mpi_openmpi_*|mpi_openmpi_*|mpi_openmpi_*|mpi_mpich_*|mpi_mpich_*|mpi_mpich_*|mpi_openmpi_*|mpi_mpich_*|mpi_mpich_*|mpi_openmpi_*|mpi_openmpi_*|mpi_mpich_*|mpi_openmpi_*|mpi_mpich_*']

Package graphite2 conflicts for:
ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0'] -> graphite2
harfbuzz -> graphite2[version='>=1.3.14,<2.0a0']
libass -> harfbuzz[version='>=8.1.1,<9.0a0'] -> graphite2

Package libssh2 conflicts for:
hdf5 -> libcurl[version='>=8.4.0,<9.0a0'] -> libssh2[version='>=1.10.0,<2.0a0|>=1.10.0|>=1.11.0,<2.0a0|>=1.9.0,<2.0a0']
libcurl -> libssh2[version='>=1.10.0|>=1.10.0,<2.0a0|>=1.11.0,<2.0a0|>=1.9.0,<2.0a0']

Package matplotlib-base conflicts for:
librosa -> matplotlib-base[version='>=1.5.0|>=3.3.0']
librosa -> matplotlib[version='>=1.5.0'] -> matplotlib-base[version='>=3.3.2,<3.3.3.0a0|>=3.3.3,<3.3.4.0a0|>=3.3.4,<3.3.5.0a0|>=3.4.1,<3.4.2.0a0|>=3.4.2,<3.4.3.0a0|>=3.4.3,<3.4.4.0a0|>=3.5.0,<3.5.1.0a0|>=3.5.1,<3.5.2.0a0|>=3.5.2,<3.5.3.0a0|>=3.5.3,<3.5.4.0a0|>=3.6.0,<3.6.1.0a0|>=3.6.1,<3.6.2.0a0|>=3.6.2,<3.6.3.0a0|>=3.6.3,<3.6.4.0a0|>=3.7.0,<3.7.1.0a0|>=3.7.1,<3.7.2.0a0|>=3.7.2,<3.7.3.0a0|>=3.7.3,<3.7.4.0a0|>=3.8.0,<3.8.1.0a0|>=3.8.1,<3.8.2.0a0|>=3.8.2,<3.8.3.0a0|>=3.8.3,<3.8.4.0a0']

Package fftw conflicts for:
librosa -> scipy[version='>=1.2.0'] -> fftw[version='>=3.3.9,<4.0a0']
scikit-learn -> scipy -> fftw[version='>=3.3.9,<4.0a0']
scipy -> fftw[version='>=3.3.9,<4.0a0']

Package libsndfile conflicts for:
pysoundfile -> libsndfile[version='>=1.2']
librosa -> pysoundfile[version='>=0.12.1'] -> libsndfile[version='>=1.2']

Package openblas-devel conflicts for:
blas-devel -> openblas=0.3.21 -> openblas-devel[version='0.3.13|0.3.17|0.3.17|0.3.18|0.3.20|0.3.21',build='hca03da5_2|hca03da5_0|hca03da5_0|hca03da5_1']
openblas -> openblas-devel[version='0.3.13|0.3.17|0.3.17|0.3.18|0.3.20|0.3.21',build='hca03da5_2|hca03da5_0|hca03da5_0|hca03da5_1']

Package xorg-libxau conflicts for:
pillow -> libxcb[version='>=1.15,<1.16.0a0'] -> xorg-libxau
libxcb -> xorg-libxau[version='>=1.0.11,<2.0a0']

Package blas-devel conflicts for:
blas -> blas-devel==3.9.0[build='0_netlib|5_netlib|8_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_accelerate|17_osxarm64_openblas|18_osxarm64_openblas|21_osxarm64_openblas|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|20_osxarm64_accelerate|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|18_osxarm64_accelerate|17_osxarm64_accelerate|16_osxarm64_accelerate|16_osxarm64_openblas|15_osxarm64_openblas|15_osxarm64_accelerate|14_osxarm64_openblas|14_osxarm64_accelerate|13_osxarm64_openblas|12_osxarm64_accelerate|12_osxarm64_openblas|7_openblas|1_netlib']
scipy -> blas=[build=openblas] -> blas-devel==3.9.0[build='8_openblas|9_openblas|11_osxarm64_openblas|15_osxarm64_openblas|17_osxarm64_openblas|18_osxarm64_openblas|19_osxarm64_openblas|21_osxarm64_openblas|22_osxarm64_openblas|20_osxarm64_openblas|16_osxarm64_openblas|14_osxarm64_openblas|13_osxarm64_openblas|12_osxarm64_openblas|10_openblas|7_openblas']
numpy -> blas=[build=openblas] -> blas-devel==3.9.0[build='8_openblas|9_openblas|11_osxarm64_openblas|15_osxarm64_openblas|17_osxarm64_openblas|18_osxarm64_openblas|19_osxarm64_openblas|21_osxarm64_openblas|22_osxarm64_openblas|20_osxarm64_openblas|16_osxarm64_openblas|14_osxarm64_openblas|13_osxarm64_openblas|12_osxarm64_openblas|10_openblas|7_openblas']

Package font-ttf-ubuntu conflicts for:
fonts-conda-forge -> font-ttf-ubuntu
fonts-conda-ecosystem -> fonts-conda-forge -> font-ttf-ubuntu

Package nettle conflicts for:
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> nettle[version='>=3.4.1|>=3.7,<3.8.0a0|>=3.8,<3.9.0a0|>=3.8.1,<3.9.0a0|>=3.9.1,<3.10.0a0|>=3.6,<3.7.0a0|>=3.7.3,<3.8.0a0']
gnutls -> nettle[version='>=3.4.1|>=3.7,<3.8.0a0|>=3.8,<3.9.0a0|>=3.8.1,<3.9.0a0|>=3.9.1,<3.10.0a0|>=3.6,<3.7.0a0|>=3.7.3,<3.8.0a0']

Package pycparser conflicts for:
cffi -> pycparser[version='>=2.06']
pysoundfile -> cffi -> pycparser[version='>=2.06']

Package fonttools conflicts for:
librosa -> matplotlib-base[version='>=3.3.0'] -> fonttools[version='>=4.22.0']
matplotlib-base -> fonttools[version='>=4.22.0']The following specifications were found to be incompatible with your system:

  - feature:/osx-arm64::__osx==13.6.3=0
  - feature:/osx-arm64::__unix==0=0
  - feature:|@/osx-arm64::__osx==13.6.3=0
  - feature:|@/osx-arm64::__unix==0=0
  - aom -> __osx[version='>=10.9']
  - audioread -> ffmpeg -> __osx[version='>=10.9']
  - cairo -> __osx[version='>=10.9']
  - ffmpeg -> __osx[version='>=10.9']
  - gettext -> ncurses[version='>=6.4,<7.0a0'] -> __osx[version='>=10.9']
  - gmp -> __osx[version='>=10.9']
  - gnutls -> __osx[version='>=10.9']
  - grpcio -> __osx[version='>=10.9']
  - h5py -> hdf5[version='>=1.14.3,<1.14.4.0a0'] -> __osx[version='>=10.9']
  - harfbuzz -> __osx[version='>=10.9']
  - hdf5 -> __osx[version='>=10.9']
  - libass -> harfbuzz[version='>=8.1.1,<9.0a0'] -> __osx[version='>=10.9']
  - libcurl -> libnghttp2[version='>=1.58.0,<2.0a0'] -> __osx[version='>=10.9']
  - libedit -> ncurses[version='>=6.2,<7.0.0a0'] -> __osx[version='>=10.9']
  - libglib -> __osx[version='>=10.9']
  - libnghttp2 -> __osx[version='>=10.9']
  - libprotobuf -> __osx[version='>=10.9']
  - librosa -> matplotlib-base[version='>=3.3.0'] -> __osx[version='>=10.9']
  - matplotlib-base -> __osx[version='>=10.9']
  - msgpack-python -> __osx[version='>=10.9']
  - ncurses -> __osx[version='>=10.9']
  - nettle -> gmp[version='>=6.2.1,<7.0a0'] -> __osx[version='>=10.9']
  - numba -> __osx[version='>=10.9']
  - numpy -> __osx[version='>=10.9']
  - openh264 -> __osx[version='>=10.9']
  - protobuf -> __osx[version='>=10.9']
  - pysocks -> __unix
  - pysocks -> __win
  - pysoundfile -> numpy -> __osx[version='>=10.9']
  - python=3.8 -> ncurses[version='>=6.4,<7.0a0'] -> __osx[version='>=10.9']
  - readline -> ncurses[version='>=6.3,<7.0a0'] -> __osx[version='>=10.9']
  - scikit-learn -> __osx[version='>=10.9']
  - scipy -> __osx[version='>=10.9']
  - soxr-python -> numpy[version='>=1.23.5,<2.0a0'] -> __osx[version='>=10.9']
  - svt-av1 -> __osx[version='>=10.9']
  - tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> __osx[version='>=10.9']
  - urllib3 -> pysocks[version='>=1.5.6,<2.0,!=1.5.7'] -> __unix

Your installed version is: 13.6.3


(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)

(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda install -c conda-forge tensorflow

conda install -c conda-forge tensorflow -n tensyflow


freetype -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.39,<1.7.0a0']
cairo -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.38,<1.7.0a0|>=1.6.39,<1.7.0a0']
fontconfig -> freetype[version='>=2.12.1,<3.0a0'] -> libpng[version='>=1.6.39,<1.7.0a0']
ffmpeg -> freetype[version='>=2.12.1,<3.0a0'] -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.39,<1.7.0a0']
openjpeg -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.39,<1.7.0a0|>=1.6.43,<1.7.0a0']
harfbuzz -> cairo[version='>=1.18.0,<2.0a0'] -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.39,<1.7.0a0|>=1.6.38,<1.7.0a0']
fontconfig -> libpng[version='>=1.6.37,<1.7.0a0']
pillow -> freetype[version='>=2.12.1,<3.0a0'] -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.39,<1.7.0a0|>=1.6.43,<1.7.0a0']
matplotlib-base -> freetype[version='>=2.12.1,<3.0a0'] -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.39,<1.7.0a0']
libass -> freetype[version='>=2.12.1,<3.0a0'] -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.39,<1.7.0a0']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> libpng[version='>=1.6.37,<1.7.0a0|>=1.6.38,<1.7.0a0|>=1.6.39,<1.7.0a0']

Package libbrotlicommon conflicts for:
brotli-bin -> libbrotlidec==1.1.0=hb547adb_1 -> libbrotlicommon[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_0|hb547adb_1|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']
libbrotlidec -> libbrotlicommon[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_0|hb547adb_1|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']
libbrotlienc -> libbrotlicommon[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_0|hb547adb_1|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']
brotli -> libbrotlidec==1.1.0=hb547adb_1 -> libbrotlicommon[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_0|hb547adb_1|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']

Package h5py conflicts for:
tensorflow-deps -> h5py[version='>=3.6.0,<3.7']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> h5py[version='2.10.*|>=2.9.0|>=3.1.0,<3.2|>=3.1.0']

Package libasprintf-devel conflicts for:
libflac -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf-devel==0.22.5=h8fbad5d_2
gnutls -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf-devel==0.22.5=h8fbad5d_2
gettext -> libasprintf-devel==0.22.5=h8fbad5d_2
libglib -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf-devel==0.22.5=h8fbad5d_2
libidn2 -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf-devel==0.22.5=h8fbad5d_2

Package libaec conflicts for:
h5py -> hdf5[version='>=1.14.3,<1.14.4.0a0'] -> libaec[version='>=1.0.6,<2.0a0|>=1.1.2,<2.0a0']
hdf5 -> libaec[version='>=1.0.6,<2.0a0|>=1.1.2,<2.0a0']

Package snappy conflicts for:
ffmpeg -> libopenvino-tensorflow-frontend[version='>=2024.0.0,<2024.0.1.0a0'] -> snappy[version='>=1.1.10,<2.0a0']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> snappy[version='>=1.1.10,<2.0a0|>=1.1.9,<2.0a0|>=1.1.8,<2.0a0']

Package ffmpeg conflicts for:
audioread -> ffmpeg
librosa -> audioread[version='>=2.1.9'] -> ffmpeg

Package libopenvino conflicts for:
ffmpeg -> libopenvino-arm-cpu-plugin[version='>=2024.0.0,<2024.0.1.0a0'] -> libopenvino[version='2023.2.0|2023.2.0|2023.2.0|2023.2.0|2023.2.0|2023.2.0|2023.3.0|2023.3.0|2023.3.0|2023.3.0|2023.3.0|2024.0.0',build='he6dadac_1|he6dadac_2|he6dadac_4|he6dadac_5|he6dadac_0|he6dadac_1|he6dadac_2|he6dadac_4|he6dadac_0|he6dadac_1|he6dadac_4|he6dadac_3|he6dadac_3|he6dadac_3|h965bd2d_0']
ffmpeg -> libopenvino[version='>=2023.2.0,<2023.2.1.0a0|>=2023.3.0,<2023.3.1.0a0|>=2024.0.0,<2024.0.1.0a0']
audioread -> ffmpeg -> libopenvino[version='>=2023.2.0,<2023.2.1.0a0|>=2023.3.0,<2023.3.1.0a0|>=2024.0.0,<2024.0.1.0a0']

Package krb5 conflicts for:
hdf5 -> libcurl[version='>=8.4.0,<9.0a0'] -> krb5[version='>=1.17.1,<1.18.0a0|>=1.19.1,<1.20.0a0|>=1.19.2,<1.20.0a0|>=1.19.4,<1.20.0a0|>=1.20.1,<1.21.0a0|>=1.21.2,<1.22.0a0|>=1.21.1,<1.22.0a0|>=1.19.3,<1.20.0a0']
libcurl -> krb5[version='>=1.17.1,<1.18.0a0|>=1.19.1,<1.20.0a0|>=1.19.2,<1.20.0a0|>=1.19.3,<1.20.0a0|>=1.20.1,<1.21.0a0|>=1.21.1,<1.22.0a0|>=1.21.2,<1.22.0a0|>=1.19.4,<1.20.0a0']

Package fftw conflicts for:
scikit-learn -> scipy -> fftw[version='>=3.3.9,<4.0a0']
librosa -> scipy[version='>=1.2.0'] -> fftw[version='>=3.3.9,<4.0a0']
scipy -> fftw[version='>=3.3.9,<4.0a0']

Package brotli conflicts for:
urllib3 -> brotli[version='>=1.0.9']
matplotlib-base -> fonttools[version='>=4.22.0'] -> brotli[version='>=1.0.1']
fonttools -> brotli[version='>=1.0.1']

Package gmp conflicts for:
audioread -> ffmpeg -> gmp[version='>=6.2.1,<7.0a0|>=6.3.0,<7.0a0']
gnutls -> gmp[version='>=6.2.1,<7.0a0']
ffmpeg -> gmp[version='>=6.2.1,<7.0a0|>=6.3.0,<7.0a0']
nettle -> gmp[version='>=6.2.1,<7.0a0']

Package pyparsing conflicts for:
librosa -> matplotlib-base[version='>=3.3.0'] -> pyparsing[version='<3,>=2.0.2|>=2.0.2,!=3.0.5|>=2.0.3,!=2.0.4,!=2.1.2,!=2.1.6|>=2.2.1|>=2.3.1|>=2.3.1,<3.1|>=2.0.2,<3|>=2.0.2']
pooch -> packaging[version='>=20.0'] -> pyparsing[version='<3,>=2.0.2|>=2.0.2,!=3.0.5|>=2.0.2,<3|>=2.0.2']
lazy_loader -> packaging -> pyparsing[version='<3,>=2.0.2|>=2.0.2,!=3.0.5|>=2.0.2,<3|>=2.0.2']
wheel -> packaging[version='>=20.2'] -> pyparsing[version='<3,>=2.0.2|>=2.0.2,!=3.0.5|>=2.0.2,<3|>=2.0.2']
matplotlib-base -> pyparsing[version='>=2.0.3,!=2.0.4,!=2.1.2,!=2.1.6|>=2.2.1|>=2.3.1|>=2.3.1,<3.1']
matplotlib-base -> packaging[version='>=20.0'] -> pyparsing[version='<3,>=2.0.2|>=2.0.2,!=3.0.5|>=2.0.2,<3|>=2.0.2']
packaging -> pyparsing[version='<3,>=2.0.2|>=2.0.2,!=3.0.5|>=2.0.2,<3|>=2.0.2']

Package win_inet_pton conflicts for:
urllib3 -> pysocks[version='>=1.5.6,<2.0,!=1.5.7'] -> win_inet_pton
pysocks -> win_inet_pton

Package wheel conflicts for:
python=3.8 -> pip -> wheel
pip -> wheel
tensorflow -> tensorflow-base==2.6.2=cpu_py39hd39b1ba_2 -> wheel[version='>=0.26|>=0.35|>=0.35,<1|>=0.35,<0.36']

Package setuptools conflicts for:
wheel -> setuptools
grpcio -> setuptools
numba -> setuptools
python=3.8 -> pip -> setuptools
librosa -> setuptools
joblib -> setuptools
librosa -> resampy[version='>=0.2.2'] -> setuptools[version='>=48']
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> setuptools
pip -> setuptools
matplotlib-base -> setuptools
tensorflow -> tensorboard=2.12 -> setuptools[version='>=41.0.0']
protobuf -> setuptools
scikit-learn -> joblib[version='>=1.2.0'] -> setuptools

Package gettext-tools conflicts for:
libglib -> gettext[version='>=0.21.1,<1.0a0'] -> gettext-tools==0.22.5=h8fbad5d_2
gnutls -> gettext[version='>=0.21.1,<1.0a0'] -> gettext-tools==0.22.5=h8fbad5d_2
libflac -> gettext[version='>=0.21.1,<1.0a0'] -> gettext-tools==0.22.5=h8fbad5d_2
libidn2 -> gettext[version='>=0.21.1,<1.0a0'] -> gettext-tools==0.22.5=h8fbad5d_2
gettext -> gettext-tools==0.22.5=h8fbad5d_2

Package ca-certificates conflicts for:
python=3.8 -> openssl[version='>=3.2.1,<4.0a0'] -> ca-certificates
libssh2 -> openssl[version='>=3.1.1,<4.0a0'] -> ca-certificates
ffmpeg -> gnutls[version='>=3.6.13,<3.7.0a0'] -> ca-certificates
grpcio -> openssl[version='>=3.0.7,<4.0a0'] -> ca-certificates
tensorflow -> openssl[version='>=1.1.1l,<1.1.2a'] -> ca-certificates
gnutls -> ca-certificates
libcurl -> openssl[version='>=3.2.1,<4.0a0'] -> ca-certificates
krb5 -> openssl[version='>=3.1.2,<4.0a0'] -> ca-certificates
libnghttp2 -> openssl[version='>=3.2.0,<4.0a0'] -> ca-certificates
openssl -> ca-certificates
hdf5 -> openssl[version='>=3.2.0,<4.0a0'] -> ca-certificates

Package x264 conflicts for:
ffmpeg -> x264[version='>=1!152.20180806,<1!153|>=1!161.3030,<1!162|>=1!164.3095,<1!165']
audioread -> ffmpeg -> x264[version='>=1!152.20180806,<1!153|>=1!161.3030,<1!162|>=1!164.3095,<1!165']

Package pcre2 conflicts for:
cairo -> libglib[version='>=2.78.0,<3.0a0'] -> pcre2[version='>=10.37,<10.38.0a0|>=10.40,<10.41.0a0|>=10.42,<10.43.0a0|>=10.43,<10.44.0a0']
libglib -> pcre2[version='>=10.37,<10.38.0a0|>=10.40,<10.41.0a0|>=10.42,<10.43.0a0|>=10.43,<10.44.0a0']
harfbuzz -> libglib[version='>=2.78.1,<3.0a0'] -> pcre2[version='>=10.37,<10.38.0a0|>=10.40,<10.41.0a0|>=10.42,<10.43.0a0|>=10.43,<10.44.0a0']

Package certifi conflicts for:
urllib3 -> certifi
joblib -> setuptools -> certifi[version='>=2016.9.26']
protobuf -> setuptools -> certifi[version='>=2016.9.26']
matplotlib-base -> certifi[version='>=2020.06.20']
setuptools -> certifi[version='>=2016.9.26']
matplotlib-base -> setuptools -> certifi[version='>=2016.9.26']
grpcio -> setuptools -> certifi[version='>=2016.9.26']
pooch -> requests[version='>=2.19.0'] -> certifi[version='>=2017.4.17']
pip -> setuptools -> certifi[version='>=2016.9.26']
numba -> setuptools -> certifi[version='>=2016.9.26']
librosa -> matplotlib-base[version='>=3.3.0'] -> certifi[version='>=2016.9.26|>=2020.06.20']
wheel -> setuptools -> certifi[version='>=2016.9.26']

Package lz4-c conflicts for:
zstd -> lz4-c[version='>=1.9.2,<1.9.3.0a0|>=1.9.3,<1.10.0a0|>=1.9.3,<1.9.4.0a0|>=1.9.4,<1.10.0a0']
libcurl -> zstd[version='>=1.5.5,<1.6.0a0'] -> lz4-c[version='>=1.9.3,<1.10.0a0|>=1.9.4,<1.10.0a0|>=1.9.3,<1.9.4.0a0']
libtiff -> zstd[version='>=1.5.5,<1.6.0a0'] -> lz4-c[version='>=1.9.2,<1.9.3.0a0|>=1.9.3,<1.10.0a0|>=1.9.4,<1.10.0a0|>=1.9.3,<1.9.4.0a0']

Package unicodedata2 conflicts for:
matplotlib-base -> fonttools[version='>=4.22.0'] -> unicodedata2[version='>=13.0.0|>=14.0.0']
fonttools -> unicodedata2[version='>=12.1.0|>=13.0.0|>=14.0.0']

Package libllvm10 conflicts for:
numba -> llvmlite[version='>=0.36.0,<0.37.0a0'] -> libllvm10[version='>=10.0.1,<10.1.0a0']
llvmlite -> libllvm10[version='>=10.0.1,<10.1.0a0']

Package typing-extensions conflicts for:
platformdirs -> typing-extensions[version='>=4.4|>=4.5|>=4.6.3']
pooch -> platformdirs[version='>=2.5.0'] -> typing-extensions[version='>=4.4|>=4.5|>=4.6.3']

Package openh264 conflicts for:
ffmpeg -> openh264[version='>=1.8.0,<1.9.0a0|>=2.1.1,<2.2.0a0|>=2.2.0,<2.3.0a0|>=2.3.0,<2.3.1.0a0|>=2.3.1,<2.3.2.0a0|>=2.4.0,<2.4.1.0a0|>=2.4.1,<2.4.2.0a0']
audioread -> ffmpeg -> openh264[version='>=1.8.0,<1.9.0a0|>=2.1.1,<2.2.0a0|>=2.2.0,<2.3.0a0|>=2.3.0,<2.3.1.0a0|>=2.3.1,<2.3.2.0a0|>=2.4.0,<2.4.1.0a0|>=2.4.1,<2.4.2.0a0']

Package cached-property conflicts for:
tensorflow-deps -> h5py[version='>=3.6.0,<3.7'] -> cached-property
h5py -> cached-property

Package grpcio conflicts for:
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> grpcio[version='1.36.*|1.37.*|1.39.*|1.40.*|1.42.*|1.43.*|1.45.*|1.46.*|1.47.*|1.51.*|1.54.*|1.59.*|>=1.8.6|>=1.48.2|>=1.24.3,<2.0|>=1.24.3']
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0']

Package jbig conflicts for:
libtiff -> jbig
lcms2 -> libtiff[version='>=4.2.0,<4.5.0a0'] -> jbig
pillow -> libtiff[version='>=4.3.0,<4.5.0a0'] -> jbig
openjpeg -> libtiff[version='>=4.3.0,<4.5.0a0'] -> jbig

Package openblas-devel conflicts for:
openblas -> openblas-devel[version='0.3.13|0.3.17|0.3.17|0.3.18|0.3.20|0.3.21',build='hca03da5_2|hca03da5_0|hca03da5_1|hca03da5_0']
blas-devel -> openblas=0.3.21 -> openblas-devel[version='0.3.13|0.3.17|0.3.17|0.3.18|0.3.20|0.3.21',build='hca03da5_2|hca03da5_0|hca03da5_1|hca03da5_0']

Package blas-devel conflicts for:
blas -> blas-devel==3.9.0[build='0_netlib|1_netlib|5_netlib|9_openblas|11_osxarm64_openblas|12_osxarm64_accelerate|13_osxarm64_accelerate|13_osxarm64_openblas|14_osxarm64_accelerate|14_osxarm64_openblas|15_osxarm64_accelerate|15_osxarm64_openblas|16_osxarm64_accelerate|18_osxarm64_accelerate|18_osxarm64_openblas|20_osxarm64_accelerate|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|21_osxarm64_openblas|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|17_osxarm64_accelerate|17_osxarm64_openblas|16_osxarm64_openblas|12_osxarm64_openblas|10_openblas|8_openblas|7_openblas']
numpy -> blas=[build=openblas] -> blas-devel==3.9.0[build='9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_openblas|15_osxarm64_openblas|16_osxarm64_openblas|18_osxarm64_openblas|20_osxarm64_openblas|22_osxarm64_openblas|21_osxarm64_openblas|19_osxarm64_openblas|17_osxarm64_openblas|14_osxarm64_openblas|12_osxarm64_openblas|8_openblas|7_openblas']
scipy -> blas=[build=openblas] -> blas-devel==3.9.0[build='9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_openblas|15_osxarm64_openblas|16_osxarm64_openblas|18_osxarm64_openblas|20_osxarm64_openblas|22_osxarm64_openblas|21_osxarm64_openblas|19_osxarm64_openblas|17_osxarm64_openblas|14_osxarm64_openblas|12_osxarm64_openblas|8_openblas|7_openblas']

Package munkres conflicts for:
fonttools -> munkres
matplotlib-base -> fonttools[version='>=4.22.0'] -> munkres

Package libsndfile conflicts for:
librosa -> pysoundfile[version='>=0.12.1'] -> libsndfile[version='>=1.2']
pysoundfile -> libsndfile[version='>=1.2']

Package libwebp-base conflicts for:
openjpeg -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libwebp-base[version='>=1.2.3,<2.0a0|>=1.2.4,<2.0a0|>=1.3.0,<2.0a0|>=1.3.1,<2.0a0|>=1.3.2,<2.0a0']
libtiff -> libwebp-base[version='>=1.2.3,<2.0a0|>=1.2.4,<2.0a0|>=1.3.0,<2.0a0|>=1.3.1,<2.0a0|>=1.3.2,<2.0a0']
pillow -> libtiff[version='>=4.4.0,<4.5.0a0'] -> libwebp-base[version='1.1.0.*|1.2.0.*|1.2.1.*|1.2.2.*|1.2.3.*|1.2.4.*|1.3.0.*|1.3.1.*|1.3.2.*|>=1.2.3,<2.0a0']
pillow -> libwebp-base[version='>=1.2.0,<1.3.0a0|>=1.2.2,<2.0a0|>=1.2.4,<2.0a0|>=1.3.0,<2.0a0|>=1.3.1,<2.0a0|>=1.3.2,<2.0a0']
matplotlib-base -> pillow[version='>=8'] -> libwebp-base[version='>=1.2.0,<1.3.0a0|>=1.2.2,<2.0a0|>=1.2.4,<2.0a0|>=1.3.0,<2.0a0|>=1.3.1,<2.0a0|>=1.3.2,<2.0a0']
lcms2 -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libwebp-base[version='>=1.2.3,<2.0a0|>=1.2.4,<2.0a0|>=1.3.0,<2.0a0|>=1.3.1,<2.0a0|>=1.3.2,<2.0a0']

Package libgfortran5 conflicts for:
libblas -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=12.2.0|>=12.3.0|>=11.1.0']
libopenblas -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.3.0|>=12.3.0|>=11.2.0|>=11.1.0']
libblas -> libgfortran=5 -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|13.2.0|13.2.0|12.3.0|>=11.3.0|>=11.2.0',build='ha3a6a3e_1|hf226fd6_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|h76267eb_1']
numpy -> libblas[version='>=3.9.0,<4.0a0'] -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|>=11.0.0.dev0|>=11.0.1.dev0|>=12.2.0|>=12.3.0|>=13.2.0|>=11.3.0|>=11.2.0|13.2.0|13.2.0|12.3.0',build='ha3a6a3e_1|hf226fd6_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|h76267eb_1']
scipy -> libgfortran=5 -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|13.2.0|13.2.0|12.3.0',build='ha3a6a3e_1|hf226fd6_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|h76267eb_1']
libopenblas -> libgfortran=5 -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|13.2.0|13.2.0|12.3.0',build='ha3a6a3e_1|hf226fd6_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|h76267eb_1']
numpy -> libgfortran5[version='>=11.1.0']
scikit-learn -> scipy -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.3.0|>=12.2.0|>=12.3.0|>=13.2.0|>=11.2.0|>=11.1.0']
pysoundfile -> numpy -> libgfortran5[version='>=11.1.0']
hdf5 -> libgfortran=5 -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|13.2.0|13.2.0|12.3.0|>=11.2.0',build='ha3a6a3e_1|hf226fd6_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|h76267eb_1']
h5py -> hdf5[version='>=1.14.3,<1.14.4.0a0'] -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.1.0|>=11.2.0|>=11.3.0|>=12.2.0|>=12.3.0|>=13.2.0']
blas -> libgfortran=5 -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|13.2.0|13.2.0|12.3.0|>=11.1.0',build='ha3a6a3e_1|hf226fd6_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|h76267eb_1']
blas -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=12.2.0|>=12.3.0|>=13.2.0']
libcblas -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.1.0']
hdf5 -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.3.0|>=12.2.0|>=12.3.0|>=13.2.0|>=11.1.0']
liblapacke -> libblas==3.9.0=22_osxarm64_accelerate -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|>=12.2.0|>=12.3.0|13.2.0|13.2.0|12.3.0',build='ha3a6a3e_1|hf226fd6_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|h76267eb_1']
blas-devel -> libblas==3.9.0=22_osxarm64_accelerate -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.3.0|>=12.2.0|>=12.3.0|>=11.1.0']
libgfortran -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|13.2.0|13.2.0|12.3.0',build='ha3a6a3e_1|hf226fd6_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|h76267eb_1']
matplotlib-base -> numpy[version='>=1.21,<2'] -> libgfortran5[version='>=11.1.0']
librosa -> numpy[version='>=1.20.3,!=1.22.0,!=1.22.1,!=1.22.2'] -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.1.0|>=12.3.0|>=13.2.0|>=12.2.0|>=11.3.0|>=11.2.0']
liblapacke -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.1.0']
numba -> numpy[version='>=1.20.3,<2.0a0'] -> libgfortran5[version='>=11.1.0']
openblas -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.3.0|>=12.3.0']
scipy -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.3.0|>=12.2.0|>=12.3.0|>=13.2.0|>=11.2.0|>=11.1.0']
liblapack -> libgfortran5[version='>=11.0.0.dev0|>=11.0.1.dev0|>=11.1.0']
libcblas -> libblas==3.9.0=22_osxarm64_accelerate -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|>=12.2.0|>=12.3.0|13.2.0|13.2.0|12.3.0',build='ha3a6a3e_1|hf226fd6_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|h76267eb_1']
liblapack -> libblas==3.9.0=22_osxarm64_accelerate -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|>=12.2.0|>=12.3.0|13.2.0|13.2.0|12.3.0',build='ha3a6a3e_1|hf226fd6_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|h76267eb_1']
openblas -> libgfortran=5 -> libgfortran5[version='11.4.0|12.3.0|12.3.0|13.2.0|13.2.0|13.2.0|12.3.0|>=11.2.0|>=11.1.0',build='ha3a6a3e_1|hf226fd6_1|hf226fd6_3|ha3a6a3e_3|hf226fd6_2|ha3a6a3e_2|h76267eb_1']

Package libxcb conflicts for:
matplotlib-base -> pillow[version='>=8'] -> libxcb[version='>=1.13,<1.14.0a0|>=1.15,<1.16.0a0']
pillow -> libxcb[version='>=1.13,<1.14.0a0|>=1.15,<1.16.0a0']

Package aom conflicts for:
ffmpeg -> aom[version='>=3.2.0,<3.3.0a0|>=3.3.0,<3.4.0a0|>=3.4.0,<3.5.0a0|>=3.5.0,<3.6.0a0|>=3.6.1,<3.7.0a0|>=3.7.0,<3.8.0a0|>=3.7.1,<3.8.0a0|>=3.8.1,<3.9.0a0|>=3.8.2,<3.9.0a0']
audioread -> ffmpeg -> aom[version='>=3.2.0,<3.3.0a0|>=3.3.0,<3.4.0a0|>=3.4.0,<3.5.0a0|>=3.5.0,<3.6.0a0|>=3.6.1,<3.7.0a0|>=3.7.0,<3.8.0a0|>=3.7.1,<3.8.0a0|>=3.8.1,<3.9.0a0|>=3.8.2,<3.9.0a0']

Package libintl conflicts for:
libidn2 -> gettext[version='>=0.21.1,<1.0a0'] -> libintl==0.22.5=h8fbad5d_2
gettext-tools -> libintl==0.22.5=h8fbad5d_2
libgettextpo -> libintl==0.22.5=h8fbad5d_2
libintl-devel -> libintl==0.22.5=h8fbad5d_2
libglib -> gettext[version='>=0.21.1,<1.0a0'] -> libintl==0.22.5=h8fbad5d_2
gettext -> libintl==0.22.5=h8fbad5d_2
gnutls -> gettext[version='>=0.21.1,<1.0a0'] -> libintl==0.22.5=h8fbad5d_2
harfbuzz -> libglib[version='>=2.78.1,<3.0a0'] -> libintl[version='>=0.22.5,<1.0a0']
libgettextpo-devel -> libintl==0.22.5=h8fbad5d_2
libflac -> gettext[version='>=0.21.1,<1.0a0'] -> libintl==0.22.5=h8fbad5d_2
libglib -> libintl[version='>=0.22.5,<1.0a0']
cairo -> libglib[version='>=2.78.0,<3.0a0'] -> libintl[version='>=0.22.5,<1.0a0']

Package gnutls conflicts for:
ffmpeg -> gnutls[version='>=3.6.13,<3.7.0a0|>=3.7.6,<3.8.0a0|>=3.7.7,<3.8.0a0|>=3.7.8,<3.8.0a0|>=3.7.9,<3.8.0a0|>=3.6.15,<3.7.0a0']
audioread -> ffmpeg -> gnutls[version='>=3.6.13,<3.7.0a0|>=3.7.6,<3.8.0a0|>=3.7.7,<3.8.0a0|>=3.7.8,<3.8.0a0|>=3.7.9,<3.8.0a0|>=3.6.15,<3.7.0a0']

Package libprotobuf conflicts for:
grpcio -> libprotobuf[version='>=3.20.1,<3.21.0a0|>=3.21.10,<3.22.0a0|>=3.21.12,<3.22.0a0|>=3.21.9,<3.22.0a0|>=3.21.8,<3.22.0a0|>=3.21.5,<3.22.0a0|>=3.20.3,<3.21.0a0']
protobuf -> libprotobuf[version='3.13.0.1.*|3.14.0.*|3.15.0.*|3.15.1.*|3.15.2.*|3.15.3.*|3.15.4.*|3.15.5.*|3.15.6.*|3.15.7.*|3.15.8.*|3.16.0.*|3.17.0.*|3.17.1.*|3.17.2.*|3.18.0.*|3.18.1.*|3.18.3.*|3.19.1.*|3.19.2.*|3.19.3.*|3.19.4.*|3.19.6.*|3.20.0.*|3.20.1.*|3.20.2.*|3.20.3.*|3.21.1.*|3.21.10.*|3.21.11.*|3.21.12.*|>=4.22.5,<4.23.0a0|>=4.23.1,<4.24.0a0|>=4.23.2,<4.23.3.0a0|>=4.23.3,<4.23.4.0a0|>=4.23.4,<4.23.5.0a0|>=4.24.3,<4.24.4.0a0|>=4.24.4,<4.24.5.0a0|>=4.25.1,<4.25.2.0a0|>=4.25.2,<4.25.3.0a0|>=4.25.3,<4.25.4.0a0|>=3.21.12,<3.22.0a0|>=3.21.11,<3.22.0a0|>=3.21.10,<3.22.0a0|3.21.9.*|>=3.21.9,<3.22.0a0|3.21.8.*|>=3.21.8,<3.22.0a0|3.21.7.*|>=3.21.7,<3.22.0a0|3.21.6.*|>=3.21.6,<3.22.0a0|3.21.5.*|>=3.21.5,<3.22.0a0|3.21.4.*|>=3.21.4,<3.22.0a0|3.21.3.*|>=3.21.3,<3.22.0a0|3.21.2.*|>=3.21.2,<3.22.0a0|>=3.21.1,<3.22.0a0|>=3.20.3,<3.21.0a0|>=3.20.2,<3.21.0a0|>=3.20.1,<3.21.0a0|>=3.20.0,<3.21.0a0|>=3.19.6,<3.20.0a0|>=3.19.4,<3.20.0a0|>=3.19.3,<3.20.0a0|>=3.19.2,<3.20.0a0|>=3.19.1,<3.20.0a0|>=3.18.3,<3.19.0a0|>=3.18.1,<3.19.0a0|>=3.18.0,<3.19.0a0|>=3.17.2,<3.18.0a0|>=3.17.1,<3.18.0a0|>=3.17.0,<3.18.0a0|>=3.16.0,<3.17.0a0|>=3.15.8,<3.16.0a0|>=3.15.7,<3.16.0a0|>=3.15.6,<3.16.0a0|>=3.15.5,<3.16.0a0|>=3.15.4,<3.16.0a0|>=3.15.3,<3.16.0a0|>=3.15.2,<3.16.0a0|>=3.15.1,<3.16.0a0|>=3.15.0,<3.16.0a0|>=3.14.0,<3.15.0a0|>=3.13.0.1,<3.14.0a0']
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> libprotobuf[version='3.19.1.*|3.19.2.*|3.19.3.*|3.19.4.*|3.19.6.*|>=3.20.1,<3.21.0a0|>=3.21.10,<3.22.0a0|>=3.21.12,<3.22.0a0|>=3.21.9,<3.22.0a0|>=3.21.8,<3.22.0a0|>=3.21.5,<3.22.0a0|>=3.20.3,<3.21.0a0|>=3.19.6,<3.20.0a0|>=3.19.4,<3.20.0a0|>=3.19.3,<3.20.0a0|>=3.19.2,<3.20.0a0|>=3.19.1,<3.20.0a0']
grpcio -> libgrpc==1.62.1=h9c18a4f_0 -> libprotobuf[version='>=3.20.3,<3.20.4.0a0|>=4.23.1,<4.24.0a0|>=4.23.2,<4.23.3.0a0|>=4.23.3,<4.23.4.0a0|>=4.23.4,<4.23.5.0a0|>=4.24.3,<4.24.4.0a0|>=4.24.4,<4.24.5.0a0|>=4.25.1,<4.25.2.0a0|>=4.25.2,<4.25.3.0a0|>=4.25.3,<4.25.4.0a0']
ffmpeg -> libopenvino-onnx-frontend[version='>=2024.0.0,<2024.0.1.0a0'] -> libprotobuf[version='>=4.24.3,<4.24.4.0a0|>=4.24.4,<4.24.5.0a0|>=4.25.1,<4.25.2.0a0|>=4.25.2,<4.25.3.0a0|>=4.25.3,<4.25.4.0a0']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> libprotobuf[version='>=3.15.7,<3.16.0a0|>=3.15.8,<3.16.0a0|>=3.16.0,<3.17.0a0|>=3.18.1,<3.19.0a0|>=3.19.4,<3.20.0a0|>=3.20.1,<3.21.0a0|>=3.21.12,<3.22.0a0|>=4.24.4,<4.24.5.0a0|>=3.21.6,<3.22.0a0|>=3.20.3,<3.21.0a0']

Package six conflicts for:
lazy_loader -> packaging -> six
packaging -> six
librosa -> six[version='>=1.3']
python-dateutil -> six[version='>=1.5']
grpcio -> six[version='>=1.5.2|>=1.6.0']
cycler -> six
pooch -> packaging[version='>=20.0'] -> six
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> six[version='1.15.0.*|>=1.10.0|>=1.12|>=1.15,<1.16|>=1.15.0']
urllib3 -> cryptography[version='>=1.3.4'] -> six[version='>=1.4.1|>=1.5.2']
protobuf -> six
wheel -> packaging[version='>=20.2'] -> six
h5py -> six
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> six[version='>=1.5.2|>=1.6.0']
librosa -> packaging[version='>=20.0'] -> six
zipp -> more-itertools -> six[version='>=1.0.0,<2.0.0']
matplotlib-base -> cycler[version='>=0.10'] -> six[version='>=1.5']

Package libtiff conflicts for:
matplotlib-base -> pillow[version='>=8'] -> libtiff[version='>=4.1.0,<4.4.0a0|>=4.2.0,<4.4.0a0|>=4.3.0,<4.4.0a0|>=4.3.0,<4.5.0a0|>=4.4.0,<4.5.0a0|>=4.5.0,<4.6.0a0|>=4.5.1,<4.6.0a0|>=4.6.0,<4.7.0a0|>=4.2.0,<5.0a0']
pillow -> libtiff[version='>=4.1.0,<4.4.0a0|>=4.2.0,<4.4.0a0|>=4.3.0,<4.4.0a0|>=4.3.0,<4.5.0a0|>=4.4.0,<4.5.0a0|>=4.5.0,<4.6.0a0|>=4.5.1,<4.6.0a0|>=4.6.0,<4.7.0a0|>=4.2.0,<5.0a0']
openjpeg -> libtiff[version='>=4.1.0,<4.5.0a0|>=4.2.0,<4.5.0a0|>=4.3.0,<4.5.0a0|>=4.4.0,<4.5.0a0|>=4.5.0,<4.6.0a0|>=4.6.0,<4.7.0a0|>=4.2.0,<5.0a0']
pillow -> lcms2[version='>=2.12,<3.0a0'] -> libtiff[version='>=4.1.0,<4.5.0a0|>=4.2.0,<4.5.0a0']
lcms2 -> libtiff[version='>=4.1.0,<4.5.0a0|>=4.2.0,<4.5.0a0|>=4.4.0,<4.5.0a0|>=4.5.0,<4.6.0a0|>=4.6.0,<4.7.0a0|>=4.2.0,<5.0a0']

Package libjpeg-turbo conflicts for:
matplotlib-base -> pillow[version='>=8'] -> libjpeg-turbo[version='>=2.1.4,<3.0a0|>=2.1.5.1,<3.0a0|>=3.0.0,<4.0a0']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> libjpeg-turbo[version='>=2.1.5.1,<3.0a0|>=3.0.0,<4.0a0']
lcms2 -> libjpeg-turbo[version='>=2.1.5.1,<3.0a0|>=3.0.0,<4.0a0']
pillow -> libjpeg-turbo[version='>=2.1.4,<3.0a0|>=2.1.5.1,<3.0a0|>=3.0.0,<4.0a0']
lcms2 -> libtiff[version='>=4.5.0,<4.6.0a0'] -> libjpeg-turbo[version='>=2.1.4,<3.0a0']
openjpeg -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libjpeg-turbo[version='>=2.1.4,<3.0a0|>=2.1.5.1,<3.0a0|>=3.0.0,<4.0a0']
libtiff -> libjpeg-turbo[version='>=2.1.4,<3.0a0|>=2.1.5.1,<3.0a0|>=3.0.0,<4.0a0']

Package libglib conflicts for:
harfbuzz -> libglib[version='>=2.66.2,<3.0a0|>=2.66.4,<3.0a0|>=2.66.7,<3.0a0|>=2.68.0,<3.0a0|>=2.68.1,<3.0a0|>=2.68.3,<3.0a0|>=2.68.4,<3.0a0|>=2.70.0,<3.0a0|>=2.70.1,<3.0a0|>=2.70.2,<3.0a0|>=2.72.1,<3.0a0|>=2.74.0,<3.0a0|>=2.74.1,<3.0a0|>=2.76.2,<3.0a0|>=2.76.4,<3.0a0|>=2.78.0,<3.0a0|>=2.78.1,<3.0a0']
libass -> harfbuzz[version='>=8.1.1,<9.0a0'] -> libglib[version='>=2.76.2,<3.0a0|>=2.76.4,<3.0a0|>=2.78.0,<3.0a0|>=2.78.1,<3.0a0']
cairo -> glib[version='>=2.69.1,<3.0a0'] -> libglib[version='2.70.0|2.70.0|2.70.1|2.70.2|2.70.2|2.70.2|2.70.2|2.70.2|2.72.1|2.74.0|2.74.1|2.74.1|2.76.1|2.76.2|2.76.3|2.76.4|2.78.0|2.78.1|2.78.1|2.78.2|2.78.3|2.78.4|2.78.4|2.78.4|2.80.0',build='h67e64d8_0|h67e64d8_1|h67e64d8_0|h67e64d8_0|h67e64d8_1|h67e64d8_3|h67e64d8_4|h4646484_0|h24e9cb9_0|h24e9cb9_0|hd9b11f9_0|hfc324ee_3|hfc324ee_4|hfc324ee_1|hfc324ee_3|hfc324ee_2|hfc324ee_0|h1635a5e_0|hb438215_0|hb438215_0|hb438215_1|h24e9cb9_0|h24e9cb9_0|h4646484_1|h14ed1c1_0|h14ed1c1_0|ha1047ec_0|h67e64d8_2']
harfbuzz -> glib[version='>=2.69.1,<3.0a0'] -> libglib[version='2.70.0|2.70.0|2.70.1|2.70.2|2.70.2|2.70.2|2.70.2|2.70.2|2.72.1|2.74.0|2.74.1|2.74.1|2.76.1|2.76.2|2.76.3|2.76.4|2.78.0|2.78.1|2.78.1|2.78.2|2.78.3|2.78.4|2.78.4|2.78.4|2.80.0',build='h67e64d8_0|h67e64d8_1|h67e64d8_0|h67e64d8_0|h67e64d8_1|h67e64d8_3|h67e64d8_4|h4646484_0|h24e9cb9_0|h24e9cb9_0|hd9b11f9_0|hfc324ee_3|hfc324ee_4|hfc324ee_1|hfc324ee_3|hfc324ee_2|hfc324ee_0|h1635a5e_0|hb438215_0|hb438215_0|hb438215_1|h24e9cb9_0|h24e9cb9_0|h4646484_1|h14ed1c1_0|h14ed1c1_0|ha1047ec_0|h67e64d8_2']
ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0'] -> libglib[version='>=2.78.1,<3.0a0']
cairo -> libglib[version='>=2.66.2,<3.0a0|>=2.66.4,<3.0a0|>=2.68.0,<3.0a0|>=2.70.0,<3.0a0|>=2.70.2,<3.0a0|>=2.72.1,<3.0a0|>=2.74.1,<3.0a0|>=2.76.2,<3.0a0|>=2.76.4,<3.0a0|>=2.78.0,<3.0a0']

Package llvmlite conflicts for:
librosa -> numba[version='>=0.51.0'] -> llvmlite[version='>=0.35.0,<0.36.0a0|>=0.36.0,<0.37.0a0|>=0.37.0,<0.38.0a0|>=0.38.0,<0.39.0a0|>=0.38.1,<0.39.0a0|>=0.39.1,<0.40.0a0|>=0.40.0,<0.41.0a0|>=0.41.1,<0.42.0a0|>=0.42.0,<0.43.0a0|>=0.41.0,<0.42.0a0|>=0.39.*,<0.40']
numba -> llvmlite[version='>=0.35.0,<0.36.0a0|>=0.36.0,<0.37.0a0|>=0.37.0,<0.38.0a0|>=0.38.0,<0.39.0a0|>=0.38.1,<0.39.0a0|>=0.39.1,<0.40.0a0|>=0.40.0,<0.41.0a0|>=0.41.1,<0.42.0a0|>=0.42.0,<0.43.0a0|>=0.41.0,<0.42.0a0|>=0.39.*,<0.40']

Package openjpeg conflicts for:
matplotlib-base -> pillow[version='>=8'] -> openjpeg[version='>=2.3.0,<3.0a0|>=2.4.0,<3.0.0a0|>=2.5.0,<2.6.0a0|>=2.5.0,<3.0a0|>=2.5.2,<3.0a0|>=2.4.0,<3.0a0']
pillow -> openjpeg[version='>=2.3.0,<3.0a0|>=2.4.0,<3.0.0a0|>=2.5.0,<2.6.0a0|>=2.5.0,<3.0a0|>=2.5.2,<3.0a0|>=2.4.0,<3.0a0']

Package cffi conflicts for:
urllib3 -> brotlipy[version='>=0.6.0'] -> cffi[version='!=1.11.3,>=1.8|>=1.0.0|>=1.12']
pysoundfile -> cffi
librosa -> pysoundfile[version='>=0.12.1'] -> cffi

Package numpy-base conflicts for:
matplotlib-base -> numpy[version='>=1.21,<2'] -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']
numpy -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']
numba -> numpy[version='>=1.23.5,<2.0a0'] -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']
scipy -> numpy[version='>=1.23.5,<1.28'] -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']
h5py -> numpy[version='>=1.22.4,<2.0a0'] -> numpy-base[version='1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4']
librosa -> numpy[version='>=1.20.3,!=1.22.0,!=1.22.1,!=1.22.2'] -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']
pysoundfile -> numpy -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']
soxr-python -> numpy[version='>=1.23.5,<2.0a0'] -> numpy-base[version='1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0']
scikit-learn -> numpy[version='>=1.23.5,<2.0a0'] -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']

Package flit-core conflicts for:
librosa -> typing_extensions[version='>=4.1.1'] -> flit-core[version='>=3.6,<4']
typing_extensions -> flit-core[version='>=3.6,<4']
importlib-metadata -> typing_extensions[version='>=3.6.4'] -> flit-core[version='>=3.6,<4']

Package jaraco.itertools conflicts for:
zipp -> jaraco.itertools
importlib-metadata -> zipp[version='>=0.5'] -> jaraco.itertools

Package brotli-bin conflicts for:
fonttools -> brotli -> brotli-bin[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_0|hb547adb_1|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']
brotli -> brotli-bin[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_0|hb547adb_1|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']
urllib3 -> brotli[version='>=1.0.9'] -> brotli-bin[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_0|hb547adb_1|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']

Package libxml2 conflicts for:
libflac -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
audioread -> ffmpeg -> libxml2[version='>=2.10.3,<3.0.0a0|>=2.10.4,<3.0.0a0|>=2.11.3,<3.0.0a0|>=2.11.4,<3.0.0a0|>=2.11.5,<3.0.0a0|>=2.11.6,<3.0.0a0|>=2.12.1,<3.0.0a0|>=2.12.2,<3.0.0a0|>=2.12.3,<3.0.0a0|>=2.12.4,<3.0a0|>=2.12.5,<3.0a0|>=2.12.6,<3.0a0|>=2.9.14,<3.0.0a0|>=2.9.13,<3.0.0a0|>=2.9.12,<3.0.0a0']
ffmpeg -> libxml2[version='>=2.10.3,<3.0.0a0|>=2.10.4,<3.0.0a0|>=2.11.3,<3.0.0a0|>=2.11.4,<3.0.0a0|>=2.11.5,<3.0.0a0|>=2.11.6,<3.0.0a0|>=2.12.1,<3.0.0a0|>=2.12.2,<3.0.0a0|>=2.12.3,<3.0.0a0|>=2.12.4,<3.0a0|>=2.12.5,<3.0a0|>=2.12.6,<3.0a0|>=2.9.14,<3.0.0a0|>=2.9.13,<3.0.0a0|>=2.9.12,<3.0.0a0']
libidn2 -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
ffmpeg -> fontconfig[version='>=2.14.1,<3.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.14,<2.10.0a0']
gettext -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
fontconfig -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<3.0.0a0|>=2.9.12,<3.0.0a0|>=2.9.14,<2.10.0a0|>=2.9.10,<2.10.0a0']
libglib -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
gnutls -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
cairo -> fontconfig[version='>=2.13.96,<3.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.12,<3.0.0a0|>=2.9.14,<2.10.0a0|>=2.9.10,<3.0.0a0|>=2.9.10,<2.10.0a0']

Package font-ttf-source-code-pro conflicts for:
fonts-conda-ecosystem -> fonts-conda-forge -> font-ttf-source-code-pro
fonts-conda-forge -> font-ttf-source-code-pro

Package libssh2 conflicts for:
libcurl -> libssh2[version='>=1.10.0|>=1.10.0,<2.0a0|>=1.11.0,<2.0a0|>=1.9.0,<2.0a0']
hdf5 -> libcurl[version='>=8.4.0,<9.0a0'] -> libssh2[version='>=1.10.0,<2.0a0|>=1.10.0|>=1.11.0,<2.0a0|>=1.9.0,<2.0a0']

Package kiwisolver conflicts for:
matplotlib-base -> kiwisolver[version='>=1.0.1|>=1.3.1']
librosa -> matplotlib-base[version='>=3.3.0'] -> kiwisolver[version='>=1.0.1|>=1.3.1']

Package fontconfig conflicts for:
libass -> fontconfig[version='>=2.14.2,<3.0a0']
harfbuzz -> cairo[version='>=1.18.0,<2.0a0'] -> fontconfig[version='>=2.13.1,<2.13.96.0a0|>=2.13.96,<3.0a0|>=2.14.2,<3.0a0|>=2.14.1,<3.0a0|>=2.13.1,<3.0a0']
cairo -> fontconfig[version='>=2.13.1,<2.13.96.0a0|>=2.13.96,<3.0a0|>=2.14.2,<3.0a0|>=2.14.1,<3.0a0|>=2.13.1,<3.0a0']
audioread -> ffmpeg -> fontconfig[version='>=2.13.96,<3.0a0|>=2.14.0,<3.0a0|>=2.14.1,<3.0a0|>=2.14.2,<3.0a0']
ffmpeg -> fontconfig[version='>=2.13.96,<3.0a0|>=2.14.0,<3.0a0|>=2.14.1,<3.0a0|>=2.14.2,<3.0a0']

Package glib-tools conflicts for:
cairo -> glib[version='>=2.69.1,<3.0a0'] -> glib-tools[version='2.70.0|2.70.0|2.70.1|2.70.2|2.70.2|2.70.2|2.70.2|2.70.2|2.72.1|2.74.0|2.74.1|2.74.1|2.76.1|2.76.2|2.76.3|2.76.4|2.78.0|2.78.1|2.78.1|2.78.2|2.78.3|2.78.4|2.78.4|2.78.4|2.78.4|2.80.0',build='hccf11d3_0|hccf11d3_2|hccf11d3_3|ha614eb4_0|ha614eb4_0|ha614eb4_0|h9e231a4_0|h9e231a4_1|h9e231a4_0|h9e231a4_0|h1059232_3|hb9a4d99_1|hb9a4d99_3|hb9a4d99_2|hb9a4d99_0|hb9a4d99_4|h1059232_4|h1059232_0|ha614eb4_0|hb5ab8b9_0|hb5ab8b9_1|hb5ab8b9_0|hb5ab8b9_0|h332123e_0|hccf11d3_4|hccf11d3_1|hccf11d3_0|hccf11d3_0|hccf11d3_1']
harfbuzz -> glib[version='>=2.69.1,<3.0a0'] -> glib-tools[version='2.70.0|2.70.0|2.70.1|2.70.2|2.70.2|2.70.2|2.70.2|2.70.2|2.72.1|2.74.0|2.74.1|2.74.1|2.76.1|2.76.2|2.76.3|2.76.4|2.78.0|2.78.1|2.78.1|2.78.2|2.78.3|2.78.4|2.78.4|2.78.4|2.78.4|2.80.0',build='hccf11d3_0|hccf11d3_2|hccf11d3_3|ha614eb4_0|ha614eb4_0|ha614eb4_0|h9e231a4_0|h9e231a4_1|h9e231a4_0|h9e231a4_0|h1059232_3|hb9a4d99_1|hb9a4d99_3|hb9a4d99_2|hb9a4d99_0|hb9a4d99_4|h1059232_4|h1059232_0|ha614eb4_0|hb5ab8b9_0|hb5ab8b9_1|hb5ab8b9_0|hb5ab8b9_0|h332123e_0|hccf11d3_4|hccf11d3_1|hccf11d3_0|hccf11d3_0|hccf11d3_1']

Package typing_extensions conflicts for:
lazy_loader -> importlib-metadata -> typing_extensions[version='>=3.6.4']
pooch -> platformdirs[version='>=2.5.0'] -> typing_extensions[version='>=4.7.1']
numba -> importlib-metadata -> typing_extensions[version='>=3.6.4']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> typing_extensions[version='3.7.4.*|>=3.6.6|>=3.6.6,<4.6.0|>=3.7.4,<3.8|>=3.7.4']
librosa -> typing_extensions[version='>=4.1.1']
platformdirs -> typing_extensions[version='>=4.7.1']
importlib-metadata -> typing_extensions[version='>=3.6.4']
platformdirs -> typing-extensions[version='>=4.6.3'] -> typing_extensions[version='4.10.0|4.11.0|4.9.0|4.8.0|4.7.1|4.7.0|4.6.3|4.7.1|4.7.1|4.7.1|4.7.1|4.7.1|4.6.3|4.6.3|4.6.3|4.6.3|4.6.2|4.6.1|4.6.0|4.5.0|4.5.0|4.5.0|4.5.0|4.5.0|4.4.0|4.4.0|4.4.0|4.4.0|4.4.0',build='py310hca03da5_0|py39hca03da5_0|pyha770c72_0|py310hca03da5_0|pyha770c72_0|pyha770c72_0|pyha770c72_0|pyha770c72_0|py310hca03da5_0|py311hca03da5_0|py311hca03da5_0|py310hca03da5_0|pyha770c72_0|py312hca03da5_0|py38hca03da5_0|py39hca03da5_0|py38hca03da5_0|py39hca03da5_0|py38hca03da5_0|py39hca03da5_0|py311hca03da5_0|py311hca03da5_0|py38hca03da5_0']

Package re2 conflicts for:
grpcio -> re2[version='>=2022.4.1,<2022.4.2.0a0|>=2022.6.1,<2022.6.2.0a0|>=2023.2.1,<2023.2.2.0a0']
grpcio -> libgrpc==1.62.1=h9c18a4f_0 -> re2[version='>=2023.2.2,<2023.2.3.0a0|>=2023.3.2,<2023.3.3.0a0']
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> re2[version='>=2022.4.1,<2022.4.2.0a0|>=2022.6.1,<2022.6.2.0a0|>=2023.2.1,<2023.2.2.0a0']

Package fonts-conda-forge conflicts for:
cairo -> fonts-conda-ecosystem -> fonts-conda-forge
libass -> fonts-conda-ecosystem -> fonts-conda-forge
ffmpeg -> fonts-conda-ecosystem -> fonts-conda-forge
fonts-conda-ecosystem -> fonts-conda-forge

Package threadpoolctl conflicts for:
scikit-learn -> threadpoolctl[version='>=2.0.0']
librosa -> scikit-learn[version='>=0.20.0'] -> threadpoolctl[version='>=2.0.0']

Package libgettextpo conflicts for:
libflac -> gettext[version='>=0.21.1,<1.0a0'] -> libgettextpo==0.22.5=h8fbad5d_2
gettext -> libgettextpo==0.22.5=h8fbad5d_2
libglib -> gettext[version='>=0.21.1,<1.0a0'] -> libgettextpo==0.22.5=h8fbad5d_2
libidn2 -> gettext[version='>=0.21.1,<1.0a0'] -> libgettextpo==0.22.5=h8fbad5d_2
libgettextpo-devel -> libgettextpo==0.22.5=h8fbad5d_2
gnutls -> gettext[version='>=0.21.1,<1.0a0'] -> libgettextpo==0.22.5=h8fbad5d_2

Package pillow conflicts for:
librosa -> matplotlib-base[version='>=3.3.0'] -> pillow[version='>=6.2.0|>=8']
matplotlib-base -> pillow[version='>=6.2.0|>=8']

Package svt-av1 conflicts for:
ffmpeg -> svt-av1[version='<1.0.0a0|>=1.1.0,<1.1.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0|>=1.3.0,<1.3.1.0a0|>=1.4.0,<1.4.1.0a0|>=1.4.1,<1.4.2.0a0|>=1.5.0,<1.5.1.0a0|>=1.6.0,<1.6.1.0a0|>=1.7.0,<1.7.1.0a0|>=1.8.0,<1.8.1.0a0|>=2.0.0,<2.0.1.0a0']
audioread -> ffmpeg -> svt-av1[version='<1.0.0a0|>=1.1.0,<1.1.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0|>=1.3.0,<1.3.1.0a0|>=1.4.0,<1.4.1.0a0|>=1.4.1,<1.4.2.0a0|>=1.5.0,<1.5.1.0a0|>=1.6.0,<1.6.1.0a0|>=1.7.0,<1.7.1.0a0|>=1.8.0,<1.8.1.0a0|>=2.0.0,<2.0.1.0a0']

Package liblapacke conflicts for:
blas-devel -> liblapacke==3.9.0[build='1_h9886b1c_netlib|0_h2ec9a88_netlib|5_h880f123_netlib|9_openblas|11_osxarm64_openblas|12_osxarm64_accelerate|13_osxarm64_accelerate|13_osxarm64_openblas|14_osxarm64_accelerate|14_osxarm64_openblas|15_osxarm64_accelerate|15_osxarm64_openblas|16_osxarm64_accelerate|18_osxarm64_accelerate|18_osxarm64_openblas|20_osxarm64_accelerate|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|21_osxarm64_openblas|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|17_osxarm64_accelerate|17_osxarm64_openblas|16_osxarm64_openblas|12_osxarm64_openblas|10_openblas|8_openblas|7_openblas']
numpy -> blas=[build=openblas] -> liblapacke==3.9.0[build='3_openblas|5_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_openblas|15_osxarm64_openblas|16_osxarm64_openblas|18_osxarm64_openblas|20_osxarm64_openblas|22_osxarm64_openblas|21_osxarm64_openblas|19_osxarm64_openblas|17_osxarm64_openblas|14_osxarm64_openblas|12_osxarm64_openblas|8_openblas|7_openblas|6_openblas|4_openblas|2_openblas|1_openblas']
blas-devel -> blas==2.106=openblas -> liblapacke==3.9.0[build='3_openblas|5_openblas|6_openblas|4_openblas|2_openblas|1_openblas']
blas -> liblapacke==3.9.0[build='0_h9886b1c_netlib|1_h9886b1c_netlib|1_h2ec9a88_netlib|3_openblas|3_he9612bc_netlib|5_h880f123_netlib|9_openblas|11_osxarm64_openblas|12_osxarm64_accelerate|13_osxarm64_accelerate|13_osxarm64_openblas|14_osxarm64_accelerate|14_osxarm64_openblas|15_osxarm64_accelerate|15_osxarm64_openblas|16_osxarm64_accelerate|18_osxarm64_accelerate|18_osxarm64_openblas|20_osxarm64_accelerate|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|21_osxarm64_openblas|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|17_osxarm64_accelerate|17_osxarm64_openblas|16_osxarm64_openblas|12_osxarm64_openblas|10_openblas|8_openblas|7_openblas|6_openblas|5_openblas|4_h880f123_netlib|4_openblas|2_openblas|2_h2ec9a88_netlib|1_openblas|0_h2ec9a88_netlib']
scipy -> blas=[build=openblas] -> liblapacke==3.9.0[build='3_openblas|5_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_openblas|15_osxarm64_openblas|16_osxarm64_openblas|18_osxarm64_openblas|20_osxarm64_openblas|22_osxarm64_openblas|21_osxarm64_openblas|19_osxarm64_openblas|17_osxarm64_openblas|14_osxarm64_openblas|12_osxarm64_openblas|8_openblas|7_openblas|6_openblas|4_openblas|2_openblas|1_openblas']

Package openmpi conflicts for:
h5py -> openmpi[version='>=4.0.5,<5.0.0a0|>=4.1.0,<5.0a0|>=4.1.1,<5.0a0|>=4.1.2,<5.0a0|>=4.1.4,<5.0a0|>=4.1.5,<5.0a0|>=4.1.6,<5.0a0']
h5py -> mpi4py[version='>=3.0'] -> openmpi[version='>=4.0,<5.0.0a0|>=4.1,<4.2.0a0|>=4.1.3,<5.0a0|>=4.1.4,<4.2.0a0']

Package libogg conflicts for:
libvorbis -> libogg[version='>=1.3.4,<1.4.0a0|>=1.3.5,<1.4.0a0|>=1.3.5,<2.0a0']
pysoundfile -> libsndfile[version='>=1.2'] -> libogg[version='>=1.3.4,<1.4.0a0']
libsndfile -> libflac[version='>=1.4.3,<1.5.0a0'] -> libogg[version='1.3.*|>=1.3.5,<1.4.0a0|>=1.3.5,<2.0a0']
libsndfile -> libogg[version='>=1.3.4,<1.4.0a0']
libflac -> libogg[version='1.3.*|>=1.3.4,<1.4.0a0']

Package libnghttp2 conflicts for:
libcurl -> libnghttp2[version='>=1.41.0,<2.0a0|>=1.43.0,<2.0a0|>=1.47.0,<2.0a0|>=1.51.0,<2.0a0|>=1.52.0,<2.0a0|>=1.58.0,<2.0a0|>=1.57.0|>=1.57.0,<2.0a0|>=1.52.0|>=1.46.0|>=1.46.0,<2.0a0']
hdf5 -> libcurl[version='>=8.4.0,<9.0a0'] -> libnghttp2[version='>=1.41.0,<2.0a0|>=1.43.0,<2.0a0|>=1.46.0,<2.0a0|>=1.46.0|>=1.47.0,<2.0a0|>=1.51.0,<2.0a0|>=1.52.0|>=1.52.0,<2.0a0|>=1.58.0,<2.0a0|>=1.57.0|>=1.57.0,<2.0a0']

Package gettext conflicts for:
gnutls -> gettext[version='>=0.19.8.1|>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0|>=0.21.0,<1.0a0']
libflac -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0']
harfbuzz -> libglib[version='>=2.78.1,<3.0a0'] -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0|>=0.21.0,<1.0a0']
libglib -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0']
cairo -> libglib[version='>=2.78.0,<3.0a0'] -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0|>=0.21.0,<1.0a0']
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> gettext[version='>=0.19.8.1|>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0|>=0.21.0,<1.0a0']
libidn2 -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0']
libsndfile -> libflac[version='>=1.4.3,<1.5.0a0'] -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0']

Package libvpx conflicts for:
audioread -> ffmpeg -> libvpx[version='>=1.10.0,<1.11.0a0|>=1.11.0,<1.12.0a0|>=1.13.0,<1.14.0a0|>=1.13.1,<1.14.0a0|>=1.14.0,<1.15.0a0']
ffmpeg -> libvpx[version='>=1.10.0,<1.11.0a0|>=1.11.0,<1.12.0a0|>=1.13.0,<1.14.0a0|>=1.13.1,<1.14.0a0|>=1.14.0,<1.15.0a0']

Package libasprintf conflicts for:
gnutls -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
libglib -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
libflac -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
gettext -> libasprintf==0.22.5=h8fbad5d_2
libidn2 -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
libasprintf-devel -> libasprintf==0.22.5=h8fbad5d_2

Package pypy3.8 conflicts for:
cffi -> python_abi==3.8[build=*_pypy38_pp73] -> pypy3.8=7.3
cffi -> pypy3.8[version='7.3.11.*|7.3.8.*']

Package matplotlib-base conflicts for:
librosa -> matplotlib[version='>=1.5.0'] -> matplotlib-base[version='>=3.3.2,<3.3.3.0a0|>=3.3.3,<3.3.4.0a0|>=3.3.4,<3.3.5.0a0|>=3.4.1,<3.4.2.0a0|>=3.4.2,<3.4.3.0a0|>=3.4.3,<3.4.4.0a0|>=3.5.0,<3.5.1.0a0|>=3.5.1,<3.5.2.0a0|>=3.5.2,<3.5.3.0a0|>=3.5.3,<3.5.4.0a0|>=3.6.0,<3.6.1.0a0|>=3.6.1,<3.6.2.0a0|>=3.6.2,<3.6.3.0a0|>=3.6.3,<3.6.4.0a0|>=3.7.0,<3.7.1.0a0|>=3.7.1,<3.7.2.0a0|>=3.7.2,<3.7.3.0a0|>=3.7.3,<3.7.4.0a0|>=3.8.0,<3.8.1.0a0|>=3.8.1,<3.8.2.0a0|>=3.8.2,<3.8.3.0a0|>=3.8.3,<3.8.4.0a0']
librosa -> matplotlib-base[version='>=1.5.0|>=3.3.0']

Package libvorbis conflicts for:
pysoundfile -> libsndfile[version='>=1.2'] -> libvorbis[version='>=1.3.7,<1.4.0a0']
libsndfile -> libvorbis[version='>=1.3.7,<1.4.0a0']

Package dav1d conflicts for:
audioread -> ffmpeg -> dav1d[version='>=1.0.0,<1.0.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0']
ffmpeg -> dav1d[version='>=1.0.0,<1.0.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0']

Package cairo conflicts for:
ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0'] -> cairo[version='>=1.18.0,<2.0a0']
libass -> harfbuzz[version='>=8.1.1,<9.0a0'] -> cairo[version='>=1.16.0,<2.0a0|>=1.18.0,<2.0a0']
harfbuzz -> cairo[version='>=1.16.0,<2.0.0a0|>=1.16.0,<2.0a0|>=1.18.0,<2.0a0']

Package libunistring conflicts for:
gnutls -> libunistring[version='>=0,<1.0a0']
ffmpeg -> gnutls[version='>=3.6.13,<3.7.0a0'] -> libunistring[version='>=0,<1.0a0']
libidn2 -> libunistring[version='>=0,<1.0a0']

Package fonttools conflicts for:
matplotlib-base -> fonttools[version='>=4.22.0']
librosa -> matplotlib-base[version='>=3.3.0'] -> fonttools[version='>=4.22.0']

Package libflac conflicts for:
libsndfile -> libflac[version='>=1.3.3,<1.4.0a0|>=1.4.1,<1.5.0a0|>=1.4.2,<1.5.0a0|>=1.4.3,<1.5.0a0']
pysoundfile -> libsndfile[version='>=1.2'] -> libflac[version='>=1.3.3,<1.4.0a0|>=1.4.1,<1.5.0a0|>=1.4.2,<1.5.0a0|>=1.4.3,<1.5.0a0']

Package nettle conflicts for:
gnutls -> nettle[version='>=3.4.1|>=3.7,<3.8.0a0|>=3.8,<3.9.0a0|>=3.8.1,<3.9.0a0|>=3.9.1,<3.10.0a0|>=3.6,<3.7.0a0|>=3.7.3,<3.8.0a0']
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> nettle[version='>=3.4.1|>=3.7,<3.8.0a0|>=3.8,<3.9.0a0|>=3.8.1,<3.9.0a0|>=3.9.1,<3.10.0a0|>=3.6,<3.7.0a0|>=3.7.3,<3.8.0a0']

Package libev conflicts for:
libnghttp2 -> libev[version='>=4.11|>=4.33,<4.34.0a0|>=4.33,<5.0a0']
libcurl -> libnghttp2[version='>=1.58.0,<2.0a0'] -> libev[version='>=4.11|>=4.33,<4.34.0a0|>=4.33,<5.0a0']

Package libopus conflicts for:
pysoundfile -> libsndfile[version='>=1.2'] -> libopus[version='>=1.3.1,<2.0a0']
audioread -> ffmpeg -> libopus[version='>=1.3,<2.0a0|>=1.3.1,<2.0a0']
ffmpeg -> libopus[version='>=1.3,<2.0a0|>=1.3.1,<2.0a0']
libsndfile -> libopus[version='>=1.3.1,<2.0a0']

Package appdirs conflicts for:
scipy -> pooch -> appdirs[version='>=1.3.0']
pooch -> appdirs[version='>=1.3.0']
librosa -> pooch[version='>=1.0'] -> appdirs[version='>=1.3.0']

Package python-dateutil conflicts for:
matplotlib-base -> python-dateutil[version='>=2.1|>=2.7']
librosa -> matplotlib-base[version='>=3.3.0'] -> python-dateutil[version='>=2.1|>=2.7']

Package gtest conflicts for:
grpcio -> abseil-cpp[version='>=20230802.0,<20230802.1.0a0'] -> gtest[version='>=1.14.0,<1.14.1.0a0']
libprotobuf -> gtest[version='>=1.14.0,<1.14.1.0a0']
protobuf -> libprotobuf[version='>=4.23.4,<4.23.5.0a0'] -> gtest[version='>=1.14.0,<1.14.1.0a0']

Package pypy3.7 conflicts for:
python_abi -> pypy3.7=7.3
pysoundfile -> cffi -> pypy3.7[version='7.3.3.*|7.3.4.*|7.3.5.*|7.3.7.*']
cffi -> pypy3.7[version='7.3.3.*|7.3.4.*|7.3.5.*|7.3.7.*']
cffi -> python_abi==3.7[build=*_pypy37_pp73] -> pypy3.7=7.3

Package platformdirs conflicts for:
pooch -> platformdirs[version='>=2.5.0']
scipy -> pooch -> platformdirs[version='>=2.5.0']
librosa -> pooch[version='>=1.0'] -> platformdirs[version='>=2.5.0']

Package fonts-anaconda conflicts for:
libass -> fonts-conda-ecosystem -> fonts-anaconda
cairo -> fonts-conda-ecosystem -> fonts-anaconda
ffmpeg -> fonts-conda-ecosystem -> fonts-anaconda
fonts-conda-ecosystem -> fonts-anaconda

Package chardet conflicts for:
requests -> chardet[version='>=3.0.2,<4|>=3.0.2,<5']
pooch -> requests[version='>=2.19.0'] -> chardet[version='>=3.0.2,<4|>=3.0.2,<5']

Package protobuf conflicts for:
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> protobuf[version='>=3.19.6|>=3.20.3,<5,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5|>=3.9.2|>=3.6.1|>=3.6.0|>=3.9.2,<3.20']
tensorflow-deps -> protobuf[version='>=3.19.1,<3.20']

Package pypy3.9 conflicts for:
cffi -> python_abi==3.9[build=*_pypy39_pp73] -> pypy3.9=7.3
cffi -> pypy3.9[version='7.3.11.*|7.3.15.*|7.3.8.*']

Package joblib conflicts for:
scikit-learn -> joblib[version='>=0.11|>=1.0.0|>=1.1.1|>=1.2.0']
librosa -> scikit-learn[version='>=0.20.0'] -> joblib[version='>=0.11|>=1.0.0|>=1.1.1|>=1.2.0']
librosa -> joblib[version='>=0.12.0|>=0.14.0|>=0.7.0']

Package mpg123 conflicts for:
pysoundfile -> libsndfile[version='>=1.2'] -> mpg123[version='>=1.30.2,<1.31.0a0|>=1.31.1,<1.32.0a0|>=1.31.3,<1.32.0a0|>=1.32.1,<1.33.0a0']
libsndfile -> mpg123[version='>=1.30.2,<1.31.0a0|>=1.31.1,<1.32.0a0|>=1.31.3,<1.32.0a0|>=1.32.1,<1.33.0a0']

Package c-ares conflicts for:
grpcio -> libgrpc==1.62.1=h9c18a4f_0 -> c-ares[version='>=1.19.0,<2.0a0|>=1.20.1,<2.0a0|>=1.21.0,<2.0a0|>=1.22.1,<2.0a0|>=1.25.0,<2.0a0|>=1.26.0,<2.0a0|>=1.27.0,<2.0a0']
grpcio -> c-ares[version='>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.19.1,<2.0a0']
libnghttp2 -> c-ares[version='>=1.16.1,<2.0a0|>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.20.1,<2.0a0|>=1.21.0,<2.0a0|>=1.23.0,<2.0a0|>=1.7.5|>=1.19.1,<2.0a0|>=1.19.0,<2.0a0']
libcurl -> libnghttp2[version='>=1.58.0,<2.0a0'] -> c-ares[version='>=1.16.1,<2.0a0|>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.20.1,<2.0a0|>=1.21.0,<2.0a0|>=1.23.0,<2.0a0|>=1.19.1,<2.0a0|>=1.7.5|>=1.19.0,<2.0a0']
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> c-ares[version='>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.19.1,<2.0a0']

Package libidn2 conflicts for:
gnutls -> libidn2[version='>=2,<3.0a0']
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> libidn2[version='>=2,<3.0a0']

Package harfbuzz conflicts for:
audioread -> ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0']
ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0']
libass -> harfbuzz[version='>=7.2.0,<8.0a0|>=8.1.1,<9.0a0']
ffmpeg -> libass[version='>=0.17.1,<0.17.2.0a0'] -> harfbuzz[version='>=7.2.0,<8.0a0|>=8.1.1,<9.0a0']

Package openblas conflicts for:
blas-devel -> openblas[version='0.3.18.*|0.3.20.*|0.3.21.*|0.3.23.*|0.3.24.*|0.3.25.*|0.3.26.*|0.3.27.*']
blas -> blas-devel==3.9.0=22_osxarm64_openblas -> openblas[version='0.3.18.*|0.3.20.*|0.3.21.*|0.3.23.*|0.3.24.*|0.3.25.*|0.3.26.*|0.3.27.*']

Package pcre conflicts for:
libglib -> pcre[version='>=8.44,<9.0a0|>=8.45,<9.0a0']
harfbuzz -> libglib[version='>=2.72.1,<3.0a0'] -> pcre[version='>=8.44,<9.0a0|>=8.45,<9.0a0']
cairo -> libglib[version='>=2.72.1,<3.0a0'] -> pcre[version='>=8.44,<9.0a0|>=8.45,<9.0a0']

Package libcurl conflicts for:
h5py -> hdf5[version='>=1.14.3,<1.14.4.0a0'] -> libcurl[version='>=7.71.1,<8.0a0|>=7.77.0,<9.0a0|>=7.79.1,<9.0a0|>=7.80.0,<9.0a0|>=7.81.0,<9.0a0|>=7.83.1,<9.0a0|>=7.87.0,<9.0a0|>=8.1.2,<9.0a0|>=8.2.1,<9.0a0|>=8.4.0,<9.0a0|>=7.88.1,<9.0a0|>=7.88.1,<8.0a0|>=7.82.0,<8.0a0|>=7.71.1,<9.0a0']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> libcurl[version='>=7.76.0,<9.0a0|>=7.76.1,<9.0a0|>=7.78.0,<9.0a0|>=7.79.1,<9.0a0|>=7.80.0,<9.0a0|>=7.81.0,<9.0a0|>=7.83.0,<9.0a0|>=7.83.1,<9.0a0|>=7.87.0,<9.0a0|>=7.88.1,<9.0a0|>=8.1.2,<9.0a0|>=8.3.0,<9.0a0|>=8.4.0,<9.0a0|>=8.5.0,<9.0a0|>=7.88.1,<8.0a0|>=7.86.0,<8.0a0']
hdf5 -> libcurl[version='>=7.71.1,<8.0a0|>=7.71.1,<9.0a0|>=7.76.0,<9.0a0|>=7.77.0,<9.0a0|>=7.79.1,<9.0a0|>=7.80.0,<9.0a0|>=7.81.0,<9.0a0|>=7.83.1,<9.0a0|>=7.87.0,<9.0a0|>=8.1.2,<9.0a0|>=8.2.1,<9.0a0|>=8.4.0,<9.0a0|>=7.88.1,<9.0a0|>=7.88.1,<8.0a0|>=7.82.0,<8.0a0']

Package xorg-libxdmcp conflicts for:
pillow -> libxcb[version='>=1.15,<1.16.0a0'] -> xorg-libxdmcp
libxcb -> xorg-libxdmcp

Package tbb conflicts for:
librosa -> numba[version='>=0.51.0'] -> tbb[version='>=2021.3.0|>=2021.5.0|>=2021.8.0']
numba -> tbb[version='>=2021.3.0|>=2021.5.0|>=2021.8.0']
ffmpeg -> libopenvino[version='>=2024.0.0,<2024.0.1.0a0'] -> tbb[version='>=2021.11.0|>=2021.5.0']

Package pooch conflicts for:
librosa -> pooch[version='>=1.0|>=1.0,<1.7']
scipy -> pooch
scikit-learn -> scipy -> pooch
librosa -> scipy[version='>=1.2.0'] -> pooch

Package p11-kit conflicts for:
gnutls -> p11-kit[version='>=0.23.21,<0.24.0a0|>=0.24.1,<0.25.0a0']
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> p11-kit[version='>=0.23.21,<0.24.0a0|>=0.24.1,<0.25.0a0']

Package lerc conflicts for:
lcms2 -> libtiff[version='>=4.6.0,<4.7.0a0'] -> lerc[version='>=2.2.1,<3.0a0|>=3.0,<4.0a0|>=4.0.0,<5.0a0']
openjpeg -> libtiff[version='>=4.6.0,<4.7.0a0'] -> lerc[version='>=2.2.1,<3.0a0|>=3.0,<4.0a0|>=4.0.0,<5.0a0']
libtiff -> lerc[version='>=2.2.1,<3.0a0|>=3.0,<4.0a0|>=4.0.0,<5.0a0']
pillow -> libtiff[version='>=4.6.0,<4.7.0a0'] -> lerc[version='>=2.2.1,<3.0a0|>=3.0,<4.0a0|>=4.0.0,<5.0a0']

Package libllvm14 conflicts for:
numba -> libllvm14[version='>=14.0.6,<14.1.0a0']
llvmlite -> libllvm14[version='>=14.0.6,<14.1.0a0']
librosa -> numba[version='>=0.51.0'] -> libllvm14[version='>=14.0.6,<14.1.0a0']

Package mpi4py conflicts for:
tensorflow-deps -> h5py[version='>=3.6.0,<3.7'] -> mpi4py[version='>=3.0']
h5py -> mpi4py[version='>=3.0']

Package x265 conflicts for:
ffmpeg -> x265[version='>=3.5,<3.6.0a0']
audioread -> ffmpeg -> x265[version='>=3.5,<3.6.0a0']

Package font-ttf-dejavu-sans-mono conflicts for:
fonts-conda-forge -> font-ttf-dejavu-sans-mono
fonts-conda-ecosystem -> fonts-conda-forge -> font-ttf-dejavu-sans-mono

Package pthread-stubs conflicts for:
libxcb -> pthread-stubs
pillow -> libxcb[version='>=1.15,<1.16.0a0'] -> pthread-stubs

Package libdeflate conflicts for:
lcms2 -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']
openjpeg -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']
pillow -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']
libtiff -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']

Package font-ttf-ubuntu conflicts for:
fonts-conda-forge -> font-ttf-ubuntu
fonts-conda-ecosystem -> fonts-conda-forge -> font-ttf-ubuntu

Package pypy3.6 conflicts for:
cffi -> python_abi==3.6[build=*_pypy36_pp73] -> pypy3.6=7.3
cffi -> pypy3.6[version='7.3.0.*|7.3.1.*|7.3.2.*|7.3.3.*']

Package importlib-metadata conflicts for:
lazy_loader -> importlib-metadata
numba -> importlib_metadata -> importlib-metadata[version='>=1.1.3,<1.1.4.0a0|>=1.5.0,<1.5.1.0a0|>=1.5.2,<1.5.3.0a0|>=1.6.0,<1.6.1.0a0|>=1.6.1,<1.6.2.0a0|>=1.7.0,<1.7.1.0a0|>=2.0.0,<2.0.1.0a0|>=3.0.0,<3.0.1.0a0|>=3.1.0,<3.1.1.0a0|>=3.1.1,<3.1.2.0a0|>=3.10.0,<3.10.1.0a0|>=3.10.1,<3.10.2.0a0|>=4.0.1,<4.0.2.0a0|>=4.10.0,<4.10.1.0a0|>=4.10.1,<4.10.2.0a0|>=4.11.0,<4.11.1.0a0|>=4.11.1,<4.11.2.0a0|>=4.11.2,<4.11.3.0a0|>=4.11.3,<4.11.4.0a0|>=4.11.4,<4.11.5.0a0|>=4.13.0,<4.13.1.0a0|>=5.0.0,<5.0.1.0a0|>=5.1.0,<5.1.1.0a0|>=5.2.0,<5.2.1.0a0|>=6.0.0,<6.0.1.0a0|>=6.1.0,<6.1.1.0a0|>=6.10.0,<6.10.1.0a0|>=7.0.0,<7.0.1.0a0|>=7.0.1,<7.0.2.0a0|>=7.0.2,<7.0.3.0a0|>=7.1.0,<7.1.1.0a0|>=6.9.0,<6.9.1.0a0|>=6.8.0,<6.8.1.0a0|>=6.7.0,<6.7.1.0a0|>=6.6.0,<6.6.1.0a0|>=6.5.1,<6.5.2.0a0|>=6.5.0,<6.5.1.0a0|>=6.4.1,<6.4.2.0a0|>=6.4.0,<6.4.1.0a0|>=6.3.0,<6.3.1.0a0|>=6.2.1,<6.2.2.0a0|>=6.2.0,<6.2.1.0a0|>=4.9.0,<4.9.1.0a0|>=4.8.3,<4.8.4.0a0|>=4.8.2,<4.8.3.0a0|>=4.8.1,<4.8.2.0a0|>=4.8.0,<4.8.1.0a0|>=4.7.1,<4.7.2.0a0|>=4.7.0,<4.7.1.0a0|>=4.6.4,<4.6.5.0a0|>=4.6.3,<4.6.4.0a0|>=4.6.2,<4.6.3.0a0|>=4.6.1,<4.6.2.0a0|>=4.6.0,<4.6.1.0a0|>=4.5.0,<4.5.1.0a0|>=4.4.0,<4.4.1.0a0|>=4.3.1,<4.3.2.0a0|>=4.3.0,<4.3.1.0a0|>=4.2.0,<4.2.1.0a0|>=3.9.1,<3.9.2.0a0|>=3.9.0,<3.9.1.0a0|>=3.8.1,<3.8.2.0a0|>=3.8.0,<3.8.1.0a0|>=3.7.3,<3.7.4.0a0|>=3.7.2,<3.7.3.0a0|>=3.7.0,<3.7.1.0a0|>=3.6.0,<3.6.1.0a0|>=3.4.0,<3.4.1.0a0|>=3.3.0,<3.3.1.0a0']
librosa -> lazy_loader[version='>=0.1'] -> importlib-metadata
numba -> importlib-metadata

Package libedit conflicts for:
krb5 -> libedit[version='>=3.1.20191231,<3.2.0a0|>=3.1.20191231,<4.0a0|>=3.1.20221030,<3.2.0a0|>=3.1.20221030,<4.0a0|>=3.1.20210216,<3.2.0a0|>=3.1.20210216,<4.0a0']
libcurl -> krb5[version='>=1.21.2,<1.22.0a0'] -> libedit[version='>=3.1.20191231,<3.2.0a0|>=3.1.20191231,<4.0a0|>=3.1.20221030,<3.2.0a0|>=3.1.20221030,<4.0a0|>=3.1.20210216,<3.2.0a0|>=3.1.20210216,<4.0a0']

Package libass conflicts for:
audioread -> ffmpeg -> libass[version='>=0.17.1,<0.17.2.0a0']
ffmpeg -> libass[version='>=0.17.1,<0.17.2.0a0']

Package libwebp conflicts for:
matplotlib-base -> pillow[version='>=8'] -> libwebp[version='>=0.3.0|>=1.2.0,<1.3.0a0|>=1.3.2,<2.0a0']
pillow -> libwebp[version='>=0.3.0|>=1.2.0,<1.3.0a0|>=1.3.2,<2.0a0']

Package pyopenssl conflicts for:
urllib3 -> pyopenssl[version='>=0.14']
requests -> urllib3[version='>=1.21.1,<3'] -> pyopenssl[version='>=0.14']

Package pixman conflicts for:
cairo -> pixman[version='>=0.40.0,<1.0a0|>=0.42.2,<1.0a0']
harfbuzz -> cairo[version='>=1.18.0,<2.0a0'] -> pixman[version='>=0.40.0,<1.0a0|>=0.42.2,<1.0a0']The following specifications were found to be incompatible with your system:

  - feature:/osx-arm64::__osx==13.6.3=0
  - feature:/osx-arm64::__unix==0=0
  - feature:|@/osx-arm64::__osx==13.6.3=0
  - feature:|@/osx-arm64::__unix==0=0
  - aom -> __osx[version='>=10.9']
  - audioread -> ffmpeg -> __osx[version='>=10.9']
  - cairo -> __osx[version='>=10.9']
  - ffmpeg -> __osx[version='>=10.9']
  - gettext -> ncurses[version='>=6.4,<7.0a0'] -> __osx[version='>=10.9']
  - gmp -> __osx[version='>=10.9']
  - gnutls -> __osx[version='>=10.9']
  - grpcio -> __osx[version='>=10.9']
  - h5py -> hdf5[version='>=1.14.3,<1.14.4.0a0'] -> __osx[version='>=10.9']
  - harfbuzz -> __osx[version='>=10.9']
  - hdf5 -> __osx[version='>=10.9']
  - libass -> harfbuzz[version='>=8.1.1,<9.0a0'] -> __osx[version='>=10.9']
  - libcurl -> libnghttp2[version='>=1.58.0,<2.0a0'] -> __osx[version='>=10.9']
  - libedit -> ncurses[version='>=6.2,<7.0.0a0'] -> __osx[version='>=10.9']
  - libglib -> __osx[version='>=10.9']
  - libnghttp2 -> __osx[version='>=10.9']
  - libprotobuf -> __osx[version='>=10.9']
  - librosa -> matplotlib-base[version='>=3.3.0'] -> __osx[version='>=10.9']
  - matplotlib-base -> __osx[version='>=10.9']
  - msgpack-python -> __osx[version='>=10.9']
  - ncurses -> __osx[version='>=10.9']
  - nettle -> gmp[version='>=6.2.1,<7.0a0'] -> __osx[version='>=10.9']
  - numba -> __osx[version='>=10.9']
  - numpy -> __osx[version='>=10.9']
  - openh264 -> __osx[version='>=10.9']
  - protobuf -> __osx[version='>=10.9']
  - pysocks -> __unix
  - pysocks -> __win
  - pysoundfile -> numpy -> __osx[version='>=10.9']
  - python=3.8 -> ncurses[version='>=6.4,<7.0a0'] -> __osx[version='>=10.9']
  - readline -> ncurses[version='>=6.3,<7.0a0'] -> __osx[version='>=10.9']
  - scikit-learn -> __osx[version='>=10.9']
  - scipy -> __osx[version='>=10.9']
  - soxr-python -> numpy[version='>=1.23.5,<2.0a0'] -> __osx[version='>=10.9']
  - svt-av1 -> __osx[version='>=10.9']
  - tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> __osx[version='>=10.9']
  - urllib3 -> pysocks[version='>=1.5.6,<2.0,!=1.5.7'] -> __unix

Your installed version is: 13.6.3


(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python -c "import tensorflow as tf; print(tf.__version__)"

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'tensorflow'
(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)


*|3.15.7.*|3.15.8.*|3.16.0.*|3.17.0.*|3.17.1.*|3.17.2.*|3.18.0.*|3.18.1.*|3.18.3.*|3.19.1.*|3.19.2.*|3.19.3.*|3.19.4.*|3.19.6.*|3.20.0.*|3.20.1.*|3.20.2.*|3.20.3.*|3.21.1.*|3.21.10.*|3.21.11.*|3.21.12.*|>=4.22.5,<4.23.0a0|>=4.23.1,<4.24.0a0|>=4.23.2,<4.23.3.0a0|>=4.23.3,<4.23.4.0a0|>=4.23.4,<4.23.5.0a0|>=4.24.3,<4.24.4.0a0|>=4.24.4,<4.24.5.0a0|>=4.25.1,<4.25.2.0a0|>=4.25.2,<4.25.3.0a0|>=4.25.3,<4.25.4.0a0|>=3.21.12,<3.22.0a0|>=3.21.11,<3.22.0a0|>=3.21.10,<3.22.0a0|3.21.9.*|>=3.21.9,<3.22.0a0|3.21.8.*|>=3.21.8,<3.22.0a0|3.21.7.*|>=3.21.7,<3.22.0a0|3.21.6.*|>=3.21.6,<3.22.0a0|3.21.5.*|>=3.21.5,<3.22.0a0|3.21.4.*|>=3.21.4,<3.22.0a0|3.21.3.*|>=3.21.3,<3.22.0a0|3.21.2.*|>=3.21.2,<3.22.0a0|>=3.21.1,<3.22.0a0|>=3.20.3,<3.21.0a0|>=3.20.2,<3.21.0a0|>=3.20.1,<3.21.0a0|>=3.20.0,<3.21.0a0|>=3.19.6,<3.20.0a0|>=3.19.4,<3.20.0a0|>=3.19.3,<3.20.0a0|>=3.19.2,<3.20.0a0|>=3.19.1,<3.20.0a0|>=3.18.3,<3.19.0a0|>=3.18.1,<3.19.0a0|>=3.18.0,<3.19.0a0|>=3.17.2,<3.18.0a0|>=3.17.1,<3.18.0a0|>=3.17.0,<3.18.0a0|>=3.16.0,<3.17.0a0|>=3.15.8,<3.16.0a0|>=3.15.7,<3.16.0a0|>=3.15.6,<3.16.0a0|>=3.15.5,<3.16.0a0|>=3.15.4,<3.16.0a0|>=3.15.3,<3.16.0a0|>=3.15.2,<3.16.0a0|>=3.15.1,<3.16.0a0|>=3.15.0,<3.16.0a0|>=3.14.0,<3.15.0a0|>=3.13.0.1,<3.14.0a0']
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> libprotobuf[version='3.19.1.*|3.19.2.*|3.19.3.*|3.19.4.*|3.19.6.*|>=3.20.1,<3.21.0a0|>=3.21.10,<3.22.0a0|>=3.21.12,<3.22.0a0|>=3.21.9,<3.22.0a0|>=3.21.8,<3.22.0a0|>=3.21.5,<3.22.0a0|>=3.20.3,<3.21.0a0|>=3.19.6,<3.20.0a0|>=3.19.4,<3.20.0a0|>=3.19.3,<3.20.0a0|>=3.19.2,<3.20.0a0|>=3.19.1,<3.20.0a0']
grpcio -> libgrpc==1.62.1=h9c18a4f_0 -> libprotobuf[version='>=3.20.3,<3.20.4.0a0|>=4.23.1,<4.24.0a0|>=4.23.2,<4.23.3.0a0|>=4.23.3,<4.23.4.0a0|>=4.23.4,<4.23.5.0a0|>=4.24.3,<4.24.4.0a0|>=4.24.4,<4.24.5.0a0|>=4.25.1,<4.25.2.0a0|>=4.25.2,<4.25.3.0a0|>=4.25.3,<4.25.4.0a0']
ffmpeg -> libopenvino-onnx-frontend[version='>=2024.0.0,<2024.0.1.0a0'] -> libprotobuf[version='>=4.24.3,<4.24.4.0a0|>=4.24.4,<4.24.5.0a0|>=4.25.1,<4.25.2.0a0|>=4.25.2,<4.25.3.0a0|>=4.25.3,<4.25.4.0a0']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> libprotobuf[version='>=3.15.7,<3.16.0a0|>=3.15.8,<3.16.0a0|>=3.16.0,<3.17.0a0|>=3.18.1,<3.19.0a0|>=3.19.4,<3.20.0a0|>=3.20.1,<3.21.0a0|>=3.21.12,<3.22.0a0|>=4.24.4,<4.24.5.0a0|>=3.21.6,<3.22.0a0|>=3.20.3,<3.21.0a0']

Package six conflicts for:
lazy_loader -> packaging -> six
packaging -> six
librosa -> six[version='>=1.3']
python-dateutil -> six[version='>=1.5']
grpcio -> six[version='>=1.5.2|>=1.6.0']
cycler -> six
pooch -> packaging[version='>=20.0'] -> six
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> six[version='1.15.0.*|>=1.10.0|>=1.12|>=1.15,<1.16|>=1.15.0']
urllib3 -> cryptography[version='>=1.3.4'] -> six[version='>=1.4.1|>=1.5.2']
protobuf -> six
wheel -> packaging[version='>=20.2'] -> six
h5py -> six
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> six[version='>=1.5.2|>=1.6.0']
librosa -> packaging[version='>=20.0'] -> six
zipp -> more-itertools -> six[version='>=1.0.0,<2.0.0']
matplotlib-base -> cycler[version='>=0.10'] -> six[version='>=1.5']

Package libtiff conflicts for:
matplotlib-base -> pillow[version='>=8'] -> libtiff[version='>=4.1.0,<4.4.0a0|>=4.2.0,<4.4.0a0|>=4.3.0,<4.4.0a0|>=4.3.0,<4.5.0a0|>=4.4.0,<4.5.0a0|>=4.5.0,<4.6.0a0|>=4.5.1,<4.6.0a0|>=4.6.0,<4.7.0a0|>=4.2.0,<5.0a0']
pillow -> libtiff[version='>=4.1.0,<4.4.0a0|>=4.2.0,<4.4.0a0|>=4.3.0,<4.4.0a0|>=4.3.0,<4.5.0a0|>=4.4.0,<4.5.0a0|>=4.5.0,<4.6.0a0|>=4.5.1,<4.6.0a0|>=4.6.0,<4.7.0a0|>=4.2.0,<5.0a0']
openjpeg -> libtiff[version='>=4.1.0,<4.5.0a0|>=4.2.0,<4.5.0a0|>=4.3.0,<4.5.0a0|>=4.4.0,<4.5.0a0|>=4.5.0,<4.6.0a0|>=4.6.0,<4.7.0a0|>=4.2.0,<5.0a0']
pillow -> lcms2[version='>=2.12,<3.0a0'] -> libtiff[version='>=4.1.0,<4.5.0a0|>=4.2.0,<4.5.0a0']
lcms2 -> libtiff[version='>=4.1.0,<4.5.0a0|>=4.2.0,<4.5.0a0|>=4.4.0,<4.5.0a0|>=4.5.0,<4.6.0a0|>=4.6.0,<4.7.0a0|>=4.2.0,<5.0a0']

Package libjpeg-turbo conflicts for:
matplotlib-base -> pillow[version='>=8'] -> libjpeg-turbo[version='>=2.1.4,<3.0a0|>=2.1.5.1,<3.0a0|>=3.0.0,<4.0a0']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> libjpeg-turbo[version='>=2.1.5.1,<3.0a0|>=3.0.0,<4.0a0']
lcms2 -> libjpeg-turbo[version='>=2.1.5.1,<3.0a0|>=3.0.0,<4.0a0']
pillow -> libjpeg-turbo[version='>=2.1.4,<3.0a0|>=2.1.5.1,<3.0a0|>=3.0.0,<4.0a0']
lcms2 -> libtiff[version='>=4.5.0,<4.6.0a0'] -> libjpeg-turbo[version='>=2.1.4,<3.0a0']
openjpeg -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libjpeg-turbo[version='>=2.1.4,<3.0a0|>=2.1.5.1,<3.0a0|>=3.0.0,<4.0a0']
libtiff -> libjpeg-turbo[version='>=2.1.4,<3.0a0|>=2.1.5.1,<3.0a0|>=3.0.0,<4.0a0']

Package libglib conflicts for:
harfbuzz -> libglib[version='>=2.66.2,<3.0a0|>=2.66.4,<3.0a0|>=2.66.7,<3.0a0|>=2.68.0,<3.0a0|>=2.68.1,<3.0a0|>=2.68.3,<3.0a0|>=2.68.4,<3.0a0|>=2.70.0,<3.0a0|>=2.70.1,<3.0a0|>=2.70.2,<3.0a0|>=2.72.1,<3.0a0|>=2.74.0,<3.0a0|>=2.74.1,<3.0a0|>=2.76.2,<3.0a0|>=2.76.4,<3.0a0|>=2.78.0,<3.0a0|>=2.78.1,<3.0a0']
libass -> harfbuzz[version='>=8.1.1,<9.0a0'] -> libglib[version='>=2.76.2,<3.0a0|>=2.76.4,<3.0a0|>=2.78.0,<3.0a0|>=2.78.1,<3.0a0']
cairo -> glib[version='>=2.69.1,<3.0a0'] -> libglib[version='2.70.0|2.70.0|2.70.1|2.70.2|2.70.2|2.70.2|2.70.2|2.70.2|2.72.1|2.74.0|2.74.1|2.74.1|2.76.1|2.76.2|2.76.3|2.76.4|2.78.0|2.78.1|2.78.1|2.78.2|2.78.3|2.78.4|2.78.4|2.78.4|2.80.0',build='h67e64d8_0|h67e64d8_1|h67e64d8_0|h67e64d8_0|h67e64d8_1|h67e64d8_3|h67e64d8_4|h4646484_0|h24e9cb9_0|h24e9cb9_0|hd9b11f9_0|hfc324ee_3|hfc324ee_4|hfc324ee_1|hfc324ee_3|hfc324ee_2|hfc324ee_0|h1635a5e_0|hb438215_0|hb438215_0|hb438215_1|h24e9cb9_0|h24e9cb9_0|h4646484_1|h14ed1c1_0|h14ed1c1_0|ha1047ec_0|h67e64d8_2']
harfbuzz -> glib[version='>=2.69.1,<3.0a0'] -> libglib[version='2.70.0|2.70.0|2.70.1|2.70.2|2.70.2|2.70.2|2.70.2|2.70.2|2.72.1|2.74.0|2.74.1|2.74.1|2.76.1|2.76.2|2.76.3|2.76.4|2.78.0|2.78.1|2.78.1|2.78.2|2.78.3|2.78.4|2.78.4|2.78.4|2.80.0',build='h67e64d8_0|h67e64d8_1|h67e64d8_0|h67e64d8_0|h67e64d8_1|h67e64d8_3|h67e64d8_4|h4646484_0|h24e9cb9_0|h24e9cb9_0|hd9b11f9_0|hfc324ee_3|hfc324ee_4|hfc324ee_1|hfc324ee_3|hfc324ee_2|hfc324ee_0|h1635a5e_0|hb438215_0|hb438215_0|hb438215_1|h24e9cb9_0|h24e9cb9_0|h4646484_1|h14ed1c1_0|h14ed1c1_0|ha1047ec_0|h67e64d8_2']
ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0'] -> libglib[version='>=2.78.1,<3.0a0']
cairo -> libglib[version='>=2.66.2,<3.0a0|>=2.66.4,<3.0a0|>=2.68.0,<3.0a0|>=2.70.0,<3.0a0|>=2.70.2,<3.0a0|>=2.72.1,<3.0a0|>=2.74.1,<3.0a0|>=2.76.2,<3.0a0|>=2.76.4,<3.0a0|>=2.78.0,<3.0a0']

Package llvmlite conflicts for:
librosa -> numba[version='>=0.51.0'] -> llvmlite[version='>=0.35.0,<0.36.0a0|>=0.36.0,<0.37.0a0|>=0.37.0,<0.38.0a0|>=0.38.0,<0.39.0a0|>=0.38.1,<0.39.0a0|>=0.39.1,<0.40.0a0|>=0.40.0,<0.41.0a0|>=0.41.1,<0.42.0a0|>=0.42.0,<0.43.0a0|>=0.41.0,<0.42.0a0|>=0.39.*,<0.40']
numba -> llvmlite[version='>=0.35.0,<0.36.0a0|>=0.36.0,<0.37.0a0|>=0.37.0,<0.38.0a0|>=0.38.0,<0.39.0a0|>=0.38.1,<0.39.0a0|>=0.39.1,<0.40.0a0|>=0.40.0,<0.41.0a0|>=0.41.1,<0.42.0a0|>=0.42.0,<0.43.0a0|>=0.41.0,<0.42.0a0|>=0.39.*,<0.40']

Package openjpeg conflicts for:
matplotlib-base -> pillow[version='>=8'] -> openjpeg[version='>=2.3.0,<3.0a0|>=2.4.0,<3.0.0a0|>=2.5.0,<2.6.0a0|>=2.5.0,<3.0a0|>=2.5.2,<3.0a0|>=2.4.0,<3.0a0']
pillow -> openjpeg[version='>=2.3.0,<3.0a0|>=2.4.0,<3.0.0a0|>=2.5.0,<2.6.0a0|>=2.5.0,<3.0a0|>=2.5.2,<3.0a0|>=2.4.0,<3.0a0']

Package cffi conflicts for:
urllib3 -> brotlipy[version='>=0.6.0'] -> cffi[version='!=1.11.3,>=1.8|>=1.0.0|>=1.12']
pysoundfile -> cffi
librosa -> pysoundfile[version='>=0.12.1'] -> cffi

Package numpy-base conflicts for:
matplotlib-base -> numpy[version='>=1.21,<2'] -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']
numpy -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']
numba -> numpy[version='>=1.23.5,<2.0a0'] -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']
scipy -> numpy[version='>=1.23.5,<1.28'] -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']
h5py -> numpy[version='>=1.22.4,<2.0a0'] -> numpy-base[version='1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4']
librosa -> numpy[version='>=1.20.3,!=1.22.0,!=1.22.1,!=1.22.2'] -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']
pysoundfile -> numpy -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']
soxr-python -> numpy[version='>=1.23.5,<2.0a0'] -> numpy-base[version='1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0']
scikit-learn -> numpy[version='>=1.23.5,<2.0a0'] -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']

Package flit-core conflicts for:
librosa -> typing_extensions[version='>=4.1.1'] -> flit-core[version='>=3.6,<4']
typing_extensions -> flit-core[version='>=3.6,<4']
importlib-metadata -> typing_extensions[version='>=3.6.4'] -> flit-core[version='>=3.6,<4']

Package jaraco.itertools conflicts for:
zipp -> jaraco.itertools
importlib-metadata -> zipp[version='>=0.5'] -> jaraco.itertools

Package brotli-bin conflicts for:
fonttools -> brotli -> brotli-bin[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_0|hb547adb_1|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']
brotli -> brotli-bin[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_0|hb547adb_1|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']
urllib3 -> brotli[version='>=1.0.9'] -> brotli-bin[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_0|hb547adb_1|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']

Package libxml2 conflicts for:
libflac -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
audioread -> ffmpeg -> libxml2[version='>=2.10.3,<3.0.0a0|>=2.10.4,<3.0.0a0|>=2.11.3,<3.0.0a0|>=2.11.4,<3.0.0a0|>=2.11.5,<3.0.0a0|>=2.11.6,<3.0.0a0|>=2.12.1,<3.0.0a0|>=2.12.2,<3.0.0a0|>=2.12.3,<3.0.0a0|>=2.12.4,<3.0a0|>=2.12.5,<3.0a0|>=2.12.6,<3.0a0|>=2.9.14,<3.0.0a0|>=2.9.13,<3.0.0a0|>=2.9.12,<3.0.0a0']
ffmpeg -> libxml2[version='>=2.10.3,<3.0.0a0|>=2.10.4,<3.0.0a0|>=2.11.3,<3.0.0a0|>=2.11.4,<3.0.0a0|>=2.11.5,<3.0.0a0|>=2.11.6,<3.0.0a0|>=2.12.1,<3.0.0a0|>=2.12.2,<3.0.0a0|>=2.12.3,<3.0.0a0|>=2.12.4,<3.0a0|>=2.12.5,<3.0a0|>=2.12.6,<3.0a0|>=2.9.14,<3.0.0a0|>=2.9.13,<3.0.0a0|>=2.9.12,<3.0.0a0']
libidn2 -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
ffmpeg -> fontconfig[version='>=2.14.1,<3.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.14,<2.10.0a0']
gettext -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
fontconfig -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<3.0.0a0|>=2.9.12,<3.0.0a0|>=2.9.14,<2.10.0a0|>=2.9.10,<2.10.0a0']
libglib -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
gnutls -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
cairo -> fontconfig[version='>=2.13.96,<3.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.12,<3.0.0a0|>=2.9.14,<2.10.0a0|>=2.9.10,<3.0.0a0|>=2.9.10,<2.10.0a0']

Package font-ttf-source-code-pro conflicts for:
fonts-conda-ecosystem -> fonts-conda-forge -> font-ttf-source-code-pro
fonts-conda-forge -> font-ttf-source-code-pro

Package libssh2 conflicts for:
libcurl -> libssh2[version='>=1.10.0|>=1.10.0,<2.0a0|>=1.11.0,<2.0a0|>=1.9.0,<2.0a0']
hdf5 -> libcurl[version='>=8.4.0,<9.0a0'] -> libssh2[version='>=1.10.0,<2.0a0|>=1.10.0|>=1.11.0,<2.0a0|>=1.9.0,<2.0a0']

Package kiwisolver conflicts for:
matplotlib-base -> kiwisolver[version='>=1.0.1|>=1.3.1']
librosa -> matplotlib-base[version='>=3.3.0'] -> kiwisolver[version='>=1.0.1|>=1.3.1']

Package fontconfig conflicts for:
libass -> fontconfig[version='>=2.14.2,<3.0a0']
harfbuzz -> cairo[version='>=1.18.0,<2.0a0'] -> fontconfig[version='>=2.13.1,<2.13.96.0a0|>=2.13.96,<3.0a0|>=2.14.2,<3.0a0|>=2.14.1,<3.0a0|>=2.13.1,<3.0a0']
cairo -> fontconfig[version='>=2.13.1,<2.13.96.0a0|>=2.13.96,<3.0a0|>=2.14.2,<3.0a0|>=2.14.1,<3.0a0|>=2.13.1,<3.0a0']
audioread -> ffmpeg -> fontconfig[version='>=2.13.96,<3.0a0|>=2.14.0,<3.0a0|>=2.14.1,<3.0a0|>=2.14.2,<3.0a0']
ffmpeg -> fontconfig[version='>=2.13.96,<3.0a0|>=2.14.0,<3.0a0|>=2.14.1,<3.0a0|>=2.14.2,<3.0a0']

Package glib-tools conflicts for:
cairo -> glib[version='>=2.69.1,<3.0a0'] -> glib-tools[version='2.70.0|2.70.0|2.70.1|2.70.2|2.70.2|2.70.2|2.70.2|2.70.2|2.72.1|2.74.0|2.74.1|2.74.1|2.76.1|2.76.2|2.76.3|2.76.4|2.78.0|2.78.1|2.78.1|2.78.2|2.78.3|2.78.4|2.78.4|2.78.4|2.78.4|2.80.0',build='hccf11d3_0|hccf11d3_2|hccf11d3_3|ha614eb4_0|ha614eb4_0|ha614eb4_0|h9e231a4_0|h9e231a4_1|h9e231a4_0|h9e231a4_0|h1059232_3|hb9a4d99_1|hb9a4d99_3|hb9a4d99_2|hb9a4d99_0|hb9a4d99_4|h1059232_4|h1059232_0|ha614eb4_0|hb5ab8b9_0|hb5ab8b9_1|hb5ab8b9_0|hb5ab8b9_0|h332123e_0|hccf11d3_4|hccf11d3_1|hccf11d3_0|hccf11d3_0|hccf11d3_1']
harfbuzz -> glib[version='>=2.69.1,<3.0a0'] -> glib-tools[version='2.70.0|2.70.0|2.70.1|2.70.2|2.70.2|2.70.2|2.70.2|2.70.2|2.72.1|2.74.0|2.74.1|2.74.1|2.76.1|2.76.2|2.76.3|2.76.4|2.78.0|2.78.1|2.78.1|2.78.2|2.78.3|2.78.4|2.78.4|2.78.4|2.78.4|2.80.0',build='hccf11d3_0|hccf11d3_2|hccf11d3_3|ha614eb4_0|ha614eb4_0|ha614eb4_0|h9e231a4_0|h9e231a4_1|h9e231a4_0|h9e231a4_0|h1059232_3|hb9a4d99_1|hb9a4d99_3|hb9a4d99_2|hb9a4d99_0|hb9a4d99_4|h1059232_4|h1059232_0|ha614eb4_0|hb5ab8b9_0|hb5ab8b9_1|hb5ab8b9_0|hb5ab8b9_0|h332123e_0|hccf11d3_4|hccf11d3_1|hccf11d3_0|hccf11d3_0|hccf11d3_1']

Package typing_extensions conflicts for:
lazy_loader -> importlib-metadata -> typing_extensions[version='>=3.6.4']
pooch -> platformdirs[version='>=2.5.0'] -> typing_extensions[version='>=4.7.1']
numba -> importlib-metadata -> typing_extensions[version='>=3.6.4']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> typing_extensions[version='3.7.4.*|>=3.6.6|>=3.6.6,<4.6.0|>=3.7.4,<3.8|>=3.7.4']
librosa -> typing_extensions[version='>=4.1.1']
platformdirs -> typing_extensions[version='>=4.7.1']
importlib-metadata -> typing_extensions[version='>=3.6.4']
platformdirs -> typing-extensions[version='>=4.6.3'] -> typing_extensions[version='4.10.0|4.11.0|4.9.0|4.8.0|4.7.1|4.7.0|4.6.3|4.7.1|4.7.1|4.7.1|4.7.1|4.7.1|4.6.3|4.6.3|4.6.3|4.6.3|4.6.2|4.6.1|4.6.0|4.5.0|4.5.0|4.5.0|4.5.0|4.5.0|4.4.0|4.4.0|4.4.0|4.4.0|4.4.0',build='py310hca03da5_0|py39hca03da5_0|pyha770c72_0|py310hca03da5_0|pyha770c72_0|pyha770c72_0|pyha770c72_0|pyha770c72_0|py310hca03da5_0|py311hca03da5_0|py311hca03da5_0|py310hca03da5_0|pyha770c72_0|py312hca03da5_0|py38hca03da5_0|py39hca03da5_0|py38hca03da5_0|py39hca03da5_0|py38hca03da5_0|py39hca03da5_0|py311hca03da5_0|py311hca03da5_0|py38hca03da5_0']

Package re2 conflicts for:
grpcio -> re2[version='>=2022.4.1,<2022.4.2.0a0|>=2022.6.1,<2022.6.2.0a0|>=2023.2.1,<2023.2.2.0a0']
grpcio -> libgrpc==1.62.1=h9c18a4f_0 -> re2[version='>=2023.2.2,<2023.2.3.0a0|>=2023.3.2,<2023.3.3.0a0']
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> re2[version='>=2022.4.1,<2022.4.2.0a0|>=2022.6.1,<2022.6.2.0a0|>=2023.2.1,<2023.2.2.0a0']

Package fonts-conda-forge conflicts for:
cairo -> fonts-conda-ecosystem -> fonts-conda-forge
libass -> fonts-conda-ecosystem -> fonts-conda-forge
ffmpeg -> fonts-conda-ecosystem -> fonts-conda-forge
fonts-conda-ecosystem -> fonts-conda-forge

Package threadpoolctl conflicts for:
scikit-learn -> threadpoolctl[version='>=2.0.0']
librosa -> scikit-learn[version='>=0.20.0'] -> threadpoolctl[version='>=2.0.0']

Package libgettextpo conflicts for:
libflac -> gettext[version='>=0.21.1,<1.0a0'] -> libgettextpo==0.22.5=h8fbad5d_2
gettext -> libgettextpo==0.22.5=h8fbad5d_2
libglib -> gettext[version='>=0.21.1,<1.0a0'] -> libgettextpo==0.22.5=h8fbad5d_2
libidn2 -> gettext[version='>=0.21.1,<1.0a0'] -> libgettextpo==0.22.5=h8fbad5d_2
libgettextpo-devel -> libgettextpo==0.22.5=h8fbad5d_2
gnutls -> gettext[version='>=0.21.1,<1.0a0'] -> libgettextpo==0.22.5=h8fbad5d_2

Package pillow conflicts for:
librosa -> matplotlib-base[version='>=3.3.0'] -> pillow[version='>=6.2.0|>=8']
matplotlib-base -> pillow[version='>=6.2.0|>=8']

Package svt-av1 conflicts for:
ffmpeg -> svt-av1[version='<1.0.0a0|>=1.1.0,<1.1.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0|>=1.3.0,<1.3.1.0a0|>=1.4.0,<1.4.1.0a0|>=1.4.1,<1.4.2.0a0|>=1.5.0,<1.5.1.0a0|>=1.6.0,<1.6.1.0a0|>=1.7.0,<1.7.1.0a0|>=1.8.0,<1.8.1.0a0|>=2.0.0,<2.0.1.0a0']
audioread -> ffmpeg -> svt-av1[version='<1.0.0a0|>=1.1.0,<1.1.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0|>=1.3.0,<1.3.1.0a0|>=1.4.0,<1.4.1.0a0|>=1.4.1,<1.4.2.0a0|>=1.5.0,<1.5.1.0a0|>=1.6.0,<1.6.1.0a0|>=1.7.0,<1.7.1.0a0|>=1.8.0,<1.8.1.0a0|>=2.0.0,<2.0.1.0a0']

Package liblapacke conflicts for:
blas-devel -> liblapacke==3.9.0[build='1_h9886b1c_netlib|0_h2ec9a88_netlib|5_h880f123_netlib|9_openblas|11_osxarm64_openblas|12_osxarm64_accelerate|13_osxarm64_accelerate|13_osxarm64_openblas|14_osxarm64_accelerate|14_osxarm64_openblas|15_osxarm64_accelerate|15_osxarm64_openblas|16_osxarm64_accelerate|18_osxarm64_accelerate|18_osxarm64_openblas|20_osxarm64_accelerate|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|21_osxarm64_openblas|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|17_osxarm64_accelerate|17_osxarm64_openblas|16_osxarm64_openblas|12_osxarm64_openblas|10_openblas|8_openblas|7_openblas']
numpy -> blas=[build=openblas] -> liblapacke==3.9.0[build='3_openblas|5_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_openblas|15_osxarm64_openblas|16_osxarm64_openblas|18_osxarm64_openblas|20_osxarm64_openblas|22_osxarm64_openblas|21_osxarm64_openblas|19_osxarm64_openblas|17_osxarm64_openblas|14_osxarm64_openblas|12_osxarm64_openblas|8_openblas|7_openblas|6_openblas|4_openblas|2_openblas|1_openblas']
blas-devel -> blas==2.106=openblas -> liblapacke==3.9.0[build='3_openblas|5_openblas|6_openblas|4_openblas|2_openblas|1_openblas']
blas -> liblapacke==3.9.0[build='0_h9886b1c_netlib|1_h9886b1c_netlib|1_h2ec9a88_netlib|3_openblas|3_he9612bc_netlib|5_h880f123_netlib|9_openblas|11_osxarm64_openblas|12_osxarm64_accelerate|13_osxarm64_accelerate|13_osxarm64_openblas|14_osxarm64_accelerate|14_osxarm64_openblas|15_osxarm64_accelerate|15_osxarm64_openblas|16_osxarm64_accelerate|18_osxarm64_accelerate|18_osxarm64_openblas|20_osxarm64_accelerate|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|21_osxarm64_openblas|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|17_osxarm64_accelerate|17_osxarm64_openblas|16_osxarm64_openblas|12_osxarm64_openblas|10_openblas|8_openblas|7_openblas|6_openblas|5_openblas|4_h880f123_netlib|4_openblas|2_openblas|2_h2ec9a88_netlib|1_openblas|0_h2ec9a88_netlib']
scipy -> blas=[build=openblas] -> liblapacke==3.9.0[build='3_openblas|5_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_openblas|15_osxarm64_openblas|16_osxarm64_openblas|18_osxarm64_openblas|20_osxarm64_openblas|22_osxarm64_openblas|21_osxarm64_openblas|19_osxarm64_openblas|17_osxarm64_openblas|14_osxarm64_openblas|12_osxarm64_openblas|8_openblas|7_openblas|6_openblas|4_openblas|2_openblas|1_openblas']

Package openmpi conflicts for:
h5py -> openmpi[version='>=4.0.5,<5.0.0a0|>=4.1.0,<5.0a0|>=4.1.1,<5.0a0|>=4.1.2,<5.0a0|>=4.1.4,<5.0a0|>=4.1.5,<5.0a0|>=4.1.6,<5.0a0']
h5py -> mpi4py[version='>=3.0'] -> openmpi[version='>=4.0,<5.0.0a0|>=4.1,<4.2.0a0|>=4.1.3,<5.0a0|>=4.1.4,<4.2.0a0']

Package libogg conflicts for:
libvorbis -> libogg[version='>=1.3.4,<1.4.0a0|>=1.3.5,<1.4.0a0|>=1.3.5,<2.0a0']
pysoundfile -> libsndfile[version='>=1.2'] -> libogg[version='>=1.3.4,<1.4.0a0']
libsndfile -> libflac[version='>=1.4.3,<1.5.0a0'] -> libogg[version='1.3.*|>=1.3.5,<1.4.0a0|>=1.3.5,<2.0a0']
libsndfile -> libogg[version='>=1.3.4,<1.4.0a0']
libflac -> libogg[version='1.3.*|>=1.3.4,<1.4.0a0']

Package libnghttp2 conflicts for:
libcurl -> libnghttp2[version='>=1.41.0,<2.0a0|>=1.43.0,<2.0a0|>=1.47.0,<2.0a0|>=1.51.0,<2.0a0|>=1.52.0,<2.0a0|>=1.58.0,<2.0a0|>=1.57.0|>=1.57.0,<2.0a0|>=1.52.0|>=1.46.0|>=1.46.0,<2.0a0']
hdf5 -> libcurl[version='>=8.4.0,<9.0a0'] -> libnghttp2[version='>=1.41.0,<2.0a0|>=1.43.0,<2.0a0|>=1.46.0,<2.0a0|>=1.46.0|>=1.47.0,<2.0a0|>=1.51.0,<2.0a0|>=1.52.0|>=1.52.0,<2.0a0|>=1.58.0,<2.0a0|>=1.57.0|>=1.57.0,<2.0a0']

Package gettext conflicts for:
gnutls -> gettext[version='>=0.19.8.1|>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0|>=0.21.0,<1.0a0']
libflac -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0']
harfbuzz -> libglib[version='>=2.78.1,<3.0a0'] -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0|>=0.21.0,<1.0a0']
libglib -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0']
cairo -> libglib[version='>=2.78.0,<3.0a0'] -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0|>=0.21.0,<1.0a0']
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> gettext[version='>=0.19.8.1|>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0|>=0.21.0,<1.0a0']
libidn2 -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0']
libsndfile -> libflac[version='>=1.4.3,<1.5.0a0'] -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0']

Package libvpx conflicts for:
audioread -> ffmpeg -> libvpx[version='>=1.10.0,<1.11.0a0|>=1.11.0,<1.12.0a0|>=1.13.0,<1.14.0a0|>=1.13.1,<1.14.0a0|>=1.14.0,<1.15.0a0']
ffmpeg -> libvpx[version='>=1.10.0,<1.11.0a0|>=1.11.0,<1.12.0a0|>=1.13.0,<1.14.0a0|>=1.13.1,<1.14.0a0|>=1.14.0,<1.15.0a0']

Package libasprintf conflicts for:
gnutls -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
libglib -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
libflac -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
gettext -> libasprintf==0.22.5=h8fbad5d_2
libidn2 -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
libasprintf-devel -> libasprintf==0.22.5=h8fbad5d_2

Package pypy3.8 conflicts for:
cffi -> python_abi==3.8[build=*_pypy38_pp73] -> pypy3.8=7.3
cffi -> pypy3.8[version='7.3.11.*|7.3.8.*']

Package matplotlib-base conflicts for:
librosa -> matplotlib[version='>=1.5.0'] -> matplotlib-base[version='>=3.3.2,<3.3.3.0a0|>=3.3.3,<3.3.4.0a0|>=3.3.4,<3.3.5.0a0|>=3.4.1,<3.4.2.0a0|>=3.4.2,<3.4.3.0a0|>=3.4.3,<3.4.4.0a0|>=3.5.0,<3.5.1.0a0|>=3.5.1,<3.5.2.0a0|>=3.5.2,<3.5.3.0a0|>=3.5.3,<3.5.4.0a0|>=3.6.0,<3.6.1.0a0|>=3.6.1,<3.6.2.0a0|>=3.6.2,<3.6.3.0a0|>=3.6.3,<3.6.4.0a0|>=3.7.0,<3.7.1.0a0|>=3.7.1,<3.7.2.0a0|>=3.7.2,<3.7.3.0a0|>=3.7.3,<3.7.4.0a0|>=3.8.0,<3.8.1.0a0|>=3.8.1,<3.8.2.0a0|>=3.8.2,<3.8.3.0a0|>=3.8.3,<3.8.4.0a0']
librosa -> matplotlib-base[version='>=1.5.0|>=3.3.0']

Package libvorbis conflicts for:
pysoundfile -> libsndfile[version='>=1.2'] -> libvorbis[version='>=1.3.7,<1.4.0a0']
libsndfile -> libvorbis[version='>=1.3.7,<1.4.0a0']

Package dav1d conflicts for:
audioread -> ffmpeg -> dav1d[version='>=1.0.0,<1.0.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0']
ffmpeg -> dav1d[version='>=1.0.0,<1.0.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0']

Package cairo conflicts for:
ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0'] -> cairo[version='>=1.18.0,<2.0a0']
libass -> harfbuzz[version='>=8.1.1,<9.0a0'] -> cairo[version='>=1.16.0,<2.0a0|>=1.18.0,<2.0a0']
harfbuzz -> cairo[version='>=1.16.0,<2.0.0a0|>=1.16.0,<2.0a0|>=1.18.0,<2.0a0']

Package libunistring conflicts for:
gnutls -> libunistring[version='>=0,<1.0a0']
ffmpeg -> gnutls[version='>=3.6.13,<3.7.0a0'] -> libunistring[version='>=0,<1.0a0']
libidn2 -> libunistring[version='>=0,<1.0a0']

Package fonttools conflicts for:
matplotlib-base -> fonttools[version='>=4.22.0']
librosa -> matplotlib-base[version='>=3.3.0'] -> fonttools[version='>=4.22.0']

Package libflac conflicts for:
libsndfile -> libflac[version='>=1.3.3,<1.4.0a0|>=1.4.1,<1.5.0a0|>=1.4.2,<1.5.0a0|>=1.4.3,<1.5.0a0']
pysoundfile -> libsndfile[version='>=1.2'] -> libflac[version='>=1.3.3,<1.4.0a0|>=1.4.1,<1.5.0a0|>=1.4.2,<1.5.0a0|>=1.4.3,<1.5.0a0']

Package nettle conflicts for:
gnutls -> nettle[version='>=3.4.1|>=3.7,<3.8.0a0|>=3.8,<3.9.0a0|>=3.8.1,<3.9.0a0|>=3.9.1,<3.10.0a0|>=3.6,<3.7.0a0|>=3.7.3,<3.8.0a0']
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> nettle[version='>=3.4.1|>=3.7,<3.8.0a0|>=3.8,<3.9.0a0|>=3.8.1,<3.9.0a0|>=3.9.1,<3.10.0a0|>=3.6,<3.7.0a0|>=3.7.3,<3.8.0a0']

Package libev conflicts for:
libnghttp2 -> libev[version='>=4.11|>=4.33,<4.34.0a0|>=4.33,<5.0a0']
libcurl -> libnghttp2[version='>=1.58.0,<2.0a0'] -> libev[version='>=4.11|>=4.33,<4.34.0a0|>=4.33,<5.0a0']

Package libopus conflicts for:
pysoundfile -> libsndfile[version='>=1.2'] -> libopus[version='>=1.3.1,<2.0a0']
audioread -> ffmpeg -> libopus[version='>=1.3,<2.0a0|>=1.3.1,<2.0a0']
ffmpeg -> libopus[version='>=1.3,<2.0a0|>=1.3.1,<2.0a0']
libsndfile -> libopus[version='>=1.3.1,<2.0a0']

Package appdirs conflicts for:
scipy -> pooch -> appdirs[version='>=1.3.0']
pooch -> appdirs[version='>=1.3.0']
librosa -> pooch[version='>=1.0'] -> appdirs[version='>=1.3.0']

Package python-dateutil conflicts for:
matplotlib-base -> python-dateutil[version='>=2.1|>=2.7']
librosa -> matplotlib-base[version='>=3.3.0'] -> python-dateutil[version='>=2.1|>=2.7']

Package gtest conflicts for:
grpcio -> abseil-cpp[version='>=20230802.0,<20230802.1.0a0'] -> gtest[version='>=1.14.0,<1.14.1.0a0']
libprotobuf -> gtest[version='>=1.14.0,<1.14.1.0a0']
protobuf -> libprotobuf[version='>=4.23.4,<4.23.5.0a0'] -> gtest[version='>=1.14.0,<1.14.1.0a0']

Package pypy3.7 conflicts for:
python_abi -> pypy3.7=7.3
pysoundfile -> cffi -> pypy3.7[version='7.3.3.*|7.3.4.*|7.3.5.*|7.3.7.*']
cffi -> pypy3.7[version='7.3.3.*|7.3.4.*|7.3.5.*|7.3.7.*']
cffi -> python_abi==3.7[build=*_pypy37_pp73] -> pypy3.7=7.3

Package platformdirs conflicts for:
pooch -> platformdirs[version='>=2.5.0']
scipy -> pooch -> platformdirs[version='>=2.5.0']
librosa -> pooch[version='>=1.0'] -> platformdirs[version='>=2.5.0']

Package fonts-anaconda conflicts for:
libass -> fonts-conda-ecosystem -> fonts-anaconda
cairo -> fonts-conda-ecosystem -> fonts-anaconda
ffmpeg -> fonts-conda-ecosystem -> fonts-anaconda
fonts-conda-ecosystem -> fonts-anaconda

Package chardet conflicts for:
requests -> chardet[version='>=3.0.2,<4|>=3.0.2,<5']
pooch -> requests[version='>=2.19.0'] -> chardet[version='>=3.0.2,<4|>=3.0.2,<5']

Package protobuf conflicts for:
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> protobuf[version='>=3.19.6|>=3.20.3,<5,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5|>=3.9.2|>=3.6.1|>=3.6.0|>=3.9.2,<3.20']
tensorflow-deps -> protobuf[version='>=3.19.1,<3.20']

Package pypy3.9 conflicts for:
cffi -> python_abi==3.9[build=*_pypy39_pp73] -> pypy3.9=7.3
cffi -> pypy3.9[version='7.3.11.*|7.3.15.*|7.3.8.*']

Package joblib conflicts for:
scikit-learn -> joblib[version='>=0.11|>=1.0.0|>=1.1.1|>=1.2.0']
librosa -> scikit-learn[version='>=0.20.0'] -> joblib[version='>=0.11|>=1.0.0|>=1.1.1|>=1.2.0']
librosa -> joblib[version='>=0.12.0|>=0.14.0|>=0.7.0']

Package mpg123 conflicts for:
pysoundfile -> libsndfile[version='>=1.2'] -> mpg123[version='>=1.30.2,<1.31.0a0|>=1.31.1,<1.32.0a0|>=1.31.3,<1.32.0a0|>=1.32.1,<1.33.0a0']
libsndfile -> mpg123[version='>=1.30.2,<1.31.0a0|>=1.31.1,<1.32.0a0|>=1.31.3,<1.32.0a0|>=1.32.1,<1.33.0a0']

Package c-ares conflicts for:
grpcio -> libgrpc==1.62.1=h9c18a4f_0 -> c-ares[version='>=1.19.0,<2.0a0|>=1.20.1,<2.0a0|>=1.21.0,<2.0a0|>=1.22.1,<2.0a0|>=1.25.0,<2.0a0|>=1.26.0,<2.0a0|>=1.27.0,<2.0a0']
grpcio -> c-ares[version='>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.19.1,<2.0a0']
libnghttp2 -> c-ares[version='>=1.16.1,<2.0a0|>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.20.1,<2.0a0|>=1.21.0,<2.0a0|>=1.23.0,<2.0a0|>=1.7.5|>=1.19.1,<2.0a0|>=1.19.0,<2.0a0']
libcurl -> libnghttp2[version='>=1.58.0,<2.0a0'] -> c-ares[version='>=1.16.1,<2.0a0|>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.20.1,<2.0a0|>=1.21.0,<2.0a0|>=1.23.0,<2.0a0|>=1.19.1,<2.0a0|>=1.7.5|>=1.19.0,<2.0a0']
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> c-ares[version='>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.19.1,<2.0a0']

Package libidn2 conflicts for:
gnutls -> libidn2[version='>=2,<3.0a0']
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> libidn2[version='>=2,<3.0a0']

Package harfbuzz conflicts for:
audioread -> ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0']
ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0']
libass -> harfbuzz[version='>=7.2.0,<8.0a0|>=8.1.1,<9.0a0']
ffmpeg -> libass[version='>=0.17.1,<0.17.2.0a0'] -> harfbuzz[version='>=7.2.0,<8.0a0|>=8.1.1,<9.0a0']

Package openblas conflicts for:
blas-devel -> openblas[version='0.3.18.*|0.3.20.*|0.3.21.*|0.3.23.*|0.3.24.*|0.3.25.*|0.3.26.*|0.3.27.*']
blas -> blas-devel==3.9.0=22_osxarm64_openblas -> openblas[version='0.3.18.*|0.3.20.*|0.3.21.*|0.3.23.*|0.3.24.*|0.3.25.*|0.3.26.*|0.3.27.*']

Package pcre conflicts for:
libglib -> pcre[version='>=8.44,<9.0a0|>=8.45,<9.0a0']
harfbuzz -> libglib[version='>=2.72.1,<3.0a0'] -> pcre[version='>=8.44,<9.0a0|>=8.45,<9.0a0']
cairo -> libglib[version='>=2.72.1,<3.0a0'] -> pcre[version='>=8.44,<9.0a0|>=8.45,<9.0a0']

Package libcurl conflicts for:
h5py -> hdf5[version='>=1.14.3,<1.14.4.0a0'] -> libcurl[version='>=7.71.1,<8.0a0|>=7.77.0,<9.0a0|>=7.79.1,<9.0a0|>=7.80.0,<9.0a0|>=7.81.0,<9.0a0|>=7.83.1,<9.0a0|>=7.87.0,<9.0a0|>=8.1.2,<9.0a0|>=8.2.1,<9.0a0|>=8.4.0,<9.0a0|>=7.88.1,<9.0a0|>=7.88.1,<8.0a0|>=7.82.0,<8.0a0|>=7.71.1,<9.0a0']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> libcurl[version='>=7.76.0,<9.0a0|>=7.76.1,<9.0a0|>=7.78.0,<9.0a0|>=7.79.1,<9.0a0|>=7.80.0,<9.0a0|>=7.81.0,<9.0a0|>=7.83.0,<9.0a0|>=7.83.1,<9.0a0|>=7.87.0,<9.0a0|>=7.88.1,<9.0a0|>=8.1.2,<9.0a0|>=8.3.0,<9.0a0|>=8.4.0,<9.0a0|>=8.5.0,<9.0a0|>=7.88.1,<8.0a0|>=7.86.0,<8.0a0']
hdf5 -> libcurl[version='>=7.71.1,<8.0a0|>=7.71.1,<9.0a0|>=7.76.0,<9.0a0|>=7.77.0,<9.0a0|>=7.79.1,<9.0a0|>=7.80.0,<9.0a0|>=7.81.0,<9.0a0|>=7.83.1,<9.0a0|>=7.87.0,<9.0a0|>=8.1.2,<9.0a0|>=8.2.1,<9.0a0|>=8.4.0,<9.0a0|>=7.88.1,<9.0a0|>=7.88.1,<8.0a0|>=7.82.0,<8.0a0']

Package xorg-libxdmcp conflicts for:
pillow -> libxcb[version='>=1.15,<1.16.0a0'] -> xorg-libxdmcp
libxcb -> xorg-libxdmcp

Package tbb conflicts for:
librosa -> numba[version='>=0.51.0'] -> tbb[version='>=2021.3.0|>=2021.5.0|>=2021.8.0']
numba -> tbb[version='>=2021.3.0|>=2021.5.0|>=2021.8.0']
ffmpeg -> libopenvino[version='>=2024.0.0,<2024.0.1.0a0'] -> tbb[version='>=2021.11.0|>=2021.5.0']

Package pooch conflicts for:
librosa -> pooch[version='>=1.0|>=1.0,<1.7']
scipy -> pooch
scikit-learn -> scipy -> pooch
librosa -> scipy[version='>=1.2.0'] -> pooch

Package p11-kit conflicts for:
gnutls -> p11-kit[version='>=0.23.21,<0.24.0a0|>=0.24.1,<0.25.0a0']
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> p11-kit[version='>=0.23.21,<0.24.0a0|>=0.24.1,<0.25.0a0']

Package lerc conflicts for:
lcms2 -> libtiff[version='>=4.6.0,<4.7.0a0'] -> lerc[version='>=2.2.1,<3.0a0|>=3.0,<4.0a0|>=4.0.0,<5.0a0']
openjpeg -> libtiff[version='>=4.6.0,<4.7.0a0'] -> lerc[version='>=2.2.1,<3.0a0|>=3.0,<4.0a0|>=4.0.0,<5.0a0']
libtiff -> lerc[version='>=2.2.1,<3.0a0|>=3.0,<4.0a0|>=4.0.0,<5.0a0']
pillow -> libtiff[version='>=4.6.0,<4.7.0a0'] -> lerc[version='>=2.2.1,<3.0a0|>=3.0,<4.0a0|>=4.0.0,<5.0a0']

Package libllvm14 conflicts for:
numba -> libllvm14[version='>=14.0.6,<14.1.0a0']
llvmlite -> libllvm14[version='>=14.0.6,<14.1.0a0']
librosa -> numba[version='>=0.51.0'] -> libllvm14[version='>=14.0.6,<14.1.0a0']

Package mpi4py conflicts for:
tensorflow-deps -> h5py[version='>=3.6.0,<3.7'] -> mpi4py[version='>=3.0']
h5py -> mpi4py[version='>=3.0']

Package x265 conflicts for:
ffmpeg -> x265[version='>=3.5,<3.6.0a0']
audioread -> ffmpeg -> x265[version='>=3.5,<3.6.0a0']

Package font-ttf-dejavu-sans-mono conflicts for:
fonts-conda-forge -> font-ttf-dejavu-sans-mono
fonts-conda-ecosystem -> fonts-conda-forge -> font-ttf-dejavu-sans-mono

Package pthread-stubs conflicts for:
libxcb -> pthread-stubs
pillow -> libxcb[version='>=1.15,<1.16.0a0'] -> pthread-stubs

Package libdeflate conflicts for:
lcms2 -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']
openjpeg -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']
pillow -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']
libtiff -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']

Package font-ttf-ubuntu conflicts for:
fonts-conda-forge -> font-ttf-ubuntu
fonts-conda-ecosystem -> fonts-conda-forge -> font-ttf-ubuntu

Package pypy3.6 conflicts for:
cffi -> python_abi==3.6[build=*_pypy36_pp73] -> pypy3.6=7.3
cffi -> pypy3.6[version='7.3.0.*|7.3.1.*|7.3.2.*|7.3.3.*']

Package importlib-metadata conflicts for:
lazy_loader -> importlib-metadata
numba -> importlib_metadata -> importlib-metadata[version='>=1.1.3,<1.1.4.0a0|>=1.5.0,<1.5.1.0a0|>=1.5.2,<1.5.3.0a0|>=1.6.0,<1.6.1.0a0|>=1.6.1,<1.6.2.0a0|>=1.7.0,<1.7.1.0a0|>=2.0.0,<2.0.1.0a0|>=3.0.0,<3.0.1.0a0|>=3.1.0,<3.1.1.0a0|>=3.1.1,<3.1.2.0a0|>=3.10.0,<3.10.1.0a0|>=3.10.1,<3.10.2.0a0|>=4.0.1,<4.0.2.0a0|>=4.10.0,<4.10.1.0a0|>=4.10.1,<4.10.2.0a0|>=4.11.0,<4.11.1.0a0|>=4.11.1,<4.11.2.0a0|>=4.11.2,<4.11.3.0a0|>=4.11.3,<4.11.4.0a0|>=4.11.4,<4.11.5.0a0|>=4.13.0,<4.13.1.0a0|>=5.0.0,<5.0.1.0a0|>=5.1.0,<5.1.1.0a0|>=5.2.0,<5.2.1.0a0|>=6.0.0,<6.0.1.0a0|>=6.1.0,<6.1.1.0a0|>=6.10.0,<6.10.1.0a0|>=7.0.0,<7.0.1.0a0|>=7.0.1,<7.0.2.0a0|>=7.0.2,<7.0.3.0a0|>=7.1.0,<7.1.1.0a0|>=6.9.0,<6.9.1.0a0|>=6.8.0,<6.8.1.0a0|>=6.7.0,<6.7.1.0a0|>=6.6.0,<6.6.1.0a0|>=6.5.1,<6.5.2.0a0|>=6.5.0,<6.5.1.0a0|>=6.4.1,<6.4.2.0a0|>=6.4.0,<6.4.1.0a0|>=6.3.0,<6.3.1.0a0|>=6.2.1,<6.2.2.0a0|>=6.2.0,<6.2.1.0a0|>=4.9.0,<4.9.1.0a0|>=4.8.3,<4.8.4.0a0|>=4.8.2,<4.8.3.0a0|>=4.8.1,<4.8.2.0a0|>=4.8.0,<4.8.1.0a0|>=4.7.1,<4.7.2.0a0|>=4.7.0,<4.7.1.0a0|>=4.6.4,<4.6.5.0a0|>=4.6.3,<4.6.4.0a0|>=4.6.2,<4.6.3.0a0|>=4.6.1,<4.6.2.0a0|>=4.6.0,<4.6.1.0a0|>=4.5.0,<4.5.1.0a0|>=4.4.0,<4.4.1.0a0|>=4.3.1,<4.3.2.0a0|>=4.3.0,<4.3.1.0a0|>=4.2.0,<4.2.1.0a0|>=3.9.1,<3.9.2.0a0|>=3.9.0,<3.9.1.0a0|>=3.8.1,<3.8.2.0a0|>=3.8.0,<3.8.1.0a0|>=3.7.3,<3.7.4.0a0|>=3.7.2,<3.7.3.0a0|>=3.7.0,<3.7.1.0a0|>=3.6.0,<3.6.1.0a0|>=3.4.0,<3.4.1.0a0|>=3.3.0,<3.3.1.0a0']
librosa -> lazy_loader[version='>=0.1'] -> importlib-metadata
numba -> importlib-metadata

Package libedit conflicts for:
krb5 -> libedit[version='>=3.1.20191231,<3.2.0a0|>=3.1.20191231,<4.0a0|>=3.1.20221030,<3.2.0a0|>=3.1.20221030,<4.0a0|>=3.1.20210216,<3.2.0a0|>=3.1.20210216,<4.0a0']
libcurl -> krb5[version='>=1.21.2,<1.22.0a0'] -> libedit[version='>=3.1.20191231,<3.2.0a0|>=3.1.20191231,<4.0a0|>=3.1.20221030,<3.2.0a0|>=3.1.20221030,<4.0a0|>=3.1.20210216,<3.2.0a0|>=3.1.20210216,<4.0a0']

Package libass conflicts for:
audioread -> ffmpeg -> libass[version='>=0.17.1,<0.17.2.0a0']
ffmpeg -> libass[version='>=0.17.1,<0.17.2.0a0']

Package libwebp conflicts for:
matplotlib-base -> pillow[version='>=8'] -> libwebp[version='>=0.3.0|>=1.2.0,<1.3.0a0|>=1.3.2,<2.0a0']
pillow -> libwebp[version='>=0.3.0|>=1.2.0,<1.3.0a0|>=1.3.2,<2.0a0']

Package pyopenssl conflicts for:
urllib3 -> pyopenssl[version='>=0.14']
requests -> urllib3[version='>=1.21.1,<3'] -> pyopenssl[version='>=0.14']

Package pixman conflicts for:
cairo -> pixman[version='>=0.40.0,<1.0a0|>=0.42.2,<1.0a0']
harfbuzz -> cairo[version='>=1.18.0,<2.0a0'] -> pixman[version='>=0.40.0,<1.0a0|>=0.42.2,<1.0a0']The following specifications were found to be incompatible with your system:

  - feature:/osx-arm64::__osx==13.6.3=0
  - feature:/osx-arm64::__unix==0=0
  - feature:|@/osx-arm64::__osx==13.6.3=0
  - feature:|@/osx-arm64::__unix==0=0
  - aom -> __osx[version='>=10.9']
  - audioread -> ffmpeg -> __osx[version='>=10.9']
  - cairo -> __osx[version='>=10.9']
  - ffmpeg -> __osx[version='>=10.9']
  - gettext -> ncurses[version='>=6.4,<7.0a0'] -> __osx[version='>=10.9']
  - gmp -> __osx[version='>=10.9']
  - gnutls -> __osx[version='>=10.9']
  - grpcio -> __osx[version='>=10.9']
  - h5py -> hdf5[version='>=1.14.3,<1.14.4.0a0'] -> __osx[version='>=10.9']
  - harfbuzz -> __osx[version='>=10.9']
  - hdf5 -> __osx[version='>=10.9']
  - libass -> harfbuzz[version='>=8.1.1,<9.0a0'] -> __osx[version='>=10.9']
  - libcurl -> libnghttp2[version='>=1.58.0,<2.0a0'] -> __osx[version='>=10.9']
  - libedit -> ncurses[version='>=6.2,<7.0.0a0'] -> __osx[version='>=10.9']
  - libglib -> __osx[version='>=10.9']
  - libnghttp2 -> __osx[version='>=10.9']
  - libprotobuf -> __osx[version='>=10.9']
  - librosa -> matplotlib-base[version='>=3.3.0'] -> __osx[version='>=10.9']
  - matplotlib-base -> __osx[version='>=10.9']
  - msgpack-python -> __osx[version='>=10.9']
  - ncurses -> __osx[version='>=10.9']
  - nettle -> gmp[version='>=6.2.1,<7.0a0'] -> __osx[version='>=10.9']
  - numba -> __osx[version='>=10.9']
  - numpy -> __osx[version='>=10.9']
  - openh264 -> __osx[version='>=10.9']
  - protobuf -> __osx[version='>=10.9']
  - pysocks -> __unix
  - pysocks -> __win
  - pysoundfile -> numpy -> __osx[version='>=10.9']
  - python=3.8 -> ncurses[version='>=6.4,<7.0a0'] -> __osx[version='>=10.9']
  - readline -> ncurses[version='>=6.3,<7.0a0'] -> __osx[version='>=10.9']
  - scikit-learn -> __osx[version='>=10.9']
  - scipy -> __osx[version='>=10.9']
  - soxr-python -> numpy[version='>=1.23.5,<2.0a0'] -> __osx[version='>=10.9']
  - svt-av1 -> __osx[version='>=10.9']
  - tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> __osx[version='>=10.9']
  - urllib3 -> pysocks[version='>=1.5.6,<2.0,!=1.5.7'] -> __unix

Your installed version is: 13.6.3


(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python -c "import tensorflow as tf; print(tf.__version__)"

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'tensorflow'
(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) cd Downloads

bash Miniforge3-MacOSX-arm64.sh

brew install miniforge
cd: no such file or directory: Downloads
bash: Miniforge3-MacOSX-arm64.sh: No such file or directory
==> Auto-updating Homebrew...
Adjust how often this is run with HOMEBREW_AUTO_UPDATE_SECS or disable with
HOMEBREW_NO_AUTO_UPDATE. Hide these hints with HOMEBREW_NO_ENV_HINTS (see `man brew`).
ç^C
(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda create -n tensorflow python=3.8

Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 23.3.1
  latest version: 24.3.0

Please update conda by running

    $ conda update -n base -c conda-forge conda

Or to minimize the number of packages updated during conda update use

     conda install conda=24.3.0



## Package Plan ##

  environment location: /Users/deangladish/miniforge3/envs/tensorflow

  added / updated specs:
    - python=3.8


The following NEW packages will be INSTALLED:

  bzip2              conda-forge/osx-arm64::bzip2-1.0.8-h93a5062_5
  ca-certificates    conda-forge/osx-arm64::ca-certificates-2024.2.2-hf0a4a13_0
  libffi             conda-forge/osx-arm64::libffi-3.4.2-h3422bc3_5
  libsqlite          conda-forge/osx-arm64::libsqlite-3.45.2-h091b4b1_0
  libzlib            conda-forge/osx-arm64::libzlib-1.2.13-h53f4e23_5
  ncurses            conda-forge/osx-arm64::ncurses-6.4.20240210-h078ce10_0
  openssl            conda-forge/osx-arm64::openssl-3.2.1-h0d3ecfb_1
  pip                conda-forge/noarch::pip-24.0-pyhd8ed1ab_0
  python             conda-forge/osx-arm64::python-3.8.19-h2469fbe_0_cpython
  readline           conda-forge/osx-arm64::readline-8.2-h92ec313_1
  setuptools         conda-forge/noarch::setuptools-69.2.0-pyhd8ed1ab_0
  tk                 conda-forge/osx-arm64::tk-8.6.13-h5083fa2_1
  wheel              conda-forge/noarch::wheel-0.43.0-pyhd8ed1ab_1
  xz                 conda-forge/osx-arm64::xz-5.2.6-h57fd34a_0


Proceed ([y]/n)? y


Downloading and Extracting Packages

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate tensorflow
#
# To deactivate an active environment, use
#
#     $ conda deactivate

(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) source activate tensorflow

source: no such file or directory: activate
(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda activate tensorflow
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda install -c apple tensorflow-deps
pip install tensorflow-macos==2.9.0
pip install tensorflow-metal==0.5.0
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Solving environment: failed with repodata from current_repodata.json, will retry with next repodata source.
Collecting package metadata (repodata.json): \ WARNING conda.models.version:get_matcher(546): Using .* with relational operator is superfluous and deprecated and will be removed in a future version of conda. Your spec was 1.8.0.*, but conda is ignoring the .* and treating it as 1.8.0
WARNING conda.models.version:get_matcher(546): Using .* with relational operator is superfluous and deprecated and will be removed in a future version of conda. Your spec was 1.9.0.*, but conda is ignoring the .* and treating it as 1.9.0
done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 23.3.1
  latest version: 24.3.0

Please update conda by running

    $ conda update -n base -c conda-forge conda

Or to minimize the number of packages updated during conda update use

     conda install conda=24.3.0



## Package Plan ##

  environment location: /Users/deangladish/miniforge3/envs/tensorflow

  added / updated specs:
    - tensorflow-deps


The following NEW packages will be INSTALLED:

  c-ares             conda-forge/osx-arm64::c-ares-1.28.1-h93a5062_0
  cached-property    conda-forge/noarch::cached-property-1.5.2-hd8ed1ab_1
  cached_property    conda-forge/noarch::cached_property-1.5.2-pyha770c72_1
  grpcio             conda-forge/osx-arm64::grpcio-1.46.3-py38h1ef021a_0
  h5py               conda-forge/osx-arm64::h5py-3.6.0-nompi_py38hacf61ce_100
  hdf5               conda-forge/osx-arm64::hdf5-1.12.1-nompi_hd9dbc9e_104
  krb5               conda-forge/osx-arm64::krb5-1.21.2-h92f50d5_0
  libblas            conda-forge/osx-arm64::libblas-3.9.0-22_osxarm64_openblas
  libcblas           conda-forge/osx-arm64::libcblas-3.9.0-22_osxarm64_openblas
  libcurl            conda-forge/osx-arm64::libcurl-8.7.1-h2d989ff_0
  libcxx             conda-forge/osx-arm64::libcxx-16.0.6-h4653b0c_0
  libedit            conda-forge/osx-arm64::libedit-3.1.20191231-hc8eb9b7_2
  libev              conda-forge/osx-arm64::libev-4.33-h93a5062_2
  libgfortran        conda-forge/osx-arm64::libgfortran-5.0.0-13_2_0_hd922786_3
  libgfortran5       conda-forge/osx-arm64::libgfortran5-13.2.0-hf226fd6_3
  liblapack          conda-forge/osx-arm64::liblapack-3.9.0-22_osxarm64_openblas
  libnghttp2         conda-forge/osx-arm64::libnghttp2-1.58.0-ha4dd798_1
  libopenblas        conda-forge/osx-arm64::libopenblas-0.3.27-openmp_h6c19121_0
  libprotobuf        conda-forge/osx-arm64::libprotobuf-3.19.6-hb5ab8b9_0
  libssh2            conda-forge/osx-arm64::libssh2-1.11.0-h7a5bd25_0
  llvm-openmp        conda-forge/osx-arm64::llvm-openmp-18.1.2-hcd81f8e_0
  numpy              conda-forge/osx-arm64::numpy-1.23.2-py38h579d673_0
  protobuf           conda-forge/osx-arm64::protobuf-3.19.6-py38h2b1e499_0
  python_abi         conda-forge/osx-arm64::python_abi-3.8-4_cp38
  six                conda-forge/noarch::six-1.16.0-pyh6c4a22f_0
  tensorflow-deps    apple/osx-arm64::tensorflow-deps-2.10.0-0
  zlib               conda-forge/osx-arm64::zlib-1.2.13-h53f4e23_5
  zstd               conda-forge/osx-arm64::zstd-1.5.5-h4f39d0f_0


Proceed ([y]/n)? y


Downloading and Extracting Packages

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
Defaulting to user installation because normal site-packages is not writeable
Collecting tensorflow-macos==2.9.0
  Using cached tensorflow_macos-2.9.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (2.9 kB)
Requirement already satisfied: absl-py>=1.0.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (1.4.0)
Requirement already satisfied: astunparse>=1.6.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (1.6.3)
Collecting flatbuffers<2,>=1.12 (from tensorflow-macos==2.9.0)
  Using cached flatbuffers-1.12-py2.py3-none-any.whl.metadata (872 bytes)
Collecting gast<=0.4.0,>=0.2.1 (from tensorflow-macos==2.9.0)
  Using cached gast-0.4.0-py3-none-any.whl.metadata (1.1 kB)
Requirement already satisfied: google-pasta>=0.1.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (0.2.0)
Requirement already satisfied: h5py>=2.9.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (3.10.0)
Collecting keras-preprocessing>=1.1.1 (from tensorflow-macos==2.9.0)
  Using cached Keras_Preprocessing-1.1.2-py2.py3-none-any.whl.metadata (1.9 kB)
Requirement already satisfied: libclang>=13.0.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (18.1.1)
Requirement already satisfied: numpy>=1.20 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (1.26.4)
Requirement already satisfied: opt-einsum>=2.3.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (3.3.0)
Requirement already satisfied: packaging in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow-macos==2.9.0) (21.3)
Requirement already satisfied: protobuf>=3.9.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (4.25.3)
Requirement already satisfied: setuptools in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow-macos==2.9.0) (63.2.0)
Requirement already satisfied: six>=1.12.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow-macos==2.9.0) (2.0.1)
Requirement already satisfied: typing-extensions>=3.6.6 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow-macos==2.9.0) (4.5.0)
Requirement already satisfied: wrapt>=1.11.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (1.16.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (1.62.1)
Collecting tensorboard<2.10,>=2.9 (from tensorflow-macos==2.9.0)
  Using cached tensorboard-2.9.1-py3-none-any.whl.metadata (1.9 kB)
Collecting tensorflow-estimator<2.10.0,>=2.9.0rc0 (from tensorflow-macos==2.9.0)
  Using cached tensorflow_estimator-2.9.0-py2.py3-none-any.whl.metadata (1.3 kB)
Collecting keras<2.10.0,>=2.9.0rc0 (from tensorflow-macos==2.9.0)
  Using cached keras-2.9.0-py2.py3-none-any.whl.metadata (1.3 kB)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from astunparse>=1.6.0->tensorflow-macos==2.9.0) (0.37.1)
Collecting google-auth<3,>=1.6.3 (from tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached google_auth-2.29.0-py2.py3-none-any.whl.metadata (4.7 kB)
Collecting google-auth-oauthlib<0.5,>=0.4.1 (from tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached google_auth_oauthlib-0.4.6-py2.py3-none-any.whl.metadata (2.7 kB)
Requirement already satisfied: markdown>=2.6.8 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (3.6)
Collecting protobuf>=3.9.2 (from tensorflow-macos==2.9.0)
  Using cached protobuf-3.19.6-py2.py3-none-any.whl.metadata (828 bytes)
Requirement already satisfied: requests<3,>=2.21.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (2.28.1)
Collecting tensorboard-data-server<0.7.0,>=0.6.0 (from tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached tensorboard_data_server-0.6.1-py3-none-any.whl.metadata (1.1 kB)
Collecting tensorboard-plugin-wit>=1.6.0 (from tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached tensorboard_plugin_wit-1.8.1-py3-none-any.whl.metadata (873 bytes)
Requirement already satisfied: werkzeug>=1.0.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (3.0.2)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from packaging->tensorflow-macos==2.9.0) (3.1.1)
Collecting cachetools<6.0,>=2.0.0 (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached cachetools-5.3.3-py3-none-any.whl.metadata (5.3 kB)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (0.2.8)
Collecting rsa<5,>=3.1.4 (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached rsa-4.9-py3-none-any.whl.metadata (4.2 kB)
Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached requests_oauthlib-2.0.0-py2.py3-none-any.whl.metadata (11 kB)
Requirement already satisfied: charset-normalizer<3,>=2 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (2.1.1)
Requirement already satisfied: idna<4,>=2.5 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (3.4)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (1.26.12)
Requirement already satisfied: certifi>=2017.4.17 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (2022.9.24)
Requirement already satisfied: MarkupSafe>=2.1.1 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (2.1.1)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (0.4.8)
Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached oauthlib-3.2.2-py3-none-any.whl.metadata (7.5 kB)
Using cached tensorflow_macos-2.9.0-cp310-cp310-macosx_11_0_arm64.whl (200.6 MB)
Using cached flatbuffers-1.12-py2.py3-none-any.whl (15 kB)
Using cached gast-0.4.0-py3-none-any.whl (9.8 kB)
Using cached keras-2.9.0-py2.py3-none-any.whl (1.6 MB)
Using cached Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42 kB)
Using cached tensorboard-2.9.1-py3-none-any.whl (5.8 MB)
Using cached protobuf-3.19.6-py2.py3-none-any.whl (162 kB)
Using cached tensorflow_estimator-2.9.0-py2.py3-none-any.whl (438 kB)
Using cached google_auth-2.29.0-py2.py3-none-any.whl (189 kB)
Using cached google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
Using cached tensorboard_data_server-0.6.1-py3-none-any.whl (2.4 kB)
Using cached tensorboard_plugin_wit-1.8.1-py3-none-any.whl (781 kB)
Using cached cachetools-5.3.3-py3-none-any.whl (9.3 kB)
Using cached requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)
Using cached rsa-4.9-py3-none-any.whl (34 kB)
Using cached oauthlib-3.2.2-py3-none-any.whl (151 kB)
Installing collected packages: tensorboard-plugin-wit, keras, flatbuffers, tensorflow-estimator, tensorboard-data-server, rsa, protobuf, oauthlib, keras-preprocessing, gast, cachetools, requests-oauthlib, google-auth, google-auth-oauthlib, tensorboard, tensorflow-macos
  Attempting uninstall: keras
    Found existing installation: keras 3.1.1
    Uninstalling keras-3.1.1:
      Successfully uninstalled keras-3.1.1
  Attempting uninstall: flatbuffers
    Found existing installation: flatbuffers 24.3.25
    Uninstalling flatbuffers-24.3.25:
      Successfully uninstalled flatbuffers-24.3.25
  Attempting uninstall: tensorboard-data-server
    Found existing installation: tensorboard-data-server 0.7.2
    Uninstalling tensorboard-data-server-0.7.2:
      Successfully uninstalled tensorboard-data-server-0.7.2
  WARNING: The scripts pyrsa-decrypt, pyrsa-encrypt, pyrsa-keygen, pyrsa-priv2pub, pyrsa-sign and pyrsa-verify are installed in '/Users/deangladish/Library/Python/3.10/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  Attempting uninstall: protobuf
    Found existing installation: protobuf 4.25.3
    Uninstalling protobuf-4.25.3:
      Successfully uninstalled protobuf-4.25.3
  Attempting uninstall: gast
    Found existing installation: gast 0.5.4
    Uninstalling gast-0.5.4:
      Successfully uninstalled gast-0.5.4
  WARNING: The script google-oauthlib-tool is installed in '/Users/deangladish/Library/Python/3.10/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  Attempting uninstall: tensorboard
    Found existing installation: tensorboard 2.16.2
    Uninstalling tensorboard-2.16.2:
      Successfully uninstalled tensorboard-2.16.2
  WARNING: The script tensorboard is installed in '/Users/deangladish/Library/Python/3.10/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts estimator_ckpt_converter, import_pb_to_tensorboard, saved_model_cli, tensorboard, tf_upgrade_v2, tflite_convert, toco and toco_from_protos are installed in '/Users/deangladish/Library/Python/3.10/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
Successfully installed cachetools-5.3.3 flatbuffers-1.12 gast-0.4.0 google-auth-2.29.0 google-auth-oauthlib-0.4.6 keras-2.9.0 keras-preprocessing-1.1.2 oauthlib-3.2.2 protobuf-3.19.6 requests-oauthlib-2.0.0 rsa-4.9 tensorboard-2.9.1 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 tensorflow-estimator-2.9.0 tensorflow-macos-2.9.0
Defaulting to user installation because normal site-packages is not writeable
Collecting tensorflow-metal==0.5.0
  Downloading tensorflow_metal-0.5.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (776 bytes)
Requirement already satisfied: wheel~=0.35 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow-metal==0.5.0) (0.37.1)
Collecting six~=1.15.0 (from tensorflow-metal==0.5.0)
  Downloading six-1.15.0-py2.py3-none-any.whl.metadata (1.8 kB)
Downloading tensorflow_metal-0.5.0-cp310-cp310-macosx_11_0_arm64.whl (1.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 5.6 MB/s eta 0:00:00
Downloading six-1.15.0-py2.py3-none-any.whl (10 kB)
Installing collected packages: six, tensorflow-metal
  Attempting uninstall: six
    Found existing installation: six 1.16.0
    Uninstalling six-1.16.0:
      Successfully uninstalled six-1.16.0
Successfully installed six-1.15.0 tensorflow-metal-0.5.0
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)


TensorFlow Installation on Your M2 Chip MacBook | by misun_song | Medium
https://medium.com/@msong507/tensorflow-installation-on-your-m2-chip-macbook-30af72d76023






.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0']
scikit-learn -> numpy[version='>=1.23.5,<2.0a0'] -> numpy-base[version='1.19.2|1.19.2|1.19.5|1.19.5|1.21.2|1.21.2|1.21.2|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.21.5|1.22.3|1.22.3|1.22.3|1.22.3|1.23.1|1.23.1|1.23.1|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.3|1.23.4|1.23.4|1.23.4|1.23.5|1.23.5|1.23.5|1.23.5|1.24.3|1.24.3|1.24.3|1.24.3|1.25.0|1.25.0|1.25.0|1.25.2|1.25.2|1.25.2|1.26.0',build='py38hdc56644_1|py38hdc56644_4|py38h6269429_0|py39h974a1f5_1|py38h974a1f5_2|py38hadd41eb_3|py310h742c864_3|py38h974a1f5_0|py310h5e3e9f0_0|py310h742c864_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_0|py38hadd41eb_0|py39hadd41eb_1|py310haf87e8b_0|py38h90707a3_0|py38h90707a3_0|py311h9eb1c70_0|py38h90707a3_0|py310ha9811e2_0|py311hfbfe69c_0|py310ha9811e2_0|py310ha9811e2_0|py312he047099_0|py311hfbfe69c_0|py39ha9811e2_0|py39ha9811e2_0|py311hfbfe69c_0|py39ha9811e2_0|py311h1d85a46_0|py39h90707a3_0|py310haf87e8b_0|py310haf87e8b_0|py39h90707a3_0|py39h90707a3_0|py310h742c864_1|py38hadd41eb_1|py310h742c864_0|py311h9eb1c70_1|py39h974a1f5_0|py39hadd41eb_3|py310h5e3e9f0_2|py39h974a1f5_2|py310h5e3e9f0_1|py38h974a1f5_1|py39h6269429_0|py310h6269429_0|py39hdc56644_4|py39hdc56644_1']

Package flit-core conflicts for:
librosa -> typing_extensions[version='>=4.1.1'] -> flit-core[version='>=3.6,<4']
typing_extensions -> flit-core[version='>=3.6,<4']
importlib-metadata -> typing_extensions[version='>=3.6.4'] -> flit-core[version='>=3.6,<4']

Package jaraco.itertools conflicts for:
zipp -> jaraco.itertools
importlib-metadata -> zipp[version='>=0.5'] -> jaraco.itertools

Package brotli-bin conflicts for:
fonttools -> brotli -> brotli-bin[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_0|hb547adb_1|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']
brotli -> brotli-bin[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_0|hb547adb_1|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']
urllib3 -> brotli[version='>=1.0.9'] -> brotli-bin[version='1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.0.9|1.1.0',build='h3422bc3_6|hb547adb_0|hb547adb_1|h1a8c8d9_9|h1a8c8d9_8|h1c322ee_7|h3422bc3_5|h1a28f6b_7']

Package libxml2 conflicts for:
libflac -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
audioread -> ffmpeg -> libxml2[version='>=2.10.3,<3.0.0a0|>=2.10.4,<3.0.0a0|>=2.11.3,<3.0.0a0|>=2.11.4,<3.0.0a0|>=2.11.5,<3.0.0a0|>=2.11.6,<3.0.0a0|>=2.12.1,<3.0.0a0|>=2.12.2,<3.0.0a0|>=2.12.3,<3.0.0a0|>=2.12.4,<3.0a0|>=2.12.5,<3.0a0|>=2.12.6,<3.0a0|>=2.9.14,<3.0.0a0|>=2.9.13,<3.0.0a0|>=2.9.12,<3.0.0a0']
ffmpeg -> libxml2[version='>=2.10.3,<3.0.0a0|>=2.10.4,<3.0.0a0|>=2.11.3,<3.0.0a0|>=2.11.4,<3.0.0a0|>=2.11.5,<3.0.0a0|>=2.11.6,<3.0.0a0|>=2.12.1,<3.0.0a0|>=2.12.2,<3.0.0a0|>=2.12.3,<3.0.0a0|>=2.12.4,<3.0a0|>=2.12.5,<3.0a0|>=2.12.6,<3.0a0|>=2.9.14,<3.0.0a0|>=2.9.13,<3.0.0a0|>=2.9.12,<3.0.0a0']
libidn2 -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
ffmpeg -> fontconfig[version='>=2.14.1,<3.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.14,<2.10.0a0']
gettext -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
fontconfig -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<3.0.0a0|>=2.9.12,<3.0.0a0|>=2.9.14,<2.10.0a0|>=2.9.10,<2.10.0a0']
libglib -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
gnutls -> gettext[version='>=0.19.8.1,<1.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.10,<2.10.0a0']
cairo -> fontconfig[version='>=2.13.96,<3.0a0'] -> libxml2[version='>=2.10.3,<2.11.0a0|>=2.9.12,<3.0.0a0|>=2.9.14,<2.10.0a0|>=2.9.10,<3.0.0a0|>=2.9.10,<2.10.0a0']

Package font-ttf-source-code-pro conflicts for:
fonts-conda-ecosystem -> fonts-conda-forge -> font-ttf-source-code-pro
fonts-conda-forge -> font-ttf-source-code-pro

Package libssh2 conflicts for:
libcurl -> libssh2[version='>=1.10.0|>=1.10.0,<2.0a0|>=1.11.0,<2.0a0|>=1.9.0,<2.0a0']
hdf5 -> libcurl[version='>=8.4.0,<9.0a0'] -> libssh2[version='>=1.10.0,<2.0a0|>=1.10.0|>=1.11.0,<2.0a0|>=1.9.0,<2.0a0']

Package kiwisolver conflicts for:
matplotlib-base -> kiwisolver[version='>=1.0.1|>=1.3.1']
librosa -> matplotlib-base[version='>=3.3.0'] -> kiwisolver[version='>=1.0.1|>=1.3.1']

Package fontconfig conflicts for:
libass -> fontconfig[version='>=2.14.2,<3.0a0']
harfbuzz -> cairo[version='>=1.18.0,<2.0a0'] -> fontconfig[version='>=2.13.1,<2.13.96.0a0|>=2.13.96,<3.0a0|>=2.14.2,<3.0a0|>=2.14.1,<3.0a0|>=2.13.1,<3.0a0']
cairo -> fontconfig[version='>=2.13.1,<2.13.96.0a0|>=2.13.96,<3.0a0|>=2.14.2,<3.0a0|>=2.14.1,<3.0a0|>=2.13.1,<3.0a0']
audioread -> ffmpeg -> fontconfig[version='>=2.13.96,<3.0a0|>=2.14.0,<3.0a0|>=2.14.1,<3.0a0|>=2.14.2,<3.0a0']
ffmpeg -> fontconfig[version='>=2.13.96,<3.0a0|>=2.14.0,<3.0a0|>=2.14.1,<3.0a0|>=2.14.2,<3.0a0']

Package glib-tools conflicts for:
cairo -> glib[version='>=2.69.1,<3.0a0'] -> glib-tools[version='2.70.0|2.70.0|2.70.1|2.70.2|2.70.2|2.70.2|2.70.2|2.70.2|2.72.1|2.74.0|2.74.1|2.74.1|2.76.1|2.76.2|2.76.3|2.76.4|2.78.0|2.78.1|2.78.1|2.78.2|2.78.3|2.78.4|2.78.4|2.78.4|2.78.4|2.80.0',build='hccf11d3_0|hccf11d3_2|hccf11d3_3|ha614eb4_0|ha614eb4_0|ha614eb4_0|h9e231a4_0|h9e231a4_1|h9e231a4_0|h9e231a4_0|h1059232_3|hb9a4d99_1|hb9a4d99_3|hb9a4d99_2|hb9a4d99_0|hb9a4d99_4|h1059232_4|h1059232_0|ha614eb4_0|hb5ab8b9_0|hb5ab8b9_1|hb5ab8b9_0|hb5ab8b9_0|h332123e_0|hccf11d3_4|hccf11d3_1|hccf11d3_0|hccf11d3_0|hccf11d3_1']
harfbuzz -> glib[version='>=2.69.1,<3.0a0'] -> glib-tools[version='2.70.0|2.70.0|2.70.1|2.70.2|2.70.2|2.70.2|2.70.2|2.70.2|2.72.1|2.74.0|2.74.1|2.74.1|2.76.1|2.76.2|2.76.3|2.76.4|2.78.0|2.78.1|2.78.1|2.78.2|2.78.3|2.78.4|2.78.4|2.78.4|2.78.4|2.80.0',build='hccf11d3_0|hccf11d3_2|hccf11d3_3|ha614eb4_0|ha614eb4_0|ha614eb4_0|h9e231a4_0|h9e231a4_1|h9e231a4_0|h9e231a4_0|h1059232_3|hb9a4d99_1|hb9a4d99_3|hb9a4d99_2|hb9a4d99_0|hb9a4d99_4|h1059232_4|h1059232_0|ha614eb4_0|hb5ab8b9_0|hb5ab8b9_1|hb5ab8b9_0|hb5ab8b9_0|h332123e_0|hccf11d3_4|hccf11d3_1|hccf11d3_0|hccf11d3_0|hccf11d3_1']

Package typing_extensions conflicts for:
lazy_loader -> importlib-metadata -> typing_extensions[version='>=3.6.4']
pooch -> platformdirs[version='>=2.5.0'] -> typing_extensions[version='>=4.7.1']
numba -> importlib-metadata -> typing_extensions[version='>=3.6.4']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> typing_extensions[version='3.7.4.*|>=3.6.6|>=3.6.6,<4.6.0|>=3.7.4,<3.8|>=3.7.4']
librosa -> typing_extensions[version='>=4.1.1']
platformdirs -> typing_extensions[version='>=4.7.1']
importlib-metadata -> typing_extensions[version='>=3.6.4']
platformdirs -> typing-extensions[version='>=4.6.3'] -> typing_extensions[version='4.10.0|4.11.0|4.9.0|4.8.0|4.7.1|4.7.0|4.6.3|4.7.1|4.7.1|4.7.1|4.7.1|4.7.1|4.6.3|4.6.3|4.6.3|4.6.3|4.6.2|4.6.1|4.6.0|4.5.0|4.5.0|4.5.0|4.5.0|4.5.0|4.4.0|4.4.0|4.4.0|4.4.0|4.4.0',build='py310hca03da5_0|py39hca03da5_0|pyha770c72_0|py310hca03da5_0|pyha770c72_0|pyha770c72_0|pyha770c72_0|pyha770c72_0|py310hca03da5_0|py311hca03da5_0|py311hca03da5_0|py310hca03da5_0|pyha770c72_0|py312hca03da5_0|py38hca03da5_0|py39hca03da5_0|py38hca03da5_0|py39hca03da5_0|py38hca03da5_0|py39hca03da5_0|py311hca03da5_0|py311hca03da5_0|py38hca03da5_0']

Package re2 conflicts for:
grpcio -> re2[version='>=2022.4.1,<2022.4.2.0a0|>=2022.6.1,<2022.6.2.0a0|>=2023.2.1,<2023.2.2.0a0']
grpcio -> libgrpc==1.62.1=h9c18a4f_0 -> re2[version='>=2023.2.2,<2023.2.3.0a0|>=2023.3.2,<2023.3.3.0a0']
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> re2[version='>=2022.4.1,<2022.4.2.0a0|>=2022.6.1,<2022.6.2.0a0|>=2023.2.1,<2023.2.2.0a0']

Package fonts-conda-forge conflicts for:
cairo -> fonts-conda-ecosystem -> fonts-conda-forge
libass -> fonts-conda-ecosystem -> fonts-conda-forge
ffmpeg -> fonts-conda-ecosystem -> fonts-conda-forge
fonts-conda-ecosystem -> fonts-conda-forge

Package threadpoolctl conflicts for:
scikit-learn -> threadpoolctl[version='>=2.0.0']
librosa -> scikit-learn[version='>=0.20.0'] -> threadpoolctl[version='>=2.0.0']

Package libgettextpo conflicts for:
libflac -> gettext[version='>=0.21.1,<1.0a0'] -> libgettextpo==0.22.5=h8fbad5d_2
gettext -> libgettextpo==0.22.5=h8fbad5d_2
libglib -> gettext[version='>=0.21.1,<1.0a0'] -> libgettextpo==0.22.5=h8fbad5d_2
libidn2 -> gettext[version='>=0.21.1,<1.0a0'] -> libgettextpo==0.22.5=h8fbad5d_2
libgettextpo-devel -> libgettextpo==0.22.5=h8fbad5d_2
gnutls -> gettext[version='>=0.21.1,<1.0a0'] -> libgettextpo==0.22.5=h8fbad5d_2

Package pillow conflicts for:
librosa -> matplotlib-base[version='>=3.3.0'] -> pillow[version='>=6.2.0|>=8']
matplotlib-base -> pillow[version='>=6.2.0|>=8']

Package svt-av1 conflicts for:
ffmpeg -> svt-av1[version='<1.0.0a0|>=1.1.0,<1.1.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0|>=1.3.0,<1.3.1.0a0|>=1.4.0,<1.4.1.0a0|>=1.4.1,<1.4.2.0a0|>=1.5.0,<1.5.1.0a0|>=1.6.0,<1.6.1.0a0|>=1.7.0,<1.7.1.0a0|>=1.8.0,<1.8.1.0a0|>=2.0.0,<2.0.1.0a0']
audioread -> ffmpeg -> svt-av1[version='<1.0.0a0|>=1.1.0,<1.1.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0|>=1.3.0,<1.3.1.0a0|>=1.4.0,<1.4.1.0a0|>=1.4.1,<1.4.2.0a0|>=1.5.0,<1.5.1.0a0|>=1.6.0,<1.6.1.0a0|>=1.7.0,<1.7.1.0a0|>=1.8.0,<1.8.1.0a0|>=2.0.0,<2.0.1.0a0']

Package liblapacke conflicts for:
blas-devel -> liblapacke==3.9.0[build='1_h9886b1c_netlib|0_h2ec9a88_netlib|5_h880f123_netlib|9_openblas|11_osxarm64_openblas|12_osxarm64_accelerate|13_osxarm64_accelerate|13_osxarm64_openblas|14_osxarm64_accelerate|14_osxarm64_openblas|15_osxarm64_accelerate|15_osxarm64_openblas|16_osxarm64_accelerate|18_osxarm64_accelerate|18_osxarm64_openblas|20_osxarm64_accelerate|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|21_osxarm64_openblas|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|17_osxarm64_accelerate|17_osxarm64_openblas|16_osxarm64_openblas|12_osxarm64_openblas|10_openblas|8_openblas|7_openblas']
numpy -> blas=[build=openblas] -> liblapacke==3.9.0[build='3_openblas|5_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_openblas|15_osxarm64_openblas|16_osxarm64_openblas|18_osxarm64_openblas|20_osxarm64_openblas|22_osxarm64_openblas|21_osxarm64_openblas|19_osxarm64_openblas|17_osxarm64_openblas|14_osxarm64_openblas|12_osxarm64_openblas|8_openblas|7_openblas|6_openblas|4_openblas|2_openblas|1_openblas']
blas-devel -> blas==2.106=openblas -> liblapacke==3.9.0[build='3_openblas|5_openblas|6_openblas|4_openblas|2_openblas|1_openblas']
blas -> liblapacke==3.9.0[build='0_h9886b1c_netlib|1_h9886b1c_netlib|1_h2ec9a88_netlib|3_openblas|3_he9612bc_netlib|5_h880f123_netlib|9_openblas|11_osxarm64_openblas|12_osxarm64_accelerate|13_osxarm64_accelerate|13_osxarm64_openblas|14_osxarm64_accelerate|14_osxarm64_openblas|15_osxarm64_accelerate|15_osxarm64_openblas|16_osxarm64_accelerate|18_osxarm64_accelerate|18_osxarm64_openblas|20_osxarm64_accelerate|21_osxarm64_accelerate|22_osxarm64_openblas|22_osxarm64_accelerate|21_osxarm64_openblas|20_osxarm64_openblas|19_osxarm64_accelerate|19_osxarm64_openblas|17_osxarm64_accelerate|17_osxarm64_openblas|16_osxarm64_openblas|12_osxarm64_openblas|10_openblas|8_openblas|7_openblas|6_openblas|5_openblas|4_h880f123_netlib|4_openblas|2_openblas|2_h2ec9a88_netlib|1_openblas|0_h2ec9a88_netlib']
scipy -> blas=[build=openblas] -> liblapacke==3.9.0[build='3_openblas|5_openblas|9_openblas|10_openblas|11_osxarm64_openblas|13_osxarm64_openblas|15_osxarm64_openblas|16_osxarm64_openblas|18_osxarm64_openblas|20_osxarm64_openblas|22_osxarm64_openblas|21_osxarm64_openblas|19_osxarm64_openblas|17_osxarm64_openblas|14_osxarm64_openblas|12_osxarm64_openblas|8_openblas|7_openblas|6_openblas|4_openblas|2_openblas|1_openblas']

Package openmpi conflicts for:
h5py -> openmpi[version='>=4.0.5,<5.0.0a0|>=4.1.0,<5.0a0|>=4.1.1,<5.0a0|>=4.1.2,<5.0a0|>=4.1.4,<5.0a0|>=4.1.5,<5.0a0|>=4.1.6,<5.0a0']
h5py -> mpi4py[version='>=3.0'] -> openmpi[version='>=4.0,<5.0.0a0|>=4.1,<4.2.0a0|>=4.1.3,<5.0a0|>=4.1.4,<4.2.0a0']

Package libogg conflicts for:
libvorbis -> libogg[version='>=1.3.4,<1.4.0a0|>=1.3.5,<1.4.0a0|>=1.3.5,<2.0a0']
pysoundfile -> libsndfile[version='>=1.2'] -> libogg[version='>=1.3.4,<1.4.0a0']
libsndfile -> libflac[version='>=1.4.3,<1.5.0a0'] -> libogg[version='1.3.*|>=1.3.5,<1.4.0a0|>=1.3.5,<2.0a0']
libsndfile -> libogg[version='>=1.3.4,<1.4.0a0']
libflac -> libogg[version='1.3.*|>=1.3.4,<1.4.0a0']

Package libnghttp2 conflicts for:
libcurl -> libnghttp2[version='>=1.41.0,<2.0a0|>=1.43.0,<2.0a0|>=1.47.0,<2.0a0|>=1.51.0,<2.0a0|>=1.52.0,<2.0a0|>=1.58.0,<2.0a0|>=1.57.0|>=1.57.0,<2.0a0|>=1.52.0|>=1.46.0|>=1.46.0,<2.0a0']
hdf5 -> libcurl[version='>=8.4.0,<9.0a0'] -> libnghttp2[version='>=1.41.0,<2.0a0|>=1.43.0,<2.0a0|>=1.46.0,<2.0a0|>=1.46.0|>=1.47.0,<2.0a0|>=1.51.0,<2.0a0|>=1.52.0|>=1.52.0,<2.0a0|>=1.58.0,<2.0a0|>=1.57.0|>=1.57.0,<2.0a0']

Package gettext conflicts for:
gnutls -> gettext[version='>=0.19.8.1|>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0|>=0.21.0,<1.0a0']
libflac -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0']
harfbuzz -> libglib[version='>=2.78.1,<3.0a0'] -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0|>=0.21.0,<1.0a0']
libglib -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0']
cairo -> libglib[version='>=2.78.0,<3.0a0'] -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0|>=0.21.0,<1.0a0']
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> gettext[version='>=0.19.8.1|>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0|>=0.21.0,<1.0a0']
libidn2 -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0']
libsndfile -> libflac[version='>=1.4.3,<1.5.0a0'] -> gettext[version='>=0.19.8.1,<1.0a0|>=0.21.1,<1.0a0']

Package libvpx conflicts for:
audioread -> ffmpeg -> libvpx[version='>=1.10.0,<1.11.0a0|>=1.11.0,<1.12.0a0|>=1.13.0,<1.14.0a0|>=1.13.1,<1.14.0a0|>=1.14.0,<1.15.0a0']
ffmpeg -> libvpx[version='>=1.10.0,<1.11.0a0|>=1.11.0,<1.12.0a0|>=1.13.0,<1.14.0a0|>=1.13.1,<1.14.0a0|>=1.14.0,<1.15.0a0']

Package libasprintf conflicts for:
gnutls -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
libglib -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
libflac -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
gettext -> libasprintf==0.22.5=h8fbad5d_2
libidn2 -> gettext[version='>=0.21.1,<1.0a0'] -> libasprintf==0.22.5=h8fbad5d_2
libasprintf-devel -> libasprintf==0.22.5=h8fbad5d_2

Package pypy3.8 conflicts for:
cffi -> python_abi==3.8[build=*_pypy38_pp73] -> pypy3.8=7.3
cffi -> pypy3.8[version='7.3.11.*|7.3.8.*']

Package matplotlib-base conflicts for:
librosa -> matplotlib[version='>=1.5.0'] -> matplotlib-base[version='>=3.3.2,<3.3.3.0a0|>=3.3.3,<3.3.4.0a0|>=3.3.4,<3.3.5.0a0|>=3.4.1,<3.4.2.0a0|>=3.4.2,<3.4.3.0a0|>=3.4.3,<3.4.4.0a0|>=3.5.0,<3.5.1.0a0|>=3.5.1,<3.5.2.0a0|>=3.5.2,<3.5.3.0a0|>=3.5.3,<3.5.4.0a0|>=3.6.0,<3.6.1.0a0|>=3.6.1,<3.6.2.0a0|>=3.6.2,<3.6.3.0a0|>=3.6.3,<3.6.4.0a0|>=3.7.0,<3.7.1.0a0|>=3.7.1,<3.7.2.0a0|>=3.7.2,<3.7.3.0a0|>=3.7.3,<3.7.4.0a0|>=3.8.0,<3.8.1.0a0|>=3.8.1,<3.8.2.0a0|>=3.8.2,<3.8.3.0a0|>=3.8.3,<3.8.4.0a0']
librosa -> matplotlib-base[version='>=1.5.0|>=3.3.0']

Package libvorbis conflicts for:
pysoundfile -> libsndfile[version='>=1.2'] -> libvorbis[version='>=1.3.7,<1.4.0a0']
libsndfile -> libvorbis[version='>=1.3.7,<1.4.0a0']

Package dav1d conflicts for:
audioread -> ffmpeg -> dav1d[version='>=1.0.0,<1.0.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0']
ffmpeg -> dav1d[version='>=1.0.0,<1.0.1.0a0|>=1.2.0,<1.2.1.0a0|>=1.2.1,<1.2.2.0a0']

Package cairo conflicts for:
ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0'] -> cairo[version='>=1.18.0,<2.0a0']
libass -> harfbuzz[version='>=8.1.1,<9.0a0'] -> cairo[version='>=1.16.0,<2.0a0|>=1.18.0,<2.0a0']
harfbuzz -> cairo[version='>=1.16.0,<2.0.0a0|>=1.16.0,<2.0a0|>=1.18.0,<2.0a0']

Package libunistring conflicts for:
gnutls -> libunistring[version='>=0,<1.0a0']
ffmpeg -> gnutls[version='>=3.6.13,<3.7.0a0'] -> libunistring[version='>=0,<1.0a0']
libidn2 -> libunistring[version='>=0,<1.0a0']

Package fonttools conflicts for:
matplotlib-base -> fonttools[version='>=4.22.0']
librosa -> matplotlib-base[version='>=3.3.0'] -> fonttools[version='>=4.22.0']

Package libflac conflicts for:
libsndfile -> libflac[version='>=1.3.3,<1.4.0a0|>=1.4.1,<1.5.0a0|>=1.4.2,<1.5.0a0|>=1.4.3,<1.5.0a0']
pysoundfile -> libsndfile[version='>=1.2'] -> libflac[version='>=1.3.3,<1.4.0a0|>=1.4.1,<1.5.0a0|>=1.4.2,<1.5.0a0|>=1.4.3,<1.5.0a0']

Package nettle conflicts for:
gnutls -> nettle[version='>=3.4.1|>=3.7,<3.8.0a0|>=3.8,<3.9.0a0|>=3.8.1,<3.9.0a0|>=3.9.1,<3.10.0a0|>=3.6,<3.7.0a0|>=3.7.3,<3.8.0a0']
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> nettle[version='>=3.4.1|>=3.7,<3.8.0a0|>=3.8,<3.9.0a0|>=3.8.1,<3.9.0a0|>=3.9.1,<3.10.0a0|>=3.6,<3.7.0a0|>=3.7.3,<3.8.0a0']

Package libev conflicts for:
libnghttp2 -> libev[version='>=4.11|>=4.33,<4.34.0a0|>=4.33,<5.0a0']
libcurl -> libnghttp2[version='>=1.58.0,<2.0a0'] -> libev[version='>=4.11|>=4.33,<4.34.0a0|>=4.33,<5.0a0']

Package libopus conflicts for:
pysoundfile -> libsndfile[version='>=1.2'] -> libopus[version='>=1.3.1,<2.0a0']
audioread -> ffmpeg -> libopus[version='>=1.3,<2.0a0|>=1.3.1,<2.0a0']
ffmpeg -> libopus[version='>=1.3,<2.0a0|>=1.3.1,<2.0a0']
libsndfile -> libopus[version='>=1.3.1,<2.0a0']

Package appdirs conflicts for:
scipy -> pooch -> appdirs[version='>=1.3.0']
pooch -> appdirs[version='>=1.3.0']
librosa -> pooch[version='>=1.0'] -> appdirs[version='>=1.3.0']

Package python-dateutil conflicts for:
matplotlib-base -> python-dateutil[version='>=2.1|>=2.7']
librosa -> matplotlib-base[version='>=3.3.0'] -> python-dateutil[version='>=2.1|>=2.7']

Package gtest conflicts for:
grpcio -> abseil-cpp[version='>=20230802.0,<20230802.1.0a0'] -> gtest[version='>=1.14.0,<1.14.1.0a0']
libprotobuf -> gtest[version='>=1.14.0,<1.14.1.0a0']
protobuf -> libprotobuf[version='>=4.23.4,<4.23.5.0a0'] -> gtest[version='>=1.14.0,<1.14.1.0a0']

Package pypy3.7 conflicts for:
python_abi -> pypy3.7=7.3
pysoundfile -> cffi -> pypy3.7[version='7.3.3.*|7.3.4.*|7.3.5.*|7.3.7.*']
cffi -> pypy3.7[version='7.3.3.*|7.3.4.*|7.3.5.*|7.3.7.*']
cffi -> python_abi==3.7[build=*_pypy37_pp73] -> pypy3.7=7.3

Package platformdirs conflicts for:
pooch -> platformdirs[version='>=2.5.0']
scipy -> pooch -> platformdirs[version='>=2.5.0']
librosa -> pooch[version='>=1.0'] -> platformdirs[version='>=2.5.0']

Package fonts-anaconda conflicts for:
libass -> fonts-conda-ecosystem -> fonts-anaconda
cairo -> fonts-conda-ecosystem -> fonts-anaconda
ffmpeg -> fonts-conda-ecosystem -> fonts-anaconda
fonts-conda-ecosystem -> fonts-anaconda

Package chardet conflicts for:
requests -> chardet[version='>=3.0.2,<4|>=3.0.2,<5']
pooch -> requests[version='>=2.19.0'] -> chardet[version='>=3.0.2,<4|>=3.0.2,<5']

Package protobuf conflicts for:
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> protobuf[version='>=3.19.6|>=3.20.3,<5,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5|>=3.9.2|>=3.6.1|>=3.6.0|>=3.9.2,<3.20']
tensorflow-deps -> protobuf[version='>=3.19.1,<3.20']

Package pypy3.9 conflicts for:
cffi -> python_abi==3.9[build=*_pypy39_pp73] -> pypy3.9=7.3
cffi -> pypy3.9[version='7.3.11.*|7.3.15.*|7.3.8.*']

Package joblib conflicts for:
scikit-learn -> joblib[version='>=0.11|>=1.0.0|>=1.1.1|>=1.2.0']
librosa -> scikit-learn[version='>=0.20.0'] -> joblib[version='>=0.11|>=1.0.0|>=1.1.1|>=1.2.0']
librosa -> joblib[version='>=0.12.0|>=0.14.0|>=0.7.0']

Package mpg123 conflicts for:
pysoundfile -> libsndfile[version='>=1.2'] -> mpg123[version='>=1.30.2,<1.31.0a0|>=1.31.1,<1.32.0a0|>=1.31.3,<1.32.0a0|>=1.32.1,<1.33.0a0']
libsndfile -> mpg123[version='>=1.30.2,<1.31.0a0|>=1.31.1,<1.32.0a0|>=1.31.3,<1.32.0a0|>=1.32.1,<1.33.0a0']

Package c-ares conflicts for:
grpcio -> libgrpc==1.62.1=h9c18a4f_0 -> c-ares[version='>=1.19.0,<2.0a0|>=1.20.1,<2.0a0|>=1.21.0,<2.0a0|>=1.22.1,<2.0a0|>=1.25.0,<2.0a0|>=1.26.0,<2.0a0|>=1.27.0,<2.0a0']
grpcio -> c-ares[version='>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.19.1,<2.0a0']
libnghttp2 -> c-ares[version='>=1.16.1,<2.0a0|>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.20.1,<2.0a0|>=1.21.0,<2.0a0|>=1.23.0,<2.0a0|>=1.7.5|>=1.19.1,<2.0a0|>=1.19.0,<2.0a0']
libcurl -> libnghttp2[version='>=1.58.0,<2.0a0'] -> c-ares[version='>=1.16.1,<2.0a0|>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.20.1,<2.0a0|>=1.21.0,<2.0a0|>=1.23.0,<2.0a0|>=1.19.1,<2.0a0|>=1.7.5|>=1.19.0,<2.0a0']
tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> c-ares[version='>=1.17.1,<2.0a0|>=1.17.2,<2.0a0|>=1.18.1,<2.0a0|>=1.19.1,<2.0a0']

Package libidn2 conflicts for:
gnutls -> libidn2[version='>=2,<3.0a0']
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> libidn2[version='>=2,<3.0a0']

Package harfbuzz conflicts for:
audioread -> ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0']
ffmpeg -> harfbuzz[version='>=8.3.0,<9.0a0']
libass -> harfbuzz[version='>=7.2.0,<8.0a0|>=8.1.1,<9.0a0']
ffmpeg -> libass[version='>=0.17.1,<0.17.2.0a0'] -> harfbuzz[version='>=7.2.0,<8.0a0|>=8.1.1,<9.0a0']

Package openblas conflicts for:
blas-devel -> openblas[version='0.3.18.*|0.3.20.*|0.3.21.*|0.3.23.*|0.3.24.*|0.3.25.*|0.3.26.*|0.3.27.*']
blas -> blas-devel==3.9.0=22_osxarm64_openblas -> openblas[version='0.3.18.*|0.3.20.*|0.3.21.*|0.3.23.*|0.3.24.*|0.3.25.*|0.3.26.*|0.3.27.*']

Package pcre conflicts for:
libglib -> pcre[version='>=8.44,<9.0a0|>=8.45,<9.0a0']
harfbuzz -> libglib[version='>=2.72.1,<3.0a0'] -> pcre[version='>=8.44,<9.0a0|>=8.45,<9.0a0']
cairo -> libglib[version='>=2.72.1,<3.0a0'] -> pcre[version='>=8.44,<9.0a0|>=8.45,<9.0a0']

Package libcurl conflicts for:
h5py -> hdf5[version='>=1.14.3,<1.14.4.0a0'] -> libcurl[version='>=7.71.1,<8.0a0|>=7.77.0,<9.0a0|>=7.79.1,<9.0a0|>=7.80.0,<9.0a0|>=7.81.0,<9.0a0|>=7.83.1,<9.0a0|>=7.87.0,<9.0a0|>=8.1.2,<9.0a0|>=8.2.1,<9.0a0|>=8.4.0,<9.0a0|>=7.88.1,<9.0a0|>=7.88.1,<8.0a0|>=7.82.0,<8.0a0|>=7.71.1,<9.0a0']
tensorflow -> tensorflow-base==2.15.0=cpu_py311he034567_2 -> libcurl[version='>=7.76.0,<9.0a0|>=7.76.1,<9.0a0|>=7.78.0,<9.0a0|>=7.79.1,<9.0a0|>=7.80.0,<9.0a0|>=7.81.0,<9.0a0|>=7.83.0,<9.0a0|>=7.83.1,<9.0a0|>=7.87.0,<9.0a0|>=7.88.1,<9.0a0|>=8.1.2,<9.0a0|>=8.3.0,<9.0a0|>=8.4.0,<9.0a0|>=8.5.0,<9.0a0|>=7.88.1,<8.0a0|>=7.86.0,<8.0a0']
hdf5 -> libcurl[version='>=7.71.1,<8.0a0|>=7.71.1,<9.0a0|>=7.76.0,<9.0a0|>=7.77.0,<9.0a0|>=7.79.1,<9.0a0|>=7.80.0,<9.0a0|>=7.81.0,<9.0a0|>=7.83.1,<9.0a0|>=7.87.0,<9.0a0|>=8.1.2,<9.0a0|>=8.2.1,<9.0a0|>=8.4.0,<9.0a0|>=7.88.1,<9.0a0|>=7.88.1,<8.0a0|>=7.82.0,<8.0a0']

Package xorg-libxdmcp conflicts for:
pillow -> libxcb[version='>=1.15,<1.16.0a0'] -> xorg-libxdmcp
libxcb -> xorg-libxdmcp

Package tbb conflicts for:
librosa -> numba[version='>=0.51.0'] -> tbb[version='>=2021.3.0|>=2021.5.0|>=2021.8.0']
numba -> tbb[version='>=2021.3.0|>=2021.5.0|>=2021.8.0']
ffmpeg -> libopenvino[version='>=2024.0.0,<2024.0.1.0a0'] -> tbb[version='>=2021.11.0|>=2021.5.0']

Package pooch conflicts for:
librosa -> pooch[version='>=1.0|>=1.0,<1.7']
scipy -> pooch
scikit-learn -> scipy -> pooch
librosa -> scipy[version='>=1.2.0'] -> pooch

Package p11-kit conflicts for:
gnutls -> p11-kit[version='>=0.23.21,<0.24.0a0|>=0.24.1,<0.25.0a0']
ffmpeg -> gnutls[version='>=3.7.9,<3.8.0a0'] -> p11-kit[version='>=0.23.21,<0.24.0a0|>=0.24.1,<0.25.0a0']

Package lerc conflicts for:
lcms2 -> libtiff[version='>=4.6.0,<4.7.0a0'] -> lerc[version='>=2.2.1,<3.0a0|>=3.0,<4.0a0|>=4.0.0,<5.0a0']
openjpeg -> libtiff[version='>=4.6.0,<4.7.0a0'] -> lerc[version='>=2.2.1,<3.0a0|>=3.0,<4.0a0|>=4.0.0,<5.0a0']
libtiff -> lerc[version='>=2.2.1,<3.0a0|>=3.0,<4.0a0|>=4.0.0,<5.0a0']
pillow -> libtiff[version='>=4.6.0,<4.7.0a0'] -> lerc[version='>=2.2.1,<3.0a0|>=3.0,<4.0a0|>=4.0.0,<5.0a0']

Package libllvm14 conflicts for:
numba -> libllvm14[version='>=14.0.6,<14.1.0a0']
llvmlite -> libllvm14[version='>=14.0.6,<14.1.0a0']
librosa -> numba[version='>=0.51.0'] -> libllvm14[version='>=14.0.6,<14.1.0a0']

Package mpi4py conflicts for:
tensorflow-deps -> h5py[version='>=3.6.0,<3.7'] -> mpi4py[version='>=3.0']
h5py -> mpi4py[version='>=3.0']

Package x265 conflicts for:
ffmpeg -> x265[version='>=3.5,<3.6.0a0']
audioread -> ffmpeg -> x265[version='>=3.5,<3.6.0a0']

Package font-ttf-dejavu-sans-mono conflicts for:
fonts-conda-forge -> font-ttf-dejavu-sans-mono
fonts-conda-ecosystem -> fonts-conda-forge -> font-ttf-dejavu-sans-mono

Package pthread-stubs conflicts for:
libxcb -> pthread-stubs
pillow -> libxcb[version='>=1.15,<1.16.0a0'] -> pthread-stubs

Package libdeflate conflicts for:
lcms2 -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']
openjpeg -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']
pillow -> libtiff[version='>=4.6.0,<4.7.0a0'] -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']
libtiff -> libdeflate[version='>=1.10,<1.11.0a0|>=1.12,<1.13.0a0|>=1.13,<1.14.0a0|>=1.14,<1.15.0a0|>=1.16,<1.17.0a0|>=1.17,<1.18.0a0|>=1.18,<1.19.0a0|>=1.19,<1.20.0a0|>=1.20,<1.21.0a0|>=1.8,<1.9.0a0|>=1.7,<1.8.0a0']

Package font-ttf-ubuntu conflicts for:
fonts-conda-forge -> font-ttf-ubuntu
fonts-conda-ecosystem -> fonts-conda-forge -> font-ttf-ubuntu

Package pypy3.6 conflicts for:
cffi -> python_abi==3.6[build=*_pypy36_pp73] -> pypy3.6=7.3
cffi -> pypy3.6[version='7.3.0.*|7.3.1.*|7.3.2.*|7.3.3.*']

Package importlib-metadata conflicts for:
lazy_loader -> importlib-metadata
numba -> importlib_metadata -> importlib-metadata[version='>=1.1.3,<1.1.4.0a0|>=1.5.0,<1.5.1.0a0|>=1.5.2,<1.5.3.0a0|>=1.6.0,<1.6.1.0a0|>=1.6.1,<1.6.2.0a0|>=1.7.0,<1.7.1.0a0|>=2.0.0,<2.0.1.0a0|>=3.0.0,<3.0.1.0a0|>=3.1.0,<3.1.1.0a0|>=3.1.1,<3.1.2.0a0|>=3.10.0,<3.10.1.0a0|>=3.10.1,<3.10.2.0a0|>=4.0.1,<4.0.2.0a0|>=4.10.0,<4.10.1.0a0|>=4.10.1,<4.10.2.0a0|>=4.11.0,<4.11.1.0a0|>=4.11.1,<4.11.2.0a0|>=4.11.2,<4.11.3.0a0|>=4.11.3,<4.11.4.0a0|>=4.11.4,<4.11.5.0a0|>=4.13.0,<4.13.1.0a0|>=5.0.0,<5.0.1.0a0|>=5.1.0,<5.1.1.0a0|>=5.2.0,<5.2.1.0a0|>=6.0.0,<6.0.1.0a0|>=6.1.0,<6.1.1.0a0|>=6.10.0,<6.10.1.0a0|>=7.0.0,<7.0.1.0a0|>=7.0.1,<7.0.2.0a0|>=7.0.2,<7.0.3.0a0|>=7.1.0,<7.1.1.0a0|>=6.9.0,<6.9.1.0a0|>=6.8.0,<6.8.1.0a0|>=6.7.0,<6.7.1.0a0|>=6.6.0,<6.6.1.0a0|>=6.5.1,<6.5.2.0a0|>=6.5.0,<6.5.1.0a0|>=6.4.1,<6.4.2.0a0|>=6.4.0,<6.4.1.0a0|>=6.3.0,<6.3.1.0a0|>=6.2.1,<6.2.2.0a0|>=6.2.0,<6.2.1.0a0|>=4.9.0,<4.9.1.0a0|>=4.8.3,<4.8.4.0a0|>=4.8.2,<4.8.3.0a0|>=4.8.1,<4.8.2.0a0|>=4.8.0,<4.8.1.0a0|>=4.7.1,<4.7.2.0a0|>=4.7.0,<4.7.1.0a0|>=4.6.4,<4.6.5.0a0|>=4.6.3,<4.6.4.0a0|>=4.6.2,<4.6.3.0a0|>=4.6.1,<4.6.2.0a0|>=4.6.0,<4.6.1.0a0|>=4.5.0,<4.5.1.0a0|>=4.4.0,<4.4.1.0a0|>=4.3.1,<4.3.2.0a0|>=4.3.0,<4.3.1.0a0|>=4.2.0,<4.2.1.0a0|>=3.9.1,<3.9.2.0a0|>=3.9.0,<3.9.1.0a0|>=3.8.1,<3.8.2.0a0|>=3.8.0,<3.8.1.0a0|>=3.7.3,<3.7.4.0a0|>=3.7.2,<3.7.3.0a0|>=3.7.0,<3.7.1.0a0|>=3.6.0,<3.6.1.0a0|>=3.4.0,<3.4.1.0a0|>=3.3.0,<3.3.1.0a0']
librosa -> lazy_loader[version='>=0.1'] -> importlib-metadata
numba -> importlib-metadata

Package libedit conflicts for:
krb5 -> libedit[version='>=3.1.20191231,<3.2.0a0|>=3.1.20191231,<4.0a0|>=3.1.20221030,<3.2.0a0|>=3.1.20221030,<4.0a0|>=3.1.20210216,<3.2.0a0|>=3.1.20210216,<4.0a0']
libcurl -> krb5[version='>=1.21.2,<1.22.0a0'] -> libedit[version='>=3.1.20191231,<3.2.0a0|>=3.1.20191231,<4.0a0|>=3.1.20221030,<3.2.0a0|>=3.1.20221030,<4.0a0|>=3.1.20210216,<3.2.0a0|>=3.1.20210216,<4.0a0']

Package libass conflicts for:
audioread -> ffmpeg -> libass[version='>=0.17.1,<0.17.2.0a0']
ffmpeg -> libass[version='>=0.17.1,<0.17.2.0a0']

Package libwebp conflicts for:
matplotlib-base -> pillow[version='>=8'] -> libwebp[version='>=0.3.0|>=1.2.0,<1.3.0a0|>=1.3.2,<2.0a0']
pillow -> libwebp[version='>=0.3.0|>=1.2.0,<1.3.0a0|>=1.3.2,<2.0a0']

Package pyopenssl conflicts for:
urllib3 -> pyopenssl[version='>=0.14']
requests -> urllib3[version='>=1.21.1,<3'] -> pyopenssl[version='>=0.14']

Package pixman conflicts for:
cairo -> pixman[version='>=0.40.0,<1.0a0|>=0.42.2,<1.0a0']
harfbuzz -> cairo[version='>=1.18.0,<2.0a0'] -> pixman[version='>=0.40.0,<1.0a0|>=0.42.2,<1.0a0']The following specifications were found to be incompatible with your system:

  - feature:/osx-arm64::__osx==13.6.3=0
  - feature:/osx-arm64::__unix==0=0
  - feature:|@/osx-arm64::__osx==13.6.3=0
  - feature:|@/osx-arm64::__unix==0=0
  - aom -> __osx[version='>=10.9']
  - audioread -> ffmpeg -> __osx[version='>=10.9']
  - cairo -> __osx[version='>=10.9']
  - ffmpeg -> __osx[version='>=10.9']
  - gettext -> ncurses[version='>=6.4,<7.0a0'] -> __osx[version='>=10.9']
  - gmp -> __osx[version='>=10.9']
  - gnutls -> __osx[version='>=10.9']
  - grpcio -> __osx[version='>=10.9']
  - h5py -> hdf5[version='>=1.14.3,<1.14.4.0a0'] -> __osx[version='>=10.9']
  - harfbuzz -> __osx[version='>=10.9']
  - hdf5 -> __osx[version='>=10.9']
  - libass -> harfbuzz[version='>=8.1.1,<9.0a0'] -> __osx[version='>=10.9']
  - libcurl -> libnghttp2[version='>=1.58.0,<2.0a0'] -> __osx[version='>=10.9']
  - libedit -> ncurses[version='>=6.2,<7.0.0a0'] -> __osx[version='>=10.9']
  - libglib -> __osx[version='>=10.9']
  - libnghttp2 -> __osx[version='>=10.9']
  - libprotobuf -> __osx[version='>=10.9']
  - librosa -> matplotlib-base[version='>=3.3.0'] -> __osx[version='>=10.9']
  - matplotlib-base -> __osx[version='>=10.9']
  - msgpack-python -> __osx[version='>=10.9']
  - ncurses -> __osx[version='>=10.9']
  - nettle -> gmp[version='>=6.2.1,<7.0a0'] -> __osx[version='>=10.9']
  - numba -> __osx[version='>=10.9']
  - numpy -> __osx[version='>=10.9']
  - openh264 -> __osx[version='>=10.9']
  - protobuf -> __osx[version='>=10.9']
  - pysocks -> __unix
  - pysocks -> __win
  - pysoundfile -> numpy -> __osx[version='>=10.9']
  - python=3.8 -> ncurses[version='>=6.4,<7.0a0'] -> __osx[version='>=10.9']
  - readline -> ncurses[version='>=6.3,<7.0a0'] -> __osx[version='>=10.9']
  - scikit-learn -> __osx[version='>=10.9']
  - scipy -> __osx[version='>=10.9']
  - soxr-python -> numpy[version='>=1.23.5,<2.0a0'] -> __osx[version='>=10.9']
  - svt-av1 -> __osx[version='>=10.9']
  - tensorflow-deps -> grpcio[version='>=1.37.0,<2.0'] -> __osx[version='>=10.9']
  - urllib3 -> pysocks[version='>=1.5.6,<2.0,!=1.5.7'] -> __unix

Your installed version is: 13.6.3


(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python -c "import tensorflow as tf; print(tf.__version__)"

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'tensorflow'
(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) cd Downloads

bash Miniforge3-MacOSX-arm64.sh

brew install miniforge
cd: no such file or directory: Downloads
bash: Miniforge3-MacOSX-arm64.sh: No such file or directory
==> Auto-updating Homebrew...
Adjust how often this is run with HOMEBREW_AUTO_UPDATE_SECS or disable with
HOMEBREW_NO_AUTO_UPDATE. Hide these hints with HOMEBREW_NO_ENV_HINTS (see `man brew`).
ç^C
(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda create -n tensorflow python=3.8

Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 23.3.1
  latest version: 24.3.0

Please update conda by running

    $ conda update -n base -c conda-forge conda

Or to minimize the number of packages updated during conda update use

     conda install conda=24.3.0



## Package Plan ##

  environment location: /Users/deangladish/miniforge3/envs/tensorflow

  added / updated specs:
    - python=3.8


The following NEW packages will be INSTALLED:

  bzip2              conda-forge/osx-arm64::bzip2-1.0.8-h93a5062_5
  ca-certificates    conda-forge/osx-arm64::ca-certificates-2024.2.2-hf0a4a13_0
  libffi             conda-forge/osx-arm64::libffi-3.4.2-h3422bc3_5
  libsqlite          conda-forge/osx-arm64::libsqlite-3.45.2-h091b4b1_0
  libzlib            conda-forge/osx-arm64::libzlib-1.2.13-h53f4e23_5
  ncurses            conda-forge/osx-arm64::ncurses-6.4.20240210-h078ce10_0
  openssl            conda-forge/osx-arm64::openssl-3.2.1-h0d3ecfb_1
  pip                conda-forge/noarch::pip-24.0-pyhd8ed1ab_0
  python             conda-forge/osx-arm64::python-3.8.19-h2469fbe_0_cpython
  readline           conda-forge/osx-arm64::readline-8.2-h92ec313_1
  setuptools         conda-forge/noarch::setuptools-69.2.0-pyhd8ed1ab_0
  tk                 conda-forge/osx-arm64::tk-8.6.13-h5083fa2_1
  wheel              conda-forge/noarch::wheel-0.43.0-pyhd8ed1ab_1
  xz                 conda-forge/osx-arm64::xz-5.2.6-h57fd34a_0


Proceed ([y]/n)? y


Downloading and Extracting Packages

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate tensorflow
#
# To deactivate an active environment, use
#
#     $ conda deactivate

(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) source activate tensorflow

source: no such file or directory: activate
(tensyflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda activate tensorflow
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda install -c apple tensorflow-deps
pip install tensorflow-macos==2.9.0
pip install tensorflow-metal==0.5.0
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Solving environment: failed with repodata from current_repodata.json, will retry with next repodata source.
Collecting package metadata (repodata.json): \ WARNING conda.models.version:get_matcher(546): Using .* with relational operator is superfluous and deprecated and will be removed in a future version of conda. Your spec was 1.8.0.*, but conda is ignoring the .* and treating it as 1.8.0
WARNING conda.models.version:get_matcher(546): Using .* with relational operator is superfluous and deprecated and will be removed in a future version of conda. Your spec was 1.9.0.*, but conda is ignoring the .* and treating it as 1.9.0
done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 23.3.1
  latest version: 24.3.0

Please update conda by running

    $ conda update -n base -c conda-forge conda

Or to minimize the number of packages updated during conda update use

     conda install conda=24.3.0



## Package Plan ##

  environment location: /Users/deangladish/miniforge3/envs/tensorflow

  added / updated specs:
    - tensorflow-deps


The following NEW packages will be INSTALLED:

  c-ares             conda-forge/osx-arm64::c-ares-1.28.1-h93a5062_0
  cached-property    conda-forge/noarch::cached-property-1.5.2-hd8ed1ab_1
  cached_property    conda-forge/noarch::cached_property-1.5.2-pyha770c72_1
  grpcio             conda-forge/osx-arm64::grpcio-1.46.3-py38h1ef021a_0
  h5py               conda-forge/osx-arm64::h5py-3.6.0-nompi_py38hacf61ce_100
  hdf5               conda-forge/osx-arm64::hdf5-1.12.1-nompi_hd9dbc9e_104
  krb5               conda-forge/osx-arm64::krb5-1.21.2-h92f50d5_0
  libblas            conda-forge/osx-arm64::libblas-3.9.0-22_osxarm64_openblas
  libcblas           conda-forge/osx-arm64::libcblas-3.9.0-22_osxarm64_openblas
  libcurl            conda-forge/osx-arm64::libcurl-8.7.1-h2d989ff_0
  libcxx             conda-forge/osx-arm64::libcxx-16.0.6-h4653b0c_0
  libedit            conda-forge/osx-arm64::libedit-3.1.20191231-hc8eb9b7_2
  libev              conda-forge/osx-arm64::libev-4.33-h93a5062_2
  libgfortran        conda-forge/osx-arm64::libgfortran-5.0.0-13_2_0_hd922786_3
  libgfortran5       conda-forge/osx-arm64::libgfortran5-13.2.0-hf226fd6_3
  liblapack          conda-forge/osx-arm64::liblapack-3.9.0-22_osxarm64_openblas
  libnghttp2         conda-forge/osx-arm64::libnghttp2-1.58.0-ha4dd798_1
  libopenblas        conda-forge/osx-arm64::libopenblas-0.3.27-openmp_h6c19121_0
  libprotobuf        conda-forge/osx-arm64::libprotobuf-3.19.6-hb5ab8b9_0
  libssh2            conda-forge/osx-arm64::libssh2-1.11.0-h7a5bd25_0
  llvm-openmp        conda-forge/osx-arm64::llvm-openmp-18.1.2-hcd81f8e_0
  numpy              conda-forge/osx-arm64::numpy-1.23.2-py38h579d673_0
  protobuf           conda-forge/osx-arm64::protobuf-3.19.6-py38h2b1e499_0
  python_abi         conda-forge/osx-arm64::python_abi-3.8-4_cp38
  six                conda-forge/noarch::six-1.16.0-pyh6c4a22f_0
  tensorflow-deps    apple/osx-arm64::tensorflow-deps-2.10.0-0
  zlib               conda-forge/osx-arm64::zlib-1.2.13-h53f4e23_5
  zstd               conda-forge/osx-arm64::zstd-1.5.5-h4f39d0f_0


Proceed ([y]/n)? y


Downloading and Extracting Packages

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
Defaulting to user installation because normal site-packages is not writeable
Collecting tensorflow-macos==2.9.0
  Using cached tensorflow_macos-2.9.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (2.9 kB)
Requirement already satisfied: absl-py>=1.0.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (1.4.0)
Requirement already satisfied: astunparse>=1.6.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (1.6.3)
Collecting flatbuffers<2,>=1.12 (from tensorflow-macos==2.9.0)
  Using cached flatbuffers-1.12-py2.py3-none-any.whl.metadata (872 bytes)
Collecting gast<=0.4.0,>=0.2.1 (from tensorflow-macos==2.9.0)
  Using cached gast-0.4.0-py3-none-any.whl.metadata (1.1 kB)
Requirement already satisfied: google-pasta>=0.1.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (0.2.0)
Requirement already satisfied: h5py>=2.9.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (3.10.0)
Collecting keras-preprocessing>=1.1.1 (from tensorflow-macos==2.9.0)
  Using cached Keras_Preprocessing-1.1.2-py2.py3-none-any.whl.metadata (1.9 kB)
Requirement already satisfied: libclang>=13.0.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (18.1.1)
Requirement already satisfied: numpy>=1.20 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (1.26.4)
Requirement already satisfied: opt-einsum>=2.3.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (3.3.0)
Requirement already satisfied: packaging in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow-macos==2.9.0) (21.3)
Requirement already satisfied: protobuf>=3.9.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (4.25.3)
Requirement already satisfied: setuptools in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow-macos==2.9.0) (63.2.0)
Requirement already satisfied: six>=1.12.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow-macos==2.9.0) (2.0.1)
Requirement already satisfied: typing-extensions>=3.6.6 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow-macos==2.9.0) (4.5.0)
Requirement already satisfied: wrapt>=1.11.0 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (1.16.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-macos==2.9.0) (1.62.1)
Collecting tensorboard<2.10,>=2.9 (from tensorflow-macos==2.9.0)
  Using cached tensorboard-2.9.1-py3-none-any.whl.metadata (1.9 kB)
Collecting tensorflow-estimator<2.10.0,>=2.9.0rc0 (from tensorflow-macos==2.9.0)
  Using cached tensorflow_estimator-2.9.0-py2.py3-none-any.whl.metadata (1.3 kB)
Collecting keras<2.10.0,>=2.9.0rc0 (from tensorflow-macos==2.9.0)
  Using cached keras-2.9.0-py2.py3-none-any.whl.metadata (1.3 kB)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from astunparse>=1.6.0->tensorflow-macos==2.9.0) (0.37.1)
Collecting google-auth<3,>=1.6.3 (from tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached google_auth-2.29.0-py2.py3-none-any.whl.metadata (4.7 kB)
Collecting google-auth-oauthlib<0.5,>=0.4.1 (from tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached google_auth_oauthlib-0.4.6-py2.py3-none-any.whl.metadata (2.7 kB)
Requirement already satisfied: markdown>=2.6.8 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (3.6)
Collecting protobuf>=3.9.2 (from tensorflow-macos==2.9.0)
  Using cached protobuf-3.19.6-py2.py3-none-any.whl.metadata (828 bytes)
Requirement already satisfied: requests<3,>=2.21.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (2.28.1)
Collecting tensorboard-data-server<0.7.0,>=0.6.0 (from tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached tensorboard_data_server-0.6.1-py3-none-any.whl.metadata (1.1 kB)
Collecting tensorboard-plugin-wit>=1.6.0 (from tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached tensorboard_plugin_wit-1.8.1-py3-none-any.whl.metadata (873 bytes)
Requirement already satisfied: werkzeug>=1.0.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (3.0.2)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from packaging->tensorflow-macos==2.9.0) (3.1.1)
Collecting cachetools<6.0,>=2.0.0 (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached cachetools-5.3.3-py3-none-any.whl.metadata (5.3 kB)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (0.2.8)
Collecting rsa<5,>=3.1.4 (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached rsa-4.9-py3-none-any.whl.metadata (4.2 kB)
Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached requests_oauthlib-2.0.0-py2.py3-none-any.whl.metadata (11 kB)
Requirement already satisfied: charset-normalizer<3,>=2 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (2.1.1)
Requirement already satisfied: idna<4,>=2.5 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (3.4)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (1.26.12)
Requirement already satisfied: certifi>=2017.4.17 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (2022.9.24)
Requirement already satisfied: MarkupSafe>=2.1.1 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (2.1.1)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0) (0.4.8)
Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow-macos==2.9.0)
  Using cached oauthlib-3.2.2-py3-none-any.whl.metadata (7.5 kB)
Using cached tensorflow_macos-2.9.0-cp310-cp310-macosx_11_0_arm64.whl (200.6 MB)
Using cached flatbuffers-1.12-py2.py3-none-any.whl (15 kB)
Using cached gast-0.4.0-py3-none-any.whl (9.8 kB)
Using cached keras-2.9.0-py2.py3-none-any.whl (1.6 MB)
Using cached Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42 kB)
Using cached tensorboard-2.9.1-py3-none-any.whl (5.8 MB)
Using cached protobuf-3.19.6-py2.py3-none-any.whl (162 kB)
Using cached tensorflow_estimator-2.9.0-py2.py3-none-any.whl (438 kB)
Using cached google_auth-2.29.0-py2.py3-none-any.whl (189 kB)
Using cached google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
Using cached tensorboard_data_server-0.6.1-py3-none-any.whl (2.4 kB)
Using cached tensorboard_plugin_wit-1.8.1-py3-none-any.whl (781 kB)
Using cached cachetools-5.3.3-py3-none-any.whl (9.3 kB)
Using cached requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)
Using cached rsa-4.9-py3-none-any.whl (34 kB)
Using cached oauthlib-3.2.2-py3-none-any.whl (151 kB)
Installing collected packages: tensorboard-plugin-wit, keras, flatbuffers, tensorflow-estimator, tensorboard-data-server, rsa, protobuf, oauthlib, keras-preprocessing, gast, cachetools, requests-oauthlib, google-auth, google-auth-oauthlib, tensorboard, tensorflow-macos
  Attempting uninstall: keras
    Found existing installation: keras 3.1.1
    Uninstalling keras-3.1.1:
      Successfully uninstalled keras-3.1.1
  Attempting uninstall: flatbuffers
    Found existing installation: flatbuffers 24.3.25
    Uninstalling flatbuffers-24.3.25:
      Successfully uninstalled flatbuffers-24.3.25
  Attempting uninstall: tensorboard-data-server
    Found existing installation: tensorboard-data-server 0.7.2
    Uninstalling tensorboard-data-server-0.7.2:
      Successfully uninstalled tensorboard-data-server-0.7.2
  WARNING: The scripts pyrsa-decrypt, pyrsa-encrypt, pyrsa-keygen, pyrsa-priv2pub, pyrsa-sign and pyrsa-verify are installed in '/Users/deangladish/Library/Python/3.10/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  Attempting uninstall: protobuf
    Found existing installation: protobuf 4.25.3
    Uninstalling protobuf-4.25.3:
      Successfully uninstalled protobuf-4.25.3
  Attempting uninstall: gast
    Found existing installation: gast 0.5.4
    Uninstalling gast-0.5.4:
      Successfully uninstalled gast-0.5.4
  WARNING: The script google-oauthlib-tool is installed in '/Users/deangladish/Library/Python/3.10/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  Attempting uninstall: tensorboard
    Found existing installation: tensorboard 2.16.2
    Uninstalling tensorboard-2.16.2:
      Successfully uninstalled tensorboard-2.16.2
  WARNING: The script tensorboard is installed in '/Users/deangladish/Library/Python/3.10/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts estimator_ckpt_converter, import_pb_to_tensorboard, saved_model_cli, tensorboard, tf_upgrade_v2, tflite_convert, toco and toco_from_protos are installed in '/Users/deangladish/Library/Python/3.10/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
Successfully installed cachetools-5.3.3 flatbuffers-1.12 gast-0.4.0 google-auth-2.29.0 google-auth-oauthlib-0.4.6 keras-2.9.0 keras-preprocessing-1.1.2 oauthlib-3.2.2 protobuf-3.19.6 requests-oauthlib-2.0.0 rsa-4.9 tensorboard-2.9.1 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 tensorflow-estimator-2.9.0 tensorflow-macos-2.9.0
Defaulting to user installation because normal site-packages is not writeable
Collecting tensorflow-metal==0.5.0
  Downloading tensorflow_metal-0.5.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (776 bytes)
Requirement already satisfied: wheel~=0.35 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from tensorflow-metal==0.5.0) (0.37.1)
Collecting six~=1.15.0 (from tensorflow-metal==0.5.0)
  Downloading six-1.15.0-py2.py3-none-any.whl.metadata (1.8 kB)
Downloading tensorflow_metal-0.5.0-cp310-cp310-macosx_11_0_arm64.whl (1.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 5.6 MB/s eta 0:00:00
Downloading six-1.15.0-py2.py3-none-any.whl (10 kB)
Installing collected packages: six, tensorflow-metal
  Attempting uninstall: six
    Found existing installation: six 1.16.0
    Uninstalling six-1.16.0:
      Successfully uninstalled six-1.16.0
Successfully installed six-1.15.0 tensorflow-metal-0.5.0
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 23.3.1
  latest version: 24.3.0

Please update conda by running

    $ conda update -n base -c conda-forge conda

Or to minimize the number of packages updated during conda update use

     conda install conda=24.3.0



# All requested packages already installed.

Collecting tensorflow-macos
  Downloading tensorflow_macos-2.13.0-cp38-cp38-macosx_12_0_arm64.whl.metadata (3.2 kB)
Collecting absl-py>=1.0.0 (from tensorflow-macos)
  Using cached absl_py-2.1.0-py3-none-any.whl.metadata (2.3 kB)
Collecting astunparse>=1.6.0 (from tensorflow-macos)
  Using cached astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)
Collecting flatbuffers>=23.1.21 (from tensorflow-macos)
  Using cached flatbuffers-24.3.25-py2.py3-none-any.whl.metadata (850 bytes)
Collecting gast<=0.4.0,>=0.2.1 (from tensorflow-macos)
  Using cached gast-0.4.0-py3-none-any.whl.metadata (1.1 kB)
Collecting google-pasta>=0.1.1 (from tensorflow-macos)
  Using cached google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)
Requirement already satisfied: h5py>=2.9.0 in /Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages (from tensorflow-macos) (3.6.0)
Collecting libclang>=13.0.0 (from tensorflow-macos)
  Using cached libclang-18.1.1-py2.py3-none-macosx_11_0_arm64.whl.metadata (5.2 kB)
Requirement already satisfied: numpy<=1.24.3,>=1.22 in /Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages (from tensorflow-macos) (1.23.2)
Collecting opt-einsum>=2.3.2 (from tensorflow-macos)
  Using cached opt_einsum-3.3.0-py3-none-any.whl.metadata (6.5 kB)
Collecting packaging (from tensorflow-macos)
  Using cached packaging-24.0-py3-none-any.whl.metadata (3.2 kB)
Collecting protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 (from tensorflow-macos)
  Using cached protobuf-4.25.3-cp37-abi3-macosx_10_9_universal2.whl.metadata (541 bytes)
Requirement already satisfied: setuptools in /Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages (from tensorflow-macos) (69.2.0)
Requirement already satisfied: six>=1.12.0 in /Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages (from tensorflow-macos) (1.16.0)
Collecting termcolor>=1.1.0 (from tensorflow-macos)
  Using cached termcolor-2.4.0-py3-none-any.whl.metadata (6.1 kB)
Collecting typing-extensions<4.6.0,>=3.6.6 (from tensorflow-macos)
  Downloading typing_extensions-4.5.0-py3-none-any.whl.metadata (8.5 kB)
Collecting wrapt>=1.11.0 (from tensorflow-macos)
  Downloading wrapt-1.16.0-cp38-cp38-macosx_11_0_arm64.whl.metadata (6.6 kB)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages (from tensorflow-macos) (1.46.3)
Collecting tensorboard<2.14,>=2.13 (from tensorflow-macos)
  Downloading tensorboard-2.13.0-py3-none-any.whl.metadata (1.8 kB)
Collecting tensorflow-estimator<2.14,>=2.13.0 (from tensorflow-macos)
  Downloading tensorflow_estimator-2.13.0-py2.py3-none-any.whl.metadata (1.3 kB)
Collecting keras<2.14,>=2.13.1 (from tensorflow-macos)
  Downloading keras-2.13.1-py3-none-any.whl.metadata (2.4 kB)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages (from astunparse>=1.6.0->tensorflow-macos) (0.43.0)
Collecting grpcio<2.0,>=1.24.3 (from tensorflow-macos)
  Downloading grpcio-1.62.1-cp38-cp38-macosx_10_10_universal2.whl.metadata (4.0 kB)
Collecting google-auth<3,>=1.6.3 (from tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached google_auth-2.29.0-py2.py3-none-any.whl.metadata (4.7 kB)
Collecting google-auth-oauthlib<1.1,>=0.5 (from tensorboard<2.14,>=2.13->tensorflow-macos)
  Downloading google_auth_oauthlib-1.0.0-py2.py3-none-any.whl.metadata (2.7 kB)
Collecting markdown>=2.6.8 (from tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached Markdown-3.6-py3-none-any.whl.metadata (7.0 kB)
Collecting requests<3,>=2.21.0 (from tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached requests-2.31.0-py3-none-any.whl.metadata (4.6 kB)
Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached tensorboard_data_server-0.7.2-py3-none-any.whl.metadata (1.1 kB)
Collecting werkzeug>=1.0.1 (from tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached werkzeug-3.0.2-py3-none-any.whl.metadata (4.1 kB)
Collecting cachetools<6.0,>=2.0.0 (from google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached cachetools-5.3.3-py3-none-any.whl.metadata (5.3 kB)
Collecting pyasn1-modules>=0.2.1 (from google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached pyasn1_modules-0.4.0-py3-none-any.whl.metadata (3.4 kB)
Collecting rsa<5,>=3.1.4 (from google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached rsa-4.9-py3-none-any.whl.metadata (4.2 kB)
Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<1.1,>=0.5->tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached requests_oauthlib-2.0.0-py2.py3-none-any.whl.metadata (11 kB)
Collecting importlib-metadata>=4.4 (from markdown>=2.6.8->tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached importlib_metadata-7.1.0-py3-none-any.whl.metadata (4.7 kB)
Collecting charset-normalizer<4,>=2 (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow-macos)
  Downloading charset_normalizer-3.3.2-cp38-cp38-macosx_11_0_arm64.whl.metadata (33 kB)
Collecting idna<4,>=2.5 (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached idna-3.6-py3-none-any.whl.metadata (9.9 kB)
Collecting urllib3<3,>=1.21.1 (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached urllib3-2.2.1-py3-none-any.whl.metadata (6.4 kB)
Collecting certifi>=2017.4.17 (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached certifi-2024.2.2-py3-none-any.whl.metadata (2.2 kB)
Collecting MarkupSafe>=2.1.1 (from werkzeug>=1.0.1->tensorboard<2.14,>=2.13->tensorflow-macos)
  Downloading MarkupSafe-2.1.5-cp38-cp38-macosx_10_9_universal2.whl.metadata (3.0 kB)
Collecting zipp>=0.5 (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached zipp-3.18.1-py3-none-any.whl.metadata (3.5 kB)
Collecting pyasn1<0.7.0,>=0.4.6 (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached pyasn1-0.6.0-py2.py3-none-any.whl.metadata (8.3 kB)
Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<1.1,>=0.5->tensorboard<2.14,>=2.13->tensorflow-macos)
  Using cached oauthlib-3.2.2-py3-none-any.whl.metadata (7.5 kB)
Downloading tensorflow_macos-2.13.0-cp38-cp38-macosx_12_0_arm64.whl (189.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 189.3/189.3 MB 9.1 MB/s eta 0:00:00
Using cached absl_py-2.1.0-py3-none-any.whl (133 kB)
Using cached astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
Using cached flatbuffers-24.3.25-py2.py3-none-any.whl (26 kB)
Using cached gast-0.4.0-py3-none-any.whl (9.8 kB)
Using cached google_pasta-0.2.0-py3-none-any.whl (57 kB)
Downloading keras-2.13.1-py3-none-any.whl (1.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 14.7 MB/s eta 0:00:00
Using cached libclang-18.1.1-py2.py3-none-macosx_11_0_arm64.whl (26.4 MB)
Using cached opt_einsum-3.3.0-py3-none-any.whl (65 kB)
Using cached protobuf-4.25.3-cp37-abi3-macosx_10_9_universal2.whl (394 kB)
Downloading tensorboard-2.13.0-py3-none-any.whl (5.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.6/5.6 MB 13.9 MB/s eta 0:00:00
Downloading grpcio-1.62.1-cp38-cp38-macosx_10_10_universal2.whl (10.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.1/10.1 MB 13.3 MB/s eta 0:00:00
Downloading tensorflow_estimator-2.13.0-py2.py3-none-any.whl (440 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 440.8/440.8 kB 12.0 MB/s eta 0:00:00
Using cached termcolor-2.4.0-py3-none-any.whl (7.7 kB)
Downloading typing_extensions-4.5.0-py3-none-any.whl (27 kB)
Downloading wrapt-1.16.0-cp38-cp38-macosx_11_0_arm64.whl (38 kB)
Using cached packaging-24.0-py3-none-any.whl (53 kB)
Using cached google_auth-2.29.0-py2.py3-none-any.whl (189 kB)
Downloading google_auth_oauthlib-1.0.0-py2.py3-none-any.whl (18 kB)
Using cached Markdown-3.6-py3-none-any.whl (105 kB)
Using cached requests-2.31.0-py3-none-any.whl (62 kB)
Using cached tensorboard_data_server-0.7.2-py3-none-any.whl (2.4 kB)
Using cached werkzeug-3.0.2-py3-none-any.whl (226 kB)
Using cached cachetools-5.3.3-py3-none-any.whl (9.3 kB)
Using cached certifi-2024.2.2-py3-none-any.whl (163 kB)
Downloading charset_normalizer-3.3.2-cp38-cp38-macosx_11_0_arm64.whl (119 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 119.1/119.1 kB 5.6 MB/s eta 0:00:00
Using cached idna-3.6-py3-none-any.whl (61 kB)
Using cached importlib_metadata-7.1.0-py3-none-any.whl (24 kB)
Downloading MarkupSafe-2.1.5-cp38-cp38-macosx_10_9_universal2.whl (18 kB)
Using cached pyasn1_modules-0.4.0-py3-none-any.whl (181 kB)
Using cached requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)
Using cached rsa-4.9-py3-none-any.whl (34 kB)
Using cached urllib3-2.2.1-py3-none-any.whl (121 kB)
Using cached oauthlib-3.2.2-py3-none-any.whl (151 kB)
Using cached pyasn1-0.6.0-py2.py3-none-any.whl (85 kB)
Using cached zipp-3.18.1-py3-none-any.whl (8.2 kB)
Installing collected packages: libclang, flatbuffers, zipp, wrapt, urllib3, typing-extensions, termcolor, tensorflow-estimator, tensorboard-data-server, pyasn1, protobuf, packaging, opt-einsum, oauthlib, MarkupSafe, keras, idna, grpcio, google-pasta, gast, charset-normalizer, certifi, cachetools, astunparse, absl-py, werkzeug, rsa, requests, pyasn1-modules, importlib-metadata, requests-oauthlib, markdown, google-auth, google-auth-oauthlib, tensorboard, tensorflow-macos
  Attempting uninstall: protobuf
    Found existing installation: protobuf 3.19.6
    Uninstalling protobuf-3.19.6:
      Successfully uninstalled protobuf-3.19.6
  Attempting uninstall: grpcio
    Found existing installation: grpcio 1.46.3
    Uninstalling grpcio-1.46.3:
      Successfully uninstalled grpcio-1.46.3
Successfully installed MarkupSafe-2.1.5 absl-py-2.1.0 astunparse-1.6.3 cachetools-5.3.3 certifi-2024.2.2 charset-normalizer-3.3.2 flatbuffers-24.3.25 gast-0.4.0 google-auth-2.29.0 google-auth-oauthlib-1.0.0 google-pasta-0.2.0 grpcio-1.62.1 idna-3.6 importlib-metadata-7.1.0 keras-2.13.1 libclang-18.1.1 markdown-3.6 oauthlib-3.2.2 opt-einsum-3.3.0 packaging-24.0 protobuf-4.25.3 pyasn1-0.6.0 pyasn1-modules-0.4.0 requests-2.31.0 requests-oauthlib-2.0.0 rsa-4.9 tensorboard-2.13.0 tensorboard-data-server-0.7.2 tensorflow-estimator-2.13.0 tensorflow-macos-2.13.0 termcolor-2.4.0 typing-extensions-4.5.0 urllib3-2.2.1 werkzeug-3.0.2 wrapt-1.16.0 zipp-3.18.1
Collecting tensorflow-metal
  Downloading tensorflow_metal-1.0.1-cp38-cp38-macosx_12_0_arm64.whl.metadata (1.2 kB)
Requirement already satisfied: wheel~=0.35 in /Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages (from tensorflow-metal) (0.43.0)
Requirement already satisfied: six>=1.15.0 in /Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages (from tensorflow-metal) (1.16.0)
Downloading tensorflow_metal-1.0.1-cp38-cp38-macosx_12_0_arm64.whl (1.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 4.4 MB/s eta 0:00:00
Installing collected packages: tensorflow-metal
Successfully installed tensorflow-metal-1.0.1
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)


(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python -c "import tensorflow as tf; print(tf.__version__)"

2.13.0
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)





(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "Convolutional_Neural_Network.py", line 106, in <module>
    import librosa
ModuleNotFoundError: No module named 'librosa'
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda install librosa
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 23.3.1
  latest version: 24.3.0

Please update conda by running

    $ conda update -n base -c conda-forge conda

Or to minimize the number of packages updated during conda update use

     conda install conda=24.3.0



## Package Plan ##

  environment location: /Users/deangladish/miniforge3/envs/tensorflow

  added / updated specs:
    - librosa


The following NEW packages will be INSTALLED:

  aom                conda-forge/osx-arm64::aom-3.7.1-h463b476_0
  audioread          conda-forge/osx-arm64::audioread-3.0.1-py38h10201cd_1
  blas               conda-forge/osx-arm64::blas-2.122-openblas
  blas-devel         conda-forge/osx-arm64::blas-devel-3.9.0-22_osxarm64_openblas
  brotli             conda-forge/osx-arm64::brotli-1.1.0-hb547adb_1
  brotli-bin         conda-forge/osx-arm64::brotli-bin-1.1.0-hb547adb_1
  brotli-python      conda-forge/osx-arm64::brotli-python-1.1.0-py38he333c0f_1
  cairo              conda-forge/osx-arm64::cairo-1.18.0-hd1e100b_0
  certifi            conda-forge/noarch::certifi-2024.2.2-pyhd8ed1ab_0
  cffi               conda-forge/osx-arm64::cffi-1.16.0-py38h73f40f7_0
  charset-normalizer conda-forge/noarch::charset-normalizer-3.3.2-pyhd8ed1ab_0
  cycler             conda-forge/noarch::cycler-0.12.1-pyhd8ed1ab_0
  dav1d              conda-forge/osx-arm64::dav1d-1.2.1-hb547adb_0
  decorator          conda-forge/noarch::decorator-5.1.1-pyhd8ed1ab_0
  expat              conda-forge/osx-arm64::expat-2.6.2-hebf3989_0
  ffmpeg             conda-forge/osx-arm64::ffmpeg-6.1.1-gpl_h4d3b11e_100
  font-ttf-dejavu-s~ conda-forge/noarch::font-ttf-dejavu-sans-mono-2.37-hab24e00_0
  font-ttf-inconsol~ conda-forge/noarch::font-ttf-inconsolata-3.000-h77eed37_0
  font-ttf-source-c~ conda-forge/noarch::font-ttf-source-code-pro-2.038-h77eed37_0
  font-ttf-ubuntu    conda-forge/noarch::font-ttf-ubuntu-0.83-h77eed37_1
  fontconfig         conda-forge/osx-arm64::fontconfig-2.14.2-h82840c6_0
  fonts-conda-ecosy~ conda-forge/noarch::fonts-conda-ecosystem-1-0
  fonts-conda-forge  conda-forge/noarch::fonts-conda-forge-1-0
  fonttools          conda-forge/osx-arm64::fonttools-4.51.0-py38h336bac9_0
  freetype           conda-forge/osx-arm64::freetype-2.12.1-hadb7bae_2
  fribidi            conda-forge/osx-arm64::fribidi-1.0.10-h27ca646_0
  gettext            conda-forge/osx-arm64::gettext-0.22.5-h8fbad5d_2
  gettext-tools      conda-forge/osx-arm64::gettext-tools-0.22.5-h8fbad5d_2
  gmp                conda-forge/osx-arm64::gmp-6.3.0-hebf3989_1
  gnutls             conda-forge/osx-arm64::gnutls-3.7.9-hd26332c_0
  graphite2          conda-forge/osx-arm64::graphite2-1.3.13-hebf3989_1003
  harfbuzz           conda-forge/osx-arm64::harfbuzz-8.3.0-h8f0ba13_0
  icu                conda-forge/osx-arm64::icu-73.2-hc8870d7_0
  idna               conda-forge/noarch::idna-3.6-pyhd8ed1ab_0
  importlib-metadata conda-forge/noarch::importlib-metadata-7.1.0-pyha770c72_0
  joblib             conda-forge/noarch::joblib-1.3.2-pyhd8ed1ab_0
  kiwisolver         conda-forge/osx-arm64::kiwisolver-1.4.5-py38h9afee92_1
  lame               conda-forge/osx-arm64::lame-3.100-h1a8c8d9_1003
  lazy_loader        conda-forge/noarch::lazy_loader-0.4-pyhd8ed1ab_0
  lcms2              conda-forge/osx-arm64::lcms2-2.16-ha0e7c42_0
  lerc               conda-forge/osx-arm64::lerc-4.0.0-h9a09cb3_0
  libasprintf        conda-forge/osx-arm64::libasprintf-0.22.5-h8fbad5d_2
  libasprintf-devel  conda-forge/osx-arm64::libasprintf-devel-0.22.5-h8fbad5d_2
  libass             conda-forge/osx-arm64::libass-0.17.1-hf7da4fe_1
  libbrotlicommon    conda-forge/osx-arm64::libbrotlicommon-1.1.0-hb547adb_1
  libbrotlidec       conda-forge/osx-arm64::libbrotlidec-1.1.0-hb547adb_1
  libbrotlienc       conda-forge/osx-arm64::libbrotlienc-1.1.0-hb547adb_1
  libdeflate         conda-forge/osx-arm64::libdeflate-1.20-h93a5062_0
  libexpat           conda-forge/osx-arm64::libexpat-2.6.2-hebf3989_0
  libflac            conda-forge/osx-arm64::libflac-1.4.3-hb765f3a_0
  libgettextpo       conda-forge/osx-arm64::libgettextpo-0.22.5-h8fbad5d_2
  libgettextpo-devel conda-forge/osx-arm64::libgettextpo-devel-0.22.5-h8fbad5d_2
  libglib            conda-forge/osx-arm64::libglib-2.80.0-hfc324ee_3
  libiconv           conda-forge/osx-arm64::libiconv-1.17-h0d3ecfb_2
  libidn2            conda-forge/osx-arm64::libidn2-2.3.7-h93a5062_0
  libintl            conda-forge/osx-arm64::libintl-0.22.5-h8fbad5d_2
  libintl-devel      conda-forge/osx-arm64::libintl-devel-0.22.5-h8fbad5d_2
  libjpeg-turbo      conda-forge/osx-arm64::libjpeg-turbo-3.0.0-hb547adb_1
  liblapacke         conda-forge/osx-arm64::liblapacke-3.9.0-22_osxarm64_openblas
  libllvm14          conda-forge/osx-arm64::libllvm14-14.0.6-hd1a9a77_4
  libogg             conda-forge/osx-arm64::libogg-1.3.4-h27ca646_1
  libopus            conda-forge/osx-arm64::libopus-1.3.1-h27ca646_1
  libpng             conda-forge/osx-arm64::libpng-1.6.43-h091b4b1_0
  librosa            conda-forge/noarch::librosa-0.10.1-pyhd8ed1ab_0
  libsndfile         conda-forge/osx-arm64::libsndfile-1.2.2-h9739721_1
  libtasn1           conda-forge/osx-arm64::libtasn1-4.19.0-h1a8c8d9_0
  libtiff            conda-forge/osx-arm64::libtiff-4.6.0-h07db509_3
  libunistring       conda-forge/osx-arm64::libunistring-0.9.10-h3422bc3_0
  libvorbis          conda-forge/osx-arm64::libvorbis-1.3.7-h9f76cd9_0
  libvpx             conda-forge/osx-arm64::libvpx-1.13.1-hb765f3a_0
  libwebp-base       conda-forge/osx-arm64::libwebp-base-1.3.2-hb547adb_0
  libxcb             conda-forge/osx-arm64::libxcb-1.15-hf346824_0
  libxml2            conda-forge/osx-arm64::libxml2-2.12.6-h0d0cfa8_1
  llvmlite           anaconda/osx-arm64::llvmlite-0.40.0-py38h514c7bf_0
  matplotlib-base    conda-forge/osx-arm64::matplotlib-base-3.5.3-py38h4399b95_2
  mpg123             conda-forge/osx-arm64::mpg123-1.32.6-hebf3989_0
  msgpack-python     conda-forge/osx-arm64::msgpack-python-1.0.7-py38h48d2fec_0
  munkres            conda-forge/noarch::munkres-1.1.4-pyh9f0ad1d_0
  nettle             conda-forge/osx-arm64::nettle-3.9.1-h40ed0f5_0
  numba              conda-forge/osx-arm64::numba-0.57.1-py38h11be589_0
  openblas           conda-forge/osx-arm64::openblas-0.3.27-openmp_h55c453e_0
  openh264           conda-forge/osx-arm64::openh264-2.4.0-h965bd2d_0
  openjpeg           conda-forge/osx-arm64::openjpeg-2.5.2-h9f1df11_0
  p11-kit            conda-forge/osx-arm64::p11-kit-0.24.1-h29577a5_0
  packaging          conda-forge/noarch::packaging-24.0-pyhd8ed1ab_0
  pcre2              conda-forge/osx-arm64::pcre2-10.43-h26f9a81_0
  pillow             conda-forge/osx-arm64::pillow-10.3.0-py38h9ef4633_0
  pixman             conda-forge/osx-arm64::pixman-0.43.4-hebf3989_0
  platformdirs       conda-forge/noarch::platformdirs-4.2.0-pyhd8ed1ab_0
  pooch              conda-forge/noarch::pooch-1.8.1-pyhd8ed1ab_0
  pthread-stubs      conda-forge/osx-arm64::pthread-stubs-0.4-h27ca646_1001
  pycparser          conda-forge/noarch::pycparser-2.22-pyhd8ed1ab_0
  pyparsing          conda-forge/noarch::pyparsing-3.1.2-pyhd8ed1ab_0
  pysocks            conda-forge/noarch::pysocks-1.7.1-pyha2e5f31_6
  pysoundfile        conda-forge/noarch::pysoundfile-0.12.1-pyhd8ed1ab_0
  python-dateutil    conda-forge/noarch::python-dateutil-2.9.0-pyhd8ed1ab_0
  requests           conda-forge/noarch::requests-2.31.0-pyhd8ed1ab_0
  scikit-learn       conda-forge/osx-arm64::scikit-learn-1.3.2-py38he1bc1c9_2
  scipy              anaconda/osx-arm64::scipy-1.10.1-py38h9d039d2_1
  soxr               conda-forge/osx-arm64::soxr-0.1.3-h5008568_3
  soxr-python        conda-forge/osx-arm64::soxr-python-0.3.7-py38hd97cf01_0
  svt-av1            conda-forge/osx-arm64::svt-av1-1.8.0-h463b476_0
  threadpoolctl      conda-forge/noarch::threadpoolctl-3.4.0-pyhc1e730c_0
  typing_extensions  conda-forge/noarch::typing_extensions-4.11.0-pyha770c72_0
  unicodedata2       conda-forge/osx-arm64::unicodedata2-15.1.0-py38hb192615_0
  urllib3            conda-forge/noarch::urllib3-2.2.1-pyhd8ed1ab_0
  x264               conda-forge/osx-arm64::x264-1!164.3095-h57fd34a_2
  x265               conda-forge/osx-arm64::x265-3.5-hbc6ce65_3
  xorg-libxau        conda-forge/osx-arm64::xorg-libxau-1.0.11-hb547adb_0
  xorg-libxdmcp      conda-forge/osx-arm64::xorg-libxdmcp-1.1.3-h27ca646_0
  zipp               conda-forge/noarch::zipp-3.17.0-pyhd8ed1ab_0


Proceed ([y]/n)? y


Downloading and Extracting Packages

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)









(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "Convolutional_Neural_Network.py", line 114, in <module>
    import tensorflow_model_optimization as tfmot
ModuleNotFoundError: No module named 'tensorflow_model_optimization'
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda install tensorflow_model_optmization
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.

PackagesNotFoundError: The following packages are not available from current channels:

  - tensorflow_model_optmization

Current channels:

  - https://conda.anaconda.org/conda-forge/osx-arm64
  - https://conda.anaconda.org/conda-forge/noarch
  - https://conda.anaconda.org/anaconda/osx-arm64
  - https://conda.anaconda.org/anaconda/noarch

To search for alternate channels that may provide the conda package you're
looking for, navigate to

    https://anaconda.org

and use the search bar at the top of the page.


(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)









(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "Convolutional_Neural_Network.py", line 114, in <module>
    import tensorflow_model_optimization as tfmot
ModuleNotFoundError: No module named 'tensorflow_model_optimization'
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda install tensorflow_model_optmization
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.

PackagesNotFoundError: The following packages are not available from current channels:

  - tensorflow_model_optmization

Current channels:

  - https://conda.anaconda.org/conda-forge/osx-arm64
  - https://conda.anaconda.org/conda-forge/noarch
  - https://conda.anaconda.org/anaconda/osx-arm64
  - https://conda.anaconda.org/anaconda/noarch

To search for alternate channels that may provide the conda package you're
looking for, navigate to

    https://anaconda.org

and use the search bar at the top of the page.


(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) pip install --user --upgrade tensorflow-model-optimization

Collecting tensorflow-model-optimization
  Using cached tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl.metadata (904 bytes)
Requirement already satisfied: absl-py~=1.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.4.0)
Requirement already satisfied: dm-tree~=0.1.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (0.1.8)
Requirement already satisfied: numpy~=1.23 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.26.4)
Requirement already satisfied: six~=1.14 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.15.0)
Using cached tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl (242 kB)
Installing collected packages: tensorflow-model-optimization
Successfully installed tensorflow-model-optimization-0.8.0
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "Convolutional_Neural_Network.py", line 114, in <module>
    import tensorflow_model_optimization as tfmot
ModuleNotFoundError: No module named 'tensorflow_model_optimization'
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)


(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "Convolutional_Neural_Network.py", line 114, in <module>
    import tensorflow_model_optimization as tfmot
ModuleNotFoundError: No module named 'tensorflow_model_optimization'
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda install tensorflow_model_optmization
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.

PackagesNotFoundError: The following packages are not available from current channels:

  - tensorflow_model_optmization

Current channels:

  - https://conda.anaconda.org/conda-forge/osx-arm64
  - https://conda.anaconda.org/conda-forge/noarch
  - https://conda.anaconda.org/anaconda/osx-arm64
  - https://conda.anaconda.org/anaconda/noarch

To search for alternate channels that may provide the conda package you're
looking for, navigate to

    https://anaconda.org

and use the search bar at the top of the page.


(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) pip install --user --upgrade tensorflow-model-optimization

Collecting tensorflow-model-optimization
  Using cached tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl.metadata (904 bytes)
Requirement already satisfied: absl-py~=1.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.4.0)
Requirement already satisfied: dm-tree~=0.1.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (0.1.8)
Requirement already satisfied: numpy~=1.23 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.26.4)
Requirement already satisfied: six~=1.14 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.15.0)
Using cached tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl (242 kB)
Installing collected packages: tensorflow-model-optimization
Successfully installed tensorflow-model-optimization-0.8.0
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "Convolutional_Neural_Network.py", line 114, in <module>
    import tensorflow_model_optimization as tfmot
ModuleNotFoundError: No module named 'tensorflow_model_optimization'
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) pip install --upgrade tensorflow-model-optimization

Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: tensorflow-model-optimization in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (0.8.0)
Requirement already satisfied: absl-py~=1.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.4.0)
Requirement already satisfied: dm-tree~=0.1.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (0.1.8)
Requirement already satisfied: numpy~=1.23 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.26.4)
Requirement already satisfied: six~=1.14 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.15.0)
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda install tensorflow-model-optimization
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.

PackagesNotFoundError: The following packages are not available from current channels:

  - tensorflow-model-optimization

Current channels:

  - https://conda.anaconda.org/conda-forge/osx-arm64
  - https://conda.anaconda.org/conda-forge/noarch
  - https://conda.anaconda.org/anaconda/osx-arm64
  - https://conda.anaconda.org/anaconda/noarch

To search for alternate channels that may provide the conda package you're
looking for, navigate to

    https://anaconda.org

and use the search bar at the top of the page.


(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)


(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "Convolutional_Neural_Network.py", line 114, in <module>
    import tensorflow_model_optimization as tfmot
ModuleNotFoundError: No module named 'tensorflow_model_optimization'
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda install tensorflow_model_optmization
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.

PackagesNotFoundError: The following packages are not available from current channels:

  - tensorflow_model_optmization

Current channels:

  - https://conda.anaconda.org/conda-forge/osx-arm64
  - https://conda.anaconda.org/conda-forge/noarch
  - https://conda.anaconda.org/anaconda/osx-arm64
  - https://conda.anaconda.org/anaconda/noarch

To search for alternate channels that may provide the conda package you're
looking for, navigate to

    https://anaconda.org

and use the search bar at the top of the page.


(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) pip install --user --upgrade tensorflow-model-optimization

Collecting tensorflow-model-optimization
  Using cached tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl.metadata (904 bytes)
Requirement already satisfied: absl-py~=1.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.4.0)
Requirement already satisfied: dm-tree~=0.1.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (0.1.8)
Requirement already satisfied: numpy~=1.23 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.26.4)
Requirement already satisfied: six~=1.14 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.15.0)
Using cached tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl (242 kB)
Installing collected packages: tensorflow-model-optimization
Successfully installed tensorflow-model-optimization-0.8.0
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
Traceback (most recent call last):
  File "Convolutional_Neural_Network.py", line 114, in <module>
    import tensorflow_model_optimization as tfmot
ModuleNotFoundError: No module named 'tensorflow_model_optimization'
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) pip install --upgrade tensorflow-model-optimization

Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: tensorflow-model-optimization in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (0.8.0)
Requirement already satisfied: absl-py~=1.2 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.4.0)
Requirement already satisfied: dm-tree~=0.1.1 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (0.1.8)
Requirement already satisfied: numpy~=1.23 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.26.4)
Requirement already satisfied: six~=1.14 in /Users/deangladish/Library/Python/3.10/lib/python/site-packages (from tensorflow-model-optimization) (1.15.0)
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) conda install tensorflow-model-optimization
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.

PackagesNotFoundError: The following packages are not available from current channels:

  - tensorflow-model-optimization

Current channels:

  - https://conda.anaconda.org/conda-forge/osx-arm64
  - https://conda.anaconda.org/conda-forge/noarch
  - https://conda.anaconda.org/anaconda/osx-arm64
  - https://conda.anaconda.org/anaconda/noarch

To search for alternate channels that may provide the conda package you're
looking for, navigate to

    https://anaconda.org

and use the search bar at the top of the page.


(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python -m pip install --upgrade tensorflow-model-optimization
Collecting tensorflow-model-optimization
  Using cached tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl.metadata (904 bytes)
Collecting absl-py~=1.2 (from tensorflow-model-optimization)
  Using cached absl_py-1.4.0-py3-none-any.whl.metadata (2.3 kB)
Collecting dm-tree~=0.1.1 (from tensorflow-model-optimization)
  Downloading dm_tree-0.1.8-cp38-cp38-macosx_11_0_arm64.whl.metadata (1.9 kB)
Requirement already satisfied: numpy~=1.23 in /Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages (from tensorflow-model-optimization) (1.23.2)
Requirement already satisfied: six~=1.14 in /Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages (from tensorflow-model-optimization) (1.16.0)
Using cached tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl (242 kB)
Using cached absl_py-1.4.0-py3-none-any.whl (126 kB)
Downloading dm_tree-0.1.8-cp38-cp38-macosx_11_0_arm64.whl (110 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 110.7/110.7 kB 1.3 MB/s eta 0:00:00
Installing collected packages: dm-tree, absl-py, tensorflow-model-optimization
  Attempting uninstall: absl-py
    Found existing installation: absl-py 2.1.0
    Uninstalling absl-py-2.1.0:
      Successfully uninstalled absl-py-2.1.0
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow-macos 2.13.0 requires typing-extensions<4.6.0,>=3.6.6, but you have typing-extensions 4.11.0 which is incompatible.
Successfully installed absl-py-1.4.0 dm-tree-0.1.8 tensorflow-model-optimization-0.8.0
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗) python Convolutional_Neural_Network.py
2024-04-08 00:22:22.238311: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M2 Pro
2024-04-08 00:22:22.238344: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
2024-04-08 00:22:22.238348: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.33 GB
2024-04-08 00:22:22.238571: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:303] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2024-04-08 00:22:22.238797: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:269] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
2024-04-08 00:22:23.982236: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
1/1 [==============================] - ETA: 0s - loss: 12.3695 - accuracy: 0.0000e+002024-04-08 00:22:26.436855: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
1/1 [==============================] - 3s 3s/step - loss: 12.3695 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Pruned CNN model weights and architecture saved.
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✗)

























commit 336648dde63bdb59429c6247a578b9e21aada5ca (HEAD -> main)
Author: gladishd <gladish.dean@gmail.com>
Date:   Mon Apr 8 00:27:41 2024 -0500

    commit

diff --git a/Data Zenodo/Convolutional_Neural_Network.py b/Data Zenodo/Convolutional_Neural_Network.py
index 0ce042c6..034d9186 100644
--- a/Data Zenodo/Convolutional_Neural_Network.py
+++ b/Data Zenodo/Convolutional_Neural_Network.py
@@ -8,6 +8,16 @@ something very special. They indicate that the dense layer has some kind of
 expectation; it expects the inputs to have a size of 55552, but in "reality" they
 actually have the inputs that have size 44800.

+The critical thing to note is that Python version 3.10 is too far, for Tensorflow.
+I just did
+(new_env) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✔) conda install python=3.9
+python -m pip install tensorflow_model_optimization
+python -m pip install tensorflow-macos==2.9.0 tensorflow-metal==0.6.0 --force-reinstall
+https://forums.developer.apple.com/forums/thread/710048
+python -m pip uninstall tensorflow tensorflow-macos tensorflow-metal
+python -m pip install tensorflow-macos tensorflow-metal
+
+
 Now, this brings about some kind of discrepancy. The discrepancy "arises" from
 the dimensions of the input spectrograms and "this, I think," is the way that
 the input spectrograms' dimensions match with the processing of the "convolutional"
@@ -97,11 +107,17 @@ import librosa
 import numpy as np
 import os
 import tensorflow as tf
-from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
+from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
 from tensorflow.keras.models import Sequential
 from sklearn.preprocessing import LabelEncoder
 from sklearn.model_selection import train_test_split
-from tensorflow_model_optimization.sparsity import keras as sparsity
+import tensorflow_model_optimization as tfmot
+
+# TensorFlow Model Optimization Toolkit specific imports for sparsity
+sparsity = tfmot.sparsity.keras
+
+
+

 # Pruning and Quantization Setup
 pruning_schedule = sparsity.PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.5,
diff --git a/Data Zenodo/logs/metrics/events.out.tfevents.1712553743.MacBook-Pro.local.53256.0.v2 b/Data Zenodo/logs/metrics/events.out.tfevents.1712553743.MacBook-Pro.local.53256.0.v2
new file mode 100644
index 00000000..559e93bc
Binary files /dev/null and b/Data Zenodo/logs/metrics/events.out.tfevents.1712553743.MacBook-Pro.local.53256.0.v2 differ
diff --git a/Data Zenodo/logs/train/events.out.tfevents.1712553743.MacBook-Pro.local.53256.1.v2 b/Data Zenodo/logs/train/events.out.tfevents.1712553743.MacBook-Pro.local.53256.1.v2
new file mode 100644
index 00000000..8b8fd7dd
Binary files /dev/null and b/Data Zenodo/logs/train/events.out.tfevents.1712553743.MacBook-Pro.local.53256.1.v2 differ
diff --git a/Data Zenodo/logs/validation/events.out.tfevents.1712553746.MacBook-Pro.local.53256.2.v2 b/Data Zenodo/logs/validation/events.out.tfevents.1712553746.MacBook-Pro.local.53256.2.v2
new file mode 100644
index 00000000..89e19e20
Binary files /dev/null and b/Data Zenodo/logs/validation/events.out.tfevents.1712553746.MacBook-Pro.local.53256.2.v2 differ
diff --git a/Data Zenodo/myprojectenv/bin/Activate.ps1 b/Data Zenodo/myprojectenv/bin/Activate.ps1
new file mode 100644
index 00000000..b49d77ba
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/Activate.ps1
@@ -0,0 +1,247 @@
+<#^M
+.Synopsis^M
+Activate a Python virtual environment for the current PowerShell session.^M
+^M
+.Description^M
+Pushes the python executable for a virtual environment to the front of the^M
+$Env:PATH environment variable and sets the prompt to signify that you are^M
+in a Python virtual environment. Makes use of the command line switches as^M
+well as the `pyvenv.cfg` file values present in the virtual environment.^M
+^M
+.Parameter VenvDir^M
+Path to the directory that contains the virtual environment to activate. The^M
+default value for this is the parent of the directory that the Activate.ps1^M
+script is located within.^M
+^M
+.Parameter Prompt^M
+The prompt prefix to display when this virtual environment is activated. By^M
+default, this prompt is the name of the virtual environment folder (VenvDir)^M
+surrounded by parentheses and followed by a single space (ie. '(.venv) ').^M
+^M
+.Example^M
+Activate.ps1^M
+Activates the Python virtual environment that contains the Activate.ps1 script.^M
+^M
+.Example^M
+Activate.ps1 -Verbose^M
+Activates the Python virtual environment that contains the Activate.ps1 script,^M
+and shows extra information about the activation as it executes.^M
+^M
+.Example^M
+Activate.ps1 -VenvDir C:\Users\MyUser\Common\.venv^M
+Activates the Python virtual environment located in the specified location.^M
+^M
+.Example^M
+Activate.ps1 -Prompt "MyPython"^M
+Activates the Python virtual environment that contains the Activate.ps1 script,^M
+and prefixes the current prompt with the specified string (surrounded in^M
+parentheses) while the virtual environment is active.^M
+^M
+.Notes^M
+On Windows, it may be required to enable this Activate.ps1 script by setting the^M
+execution policy for the user. You can do this by issuing the following PowerShell^M
+command:^M
+^M
+PS C:\> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser^M
+^M
+For more information on Execution Policies: ^M
+https://go.microsoft.com/fwlink/?LinkID=135170^M
+^M
+#>^M
+Param(^M
+    [Parameter(Mandatory = $false)]^M
+    [String]^M
+    $VenvDir,^M
+    [Parameter(Mandatory = $false)]^M
+    [String]^M
+    $Prompt^M
+)^M
+^M
+<# Function declarations --------------------------------------------------- #>^M
+^M
+<#^M
+.Synopsis^M
+Remove all shell session elements added by the Activate script, including the^M
+addition of the virtual environment's Python executable from the beginning of^M
+the PATH variable.^M
+^M
+.Parameter NonDestructive^M
+If present, do not remove this function from the global namespace for the^M
+session.^M
+^M
+#>^M
+function global:deactivate ([switch]$NonDestructive) {^M
+    # Revert to original values^M
+^M
+    # The prior prompt:^M
+    if (Test-Path -Path Function:_OLD_VIRTUAL_PROMPT) {^M
+        Copy-Item -Path Function:_OLD_VIRTUAL_PROMPT -Destination Function:prompt^M
+        Remove-Item -Path Function:_OLD_VIRTUAL_PROMPT^M
+    }^M
+^M
+    # The prior PYTHONHOME:^M
+    if (Test-Path -Path Env:_OLD_VIRTUAL_PYTHONHOME) {^M
+        Copy-Item -Path Env:_OLD_VIRTUAL_PYTHONHOME -Destination Env:PYTHONHOME^M
+        Remove-Item -Path Env:_OLD_VIRTUAL_PYTHONHOME^M
+    }^M
+^M
+    # The prior PATH:^M
+    if (Test-Path -Path Env:_OLD_VIRTUAL_PATH) {^M
+        Copy-Item -Path Env:_OLD_VIRTUAL_PATH -Destination Env:PATH^M
+        Remove-Item -Path Env:_OLD_VIRTUAL_PATH^M
+    }^M
+^M
+    # Just remove the VIRTUAL_ENV altogether:^M
+    if (Test-Path -Path Env:VIRTUAL_ENV) {^M
+        Remove-Item -Path env:VIRTUAL_ENV^M
+    }^M
+^M
+    # Just remove VIRTUAL_ENV_PROMPT altogether.^M
+    if (Test-Path -Path Env:VIRTUAL_ENV_PROMPT) {^M
+        Remove-Item -Path env:VIRTUAL_ENV_PROMPT^M
+    }^M
+^M
+    # Just remove the _PYTHON_VENV_PROMPT_PREFIX altogether:^M
+    if (Get-Variable -Name "_PYTHON_VENV_PROMPT_PREFIX" -ErrorAction SilentlyContinue) {^M
+        Remove-Variable -Name _PYTHON_VENV_PROMPT_PREFIX -Scope Global -Force^M
+    }^M
+^M
+    # Leave deactivate function in the global namespace if requested:^M
+    if (-not $NonDestructive) {^M
+        Remove-Item -Path function:deactivate^M
+    }^M
+}^M
+^M
+<#^M
+.Description^M
+Get-PyVenvConfig parses the values from the pyvenv.cfg file located in the^M
+given folder, and returns them in a map.^M
+^M
+For each line in the pyvenv.cfg file, if that line can be parsed into exactly^M
+two strings separated by `=` (with any amount of whitespace surrounding the =)^M
+then it is considered a `key = value` line. The left hand string is the key,^M
+the right hand is the value.^M
+^M
+If the value starts with a `'` or a `"` then the first and last character is^M
+stripped from the value before being captured.^M
+^M
+.Parameter ConfigDir^M
+Path to the directory that contains the `pyvenv.cfg` file.^M
+#>^M
+function Get-PyVenvConfig(^M
+    [String]^M
+    $ConfigDir^M
+) {^M
+    Write-Verbose "Given ConfigDir=$ConfigDir, obtain values in pyvenv.cfg"^M
+^M
+    # Ensure the file exists, and issue a warning if it doesn't (but still allow the function to continue).^M
+    $pyvenvConfigPath = Join-Path -Resolve -Path $ConfigDir -ChildPath 'pyvenv.cfg' -ErrorAction Continue^M
+^M
+    # An empty map will be returned if no config file is found.^M
+    $pyvenvConfig = @{ }^M
+^M
+    if ($pyvenvConfigPath) {^M
+^M
+        Write-Verbose "File exists, parse `key = value` lines"^M
+        $pyvenvConfigContent = Get-Content -Path $pyvenvConfigPath^M
+^M
+        $pyvenvConfigContent | ForEach-Object {^M
+            $keyval = $PSItem -split "\s*=\s*", 2^M
+            if ($keyval[0] -and $keyval[1]) {^M
+                $val = $keyval[1]^M
+^M
+                # Remove extraneous quotations around a string value.^M
+                if ("'""".Contains($val.Substring(0, 1))) {^M
+                    $val = $val.Substring(1, $val.Length - 2)^M
+                }^M
+^M
+                $pyvenvConfig[$keyval[0]] = $val^M
+                Write-Verbose "Adding Key: '$($keyval[0])'='$val'"^M
+            }^M
+        }^M
+    }^M
+    return $pyvenvConfig^M
+}^M
+^M
+^M
+<# Begin Activate script --------------------------------------------------- #>^M
+^M
+# Determine the containing directory of this script^M
+$VenvExecPath = Split-Path -Parent $MyInvocation.MyCommand.Definition^M
+$VenvExecDir = Get-Item -Path $VenvExecPath^M
+^M
+Write-Verbose "Activation script is located in path: '$VenvExecPath'"^M
+Write-Verbose "VenvExecDir Fullname: '$($VenvExecDir.FullName)"^M
+Write-Verbose "VenvExecDir Name: '$($VenvExecDir.Name)"^M
+^M
+# Set values required in priority: CmdLine, ConfigFile, Default^M
+# First, get the location of the virtual environment, it might not be^M
+# VenvExecDir if specified on the command line.^M
+if ($VenvDir) {^M
+    Write-Verbose "VenvDir given as parameter, using '$VenvDir' to determine values"^M
+}^M
+else {^M
+    Write-Verbose "VenvDir not given as a parameter, using parent directory name as VenvDir."^M
+    $VenvDir = $VenvExecDir.Parent.FullName.TrimEnd("\\/")^M
+    Write-Verbose "VenvDir=$VenvDir"^M
+}^M
+^M
+# Next, read the `pyvenv.cfg` file to determine any required value such^M
+# as `prompt`.^M
+$pyvenvCfg = Get-PyVenvConfig -ConfigDir $VenvDir^M
+^M
+# Next, set the prompt from the command line, or the config file, or^M
+# just use the name of the virtual environment folder.^M
+if ($Prompt) {^M
+    Write-Verbose "Prompt specified as argument, using '$Prompt'"^M
+}^M
+else {^M
+    Write-Verbose "Prompt not specified as argument to script, checking pyvenv.cfg value"^M
+    if ($pyvenvCfg -and $pyvenvCfg['prompt']) {^M
+        Write-Verbose "  Setting based on value in pyvenv.cfg='$($pyvenvCfg['prompt'])'"^M
+        $Prompt = $pyvenvCfg['prompt'];^M
+    }^M
+    else {^M
+        Write-Verbose "  Setting prompt based on parent's directory's name. (Is the directory name passed to venv module when creating the virtual environment)"^M
+        Write-Verbose "  Got leaf-name of $VenvDir='$(Split-Path -Path $venvDir -Leaf)'"^M
+        $Prompt = Split-Path -Path $venvDir -Leaf^M
+    }^M
+}^M
+^M
+Write-Verbose "Prompt = '$Prompt'"^M
+Write-Verbose "VenvDir='$VenvDir'"^M
+^M
+# Deactivate any currently active virtual environment, but leave the^M
+# deactivate function in place.^M
+deactivate -nondestructive^M
+^M
+# Now set the environment variable VIRTUAL_ENV, used by many tools to determine^M
+# that there is an activated venv.^M
+$env:VIRTUAL_ENV = $VenvDir^M
+^M
+if (-not $Env:VIRTUAL_ENV_DISABLE_PROMPT) {^M
+^M
+    Write-Verbose "Setting prompt to '$Prompt'"^M
+^M
+    # Set the prompt to include the env name^M
+    # Make sure _OLD_VIRTUAL_PROMPT is global^M
+    function global:_OLD_VIRTUAL_PROMPT { "" }^M
+    Copy-Item -Path function:prompt -Destination function:_OLD_VIRTUAL_PROMPT^M
+    New-Variable -Name _PYTHON_VENV_PROMPT_PREFIX -Description "Python virtual environment prompt prefix" -Scope Global -Option ReadOnly -Visibility Public -Value $Prompt^M
+^M
+    function global:prompt {^M
+        Write-Host -NoNewline -ForegroundColor Green "($_PYTHON_VENV_PROMPT_PREFIX) "^M
+        _OLD_VIRTUAL_PROMPT^M
+    }^M
+    $env:VIRTUAL_ENV_PROMPT = $Prompt^M
+}^M
+^M
+# Clear PYTHONHOME^M
+if (Test-Path -Path Env:PYTHONHOME) {^M
+    Copy-Item -Path Env:PYTHONHOME -Destination Env:_OLD_VIRTUAL_PYTHONHOME^M
+    Remove-Item -Path Env:PYTHONHOME^M
+}^M
+^M
+# Add the venv to the PATH^M
+Copy-Item -Path Env:PATH -Destination Env:_OLD_VIRTUAL_PATH^M
+$Env:PATH = "$VenvExecDir$([System.IO.Path]::PathSeparator)$Env:PATH"^M
diff --git a/Data Zenodo/myprojectenv/bin/activate b/Data Zenodo/myprojectenv/bin/activate
new file mode 100644
index 00000000..c0ce3238
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/activate
@@ -0,0 +1,69 @@
+# This file must be used with "source bin/activate" *from bash*
+# you cannot run it directly
+
+deactivate () {
+    # reset old environment variables
+    if [ -n "${_OLD_VIRTUAL_PATH:-}" ] ; then
+        PATH="${_OLD_VIRTUAL_PATH:-}"
+        export PATH
+        unset _OLD_VIRTUAL_PATH
+    fi
+    if [ -n "${_OLD_VIRTUAL_PYTHONHOME:-}" ] ; then
+        PYTHONHOME="${_OLD_VIRTUAL_PYTHONHOME:-}"
+        export PYTHONHOME
+        unset _OLD_VIRTUAL_PYTHONHOME
+    fi
+
+    # This should detect bash and zsh, which have a hash command that must
+    # be called to get it to forget past commands.  Without forgetting
+    # past commands the $PATH changes we made may not be respected
+    if [ -n "${BASH:-}" -o -n "${ZSH_VERSION:-}" ] ; then
+        hash -r 2> /dev/null
+    fi
+
+    if [ -n "${_OLD_VIRTUAL_PS1:-}" ] ; then
+        PS1="${_OLD_VIRTUAL_PS1:-}"
+        export PS1
+        unset _OLD_VIRTUAL_PS1
+    fi
+
+    unset VIRTUAL_ENV
+    unset VIRTUAL_ENV_PROMPT
+    if [ ! "${1:-}" = "nondestructive" ] ; then
+    # Self destruct!
+        unset -f deactivate
+    fi
+}
+
+# unset irrelevant variables
+deactivate nondestructive
+
+VIRTUAL_ENV="/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv"
+export VIRTUAL_ENV
+
+_OLD_VIRTUAL_PATH="$PATH"
+PATH="$VIRTUAL_ENV/bin:$PATH"
+export PATH
+
+# unset PYTHONHOME if set
+# this will fail if PYTHONHOME is set to the empty string (which is bad anyway)
+# could use `if (set -u; : $PYTHONHOME) ;` in bash
+if [ -n "${PYTHONHOME:-}" ] ; then
+    _OLD_VIRTUAL_PYTHONHOME="${PYTHONHOME:-}"
+    unset PYTHONHOME
+fi
+
+if [ -z "${VIRTUAL_ENV_DISABLE_PROMPT:-}" ] ; then
+    _OLD_VIRTUAL_PS1="${PS1:-}"
+    PS1="(myprojectenv) ${PS1:-}"
+    export PS1
+    VIRTUAL_ENV_PROMPT="(myprojectenv) "
+    export VIRTUAL_ENV_PROMPT
+fi
+
+# This should detect bash and zsh, which have a hash command that must
+# be called to get it to forget past commands.  Without forgetting
+# past commands the $PATH changes we made may not be respected
+if [ -n "${BASH:-}" -o -n "${ZSH_VERSION:-}" ] ; then
+    hash -r 2> /dev/null
+fi
diff --git a/Data Zenodo/myprojectenv/bin/activate.csh b/Data Zenodo/myprojectenv/bin/activate.csh
new file mode 100644
index 00000000..e2533693
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/activate.csh
@@ -0,0 +1,26 @@
+# This file must be used with "source bin/activate.csh" *from csh*.
+# You cannot run it directly.
+# Created by Davide Di Blasi <davidedb@gmail.com>.
+# Ported to Python 3.3 venv by Andrew Svetlov <andrew.svetlov@gmail.com>
+
+alias deactivate 'test $?_OLD_VIRTUAL_PATH != 0 && setenv PATH "$_OLD_VIRTUAL_PATH" && unset _OLD_VIRTUAL_PATH; rehash; test $?_OLD_VIRTUAL_PROMPT != 0 && set prompt="$_OLD_VIRTUAL_PROMPT" && unset _OLD_VIRTUAL_PROMPT; unsetenv VIRTUAL_ENV; unsetenv VIRTUAL_ENV_PROMPT; test "\!:*" != "nondestructive" && unalias deactivate'
+
+# Unset irrelevant variables.
+deactivate nondestructive
+
+setenv VIRTUAL_ENV "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv"
+
+set _OLD_VIRTUAL_PATH="$PATH"
+setenv PATH "$VIRTUAL_ENV/bin:$PATH"
+
+
+set _OLD_VIRTUAL_PROMPT="$prompt"
+
+if (! "$?VIRTUAL_ENV_DISABLE_PROMPT") then
+    set prompt = "(myprojectenv) $prompt"
+    setenv VIRTUAL_ENV_PROMPT "(myprojectenv) "
+endif
+
+alias pydoc python -m pydoc
+
+rehash
diff --git a/Data Zenodo/myprojectenv/bin/activate.fish b/Data Zenodo/myprojectenv/bin/activate.fish
new file mode 100644
index 00000000..bc8f5084
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/activate.fish
@@ -0,0 +1,66 @@
+# This file must be used with "source <venv>/bin/activate.fish" *from fish*
+# (https://fishshell.com/); you cannot run it directly.
+
+function deactivate  -d "Exit virtual environment and return to normal shell environment"
+    # reset old environment variables
+    if test -n "$_OLD_VIRTUAL_PATH"
+        set -gx PATH $_OLD_VIRTUAL_PATH
+        set -e _OLD_VIRTUAL_PATH
+    end
+    if test -n "$_OLD_VIRTUAL_PYTHONHOME"
+        set -gx PYTHONHOME $_OLD_VIRTUAL_PYTHONHOME
+        set -e _OLD_VIRTUAL_PYTHONHOME
+    end
+
+    if test -n "$_OLD_FISH_PROMPT_OVERRIDE"
+        functions -e fish_prompt
+        set -e _OLD_FISH_PROMPT_OVERRIDE
+        functions -c _old_fish_prompt fish_prompt
+        functions -e _old_fish_prompt
+    end
+
+    set -e VIRTUAL_ENV
+    set -e VIRTUAL_ENV_PROMPT
+    if test "$argv[1]" != "nondestructive"
+        # Self-destruct!
+        functions -e deactivate
+    end
+end
+
+# Unset irrelevant variables.
+deactivate nondestructive
+
+set -gx VIRTUAL_ENV "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv"
+
+set -gx _OLD_VIRTUAL_PATH $PATH
+set -gx PATH "$VIRTUAL_ENV/bin" $PATH
+
+# Unset PYTHONHOME if set.
+if set -q PYTHONHOME
+    set -gx _OLD_VIRTUAL_PYTHONHOME $PYTHONHOME
+    set -e PYTHONHOME
+end
+
+if test -z "$VIRTUAL_ENV_DISABLE_PROMPT"
+    # fish uses a function instead of an env var to generate the prompt.
+
+    # Save the current fish_prompt function as the function _old_fish_prompt.
+    functions -c fish_prompt _old_fish_prompt
+
+    # With the original prompt function renamed, we can override with our own.
+    function fish_prompt
+        # Save the return status of the last command.
+        set -l old_status $status
+
+        # Output the venv prompt; color taken from the blue of the Python logo.
+        printf "%s%s%s" (set_color 4B8BBE) "(myprojectenv) " (set_color normal)
+
+        # Restore the return status of the previous command.
+        echo "exit $old_status" | .
+        # Output the original/"old" prompt.
+        _old_fish_prompt
+    end
+
+    set -gx _OLD_FISH_PROMPT_OVERRIDE "$VIRTUAL_ENV"
+    set -gx VIRTUAL_ENV_PROMPT "(myprojectenv) "
+end
diff --git a/Data Zenodo/myprojectenv/bin/f2py b/Data Zenodo/myprojectenv/bin/f2py
new file mode 100755
index 00000000..7121ab10
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/f2py
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from numpy.f2py.f2py2e import main
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(main())
diff --git a/Data Zenodo/myprojectenv/bin/import_pb_to_tensorboard b/Data Zenodo/myprojectenv/bin/import_pb_to_tensorboard
new file mode 100755
index 00000000..f10c60da
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/import_pb_to_tensorboard
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from tensorflow.python.tools.import_pb_to_tensorboard import main
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(main())
diff --git a/Data Zenodo/myprojectenv/bin/markdown-it b/Data Zenodo/myprojectenv/bin/markdown-it
new file mode 100755
index 00000000..2ed3a6cd
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/markdown-it
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from markdown_it.cli.parse import main
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(main())
diff --git a/Data Zenodo/myprojectenv/bin/markdown_py b/Data Zenodo/myprojectenv/bin/markdown_py
new file mode 100755
index 00000000..d77bfb04
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/markdown_py
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from markdown.__main__ import run
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(run())
diff --git a/Data Zenodo/myprojectenv/bin/normalizer b/Data Zenodo/myprojectenv/bin/normalizer
new file mode 100755
index 00000000..c8d275a7
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/normalizer
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from charset_normalizer.cli import cli_detect
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(cli_detect())
diff --git a/Data Zenodo/myprojectenv/bin/numba b/Data Zenodo/myprojectenv/bin/numba
new file mode 100755
index 00000000..d8d3c4f3
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/numba
@@ -0,0 +1,8 @@
+#!/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3
+# -*- coding: UTF-8 -*-
+from __future__ import print_function, division, absolute_import
+
+from numba.misc.numba_entry import main
+
+if __name__ == "__main__":
+    main()
diff --git a/Data Zenodo/myprojectenv/bin/pip b/Data Zenodo/myprojectenv/bin/pip
new file mode 100755
index 00000000..39fdc80f
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/pip
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from pip._internal.cli.main import main
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(main())
diff --git a/Data Zenodo/myprojectenv/bin/pip3 b/Data Zenodo/myprojectenv/bin/pip3
new file mode 100755
index 00000000..39fdc80f
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/pip3
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from pip._internal.cli.main import main
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(main())
diff --git a/Data Zenodo/myprojectenv/bin/pip3.10 b/Data Zenodo/myprojectenv/bin/pip3.10
new file mode 100755
index 00000000..39fdc80f
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/pip3.10
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from pip._internal.cli.main import main
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(main())
diff --git a/Data Zenodo/myprojectenv/bin/pygmentize b/Data Zenodo/myprojectenv/bin/pygmentize
new file mode 100755
index 00000000..675c4eb5
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/pygmentize
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from pygments.cmdline import main
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(main())
diff --git a/Data Zenodo/myprojectenv/bin/python b/Data Zenodo/myprojectenv/bin/python
new file mode 120000
index 00000000..b8a0adbb
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/python
@@ -0,0 +1 @@
+python3
\ No newline at end of file
diff --git a/Data Zenodo/myprojectenv/bin/python3 b/Data Zenodo/myprojectenv/bin/python3
new file mode 120000
index 00000000..1ec499c5
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/python3
@@ -0,0 +1 @@
+/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
\ No newline at end of file
diff --git a/Data Zenodo/myprojectenv/bin/python3.10 b/Data Zenodo/myprojectenv/bin/python3.10
new file mode 120000
index 00000000..b8a0adbb
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/python3.10
@@ -0,0 +1 @@
+python3
\ No newline at end of file
diff --git a/Data Zenodo/myprojectenv/bin/saved_model_cli b/Data Zenodo/myprojectenv/bin/saved_model_cli
new file mode 100755
index 00000000..5418b73b
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/saved_model_cli
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from tensorflow.python.tools.saved_model_cli import main
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(main())
diff --git a/Data Zenodo/myprojectenv/bin/tensorboard b/Data Zenodo/myprojectenv/bin/tensorboard
new file mode 100755
index 00000000..152f3bcd
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/tensorboard
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from tensorboard.main import run_main
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(run_main())
diff --git a/Data Zenodo/myprojectenv/bin/tf_upgrade_v2 b/Data Zenodo/myprojectenv/bin/tf_upgrade_v2
new file mode 100755
index 00000000..4ce64b45
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/tf_upgrade_v2
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from tensorflow.tools.compatibility.tf_upgrade_v2_main import main
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(main())
diff --git a/Data Zenodo/myprojectenv/bin/tflite_convert b/Data Zenodo/myprojectenv/bin/tflite_convert
new file mode 100755
index 00000000..88db4574
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/tflite_convert
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from tensorflow.lite.python.tflite_convert import main
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(main())
diff --git a/Data Zenodo/myprojectenv/bin/toco b/Data Zenodo/myprojectenv/bin/toco
new file mode 100755
index 00000000..88db4574
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/toco
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from tensorflow.lite.python.tflite_convert import main
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(main())
diff --git a/Data Zenodo/myprojectenv/bin/toco_from_protos b/Data Zenodo/myprojectenv/bin/toco_from_protos
new file mode 100755
index 00000000..a816c69f
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/toco_from_protos
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from tensorflow.lite.toco.python.toco_from_protos import main
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(main())
diff --git a/Data Zenodo/myprojectenv/bin/wheel b/Data Zenodo/myprojectenv/bin/wheel
new file mode 100755
index 00000000..6a82fa83
--- /dev/null
+++ b/Data Zenodo/myprojectenv/bin/wheel
@@ -0,0 +1,10 @@
+#!/bin/sh
+'''exec' "/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/myprojectenv/bin/python3" "$0" "$@"
+' '''
+# -*- coding: utf-8 -*-
+import re
+import sys
+from wheel.cli import main
+if __name__ == '__main__':
+    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
+    sys.exit(main())
diff --git a/Data Zenodo/myprojectenv/lib/python3.10/site-packages/Markdown-3.6.dist-info/INSTALLER b/Data Zenodo/myprojectenv/lib/python3.10/site-packages/Markdown-3.6.dist-info/INSTALLER
new file mode 100644
index 00000000..a1b589e3
--- /dev/null
+++ b/Data Zenodo/myprojectenv/lib/python3.10/site-packages/Markdown-3.6.dist-info/INSTALLER
@@ -0,0 +1 @@
+pip
diff --git a/Data Zenodo/myprojectenv/lib/python3.10/site-packages/Markdown-3.6.dist-info/LICENSE.md b/Data Zenodo/myprojectenv/lib/python3.10/site-packages/Markdown-3.6.dist-info/LICENSE.md
new file mode 100644
index 00000000..6249d60c
--- /dev/null
+++ b/Data Zenodo/myprojectenv/lib/python3.10/site-packages/Markdown-3.6.dist-info/LICENSE.md
@@ -0,0 +1,30 @@
+BSD 3-Clause License
+
+Copyright 2007, 2008 The Python Markdown Project (v. 1.7 and later)
+Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
+Copyright 2004 Manfred Stienstra (the original version)
+
+Redistribution and use in source and binary forms, with or without
+modification, are permitted provided that the following conditions are met:
+
+1. Redistributions of source code must retain the above copyright notice, this
+   list of conditions and the following disclaimer.
+
+2. Redistributions in binary form must reproduce the above copyright notice,
+   this list of conditions and the following disclaimer in the documentation
+   and/or other materials provided with the distribution.
+
+3. Neither the name of the copyright holder nor the names of its
+   contributors may be used to endorse or promote products derived from
+   this software without specific prior written permission.
+
+THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
+AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
+IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
+DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
+FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
+DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
+SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
+CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
+OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
+OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
diff --git a/Data Zenodo/myprojectenv/lib/python3.10/site-packages/Markdown-3.6.dist-info/METADATA b/Data Zenodo/myprojectenv/lib/python3.10/site-packages/Markdown-3.6.dist-info/METADATA
new file mode 100644
index 00000000..516d18d6
--- /dev/null
+++ b/Data Zenodo/myprojectenv/lib/python3.10/site-packages/Markdown-3.6.dist-info/METADATA
@@ -0,0 +1,146 @@
+Metadata-Version: 2.1
+Name: Markdown
+Version: 3.6
+Summary: Python implementation of John Gruber's Markdown.
+Author: Manfred Stienstra, Yuri Takhteyev
+Author-email: Waylan limberg <python.markdown@gmail.com>
+Maintainer: Isaac Muse
+Maintainer-email: Waylan Limberg <python.markdown@gmail.com>
+License: BSD 3-Clause License
+
+        Copyright 2007, 2008 The Python Markdown Project (v. 1.7 and later)
+        Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
+        Copyright 2004 Manfred Stienstra (the original version)
+
+        Redistribution and use in source and binary forms, with or without
+        modification, are permitted provided that the following conditions are met:
+
+        1. Redistributions of source code must retain the above copyright notice, this
+           list of conditions and the following disclaimer.
+
+        2. Redistributions in binary form must reproduce the above copyright notice,
+           this list of conditions and the following disclaimer in the documentation
+           and/or other materials provided with the distribution.
+
+        3. Neither the name of the copyright holder nor the names of its
+           contributors may be used to endorse or promote products derived from
+           this software without specific prior written permission.
+
+        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
+        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
+        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
+        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
+        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
+        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
+        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
+        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
+        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
+        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
+
+Project-URL: Homepage, https://Python-Markdown.github.io/
+Project-URL: Documentation, https://Python-Markdown.github.io/
+Project-URL: Repository, https://github.com/Python-Markdown/markdown
+Project-URL: Issue Tracker, https://github.com/Python-Markdown/markdown/issues
+Project-URL: Changelog, https://python-markdown.github.io/changelog/
+Keywords: markdown,markdown-parser,python-markdown,markdown-to-html
+Classifier: Development Status :: 5 - Production/Stable
+Classifier: License :: OSI Approved :: BSD License
+Classifier: Operating System :: OS Independent
+Classifier: Programming Language :: Python
+Classifier: Programming Language :: Python :: 3
+Classifier: Programming Language :: Python :: 3.8
+Classifier: Programming Language :: Python :: 3.9
+Classifier: Programming Language :: Python :: 3.10
+Classifier: Programming Language :: Python :: 3.11
+Classifier: Programming Language :: Python :: 3.12
+Classifier: Programming Language :: Python :: 3 :: Only
+Classifier: Programming Language :: Python :: Implementation :: CPython
+Classifier: Programming Language :: Python :: Implementation :: PyPy
+Classifier: Topic :: Communications :: Email :: Filters
+Classifier: Topic :: Internet :: WWW/HTTP :: Dynamic Content :: CGI Tools/Libraries
+Classifier: Topic :: Internet :: WWW/HTTP :: Site Management
+Classifier: Topic :: Software Development :: Documentation
+Classifier: Topic :: Software Development :: Libraries :: Python Modules
+Classifier: Topic :: Text Processing :: Filters
+Classifier: Topic :: Text Processing :: Markup :: HTML
+Classifier: Topic :: Text Processing :: Markup :: Markdown
+Requires-Python: >=3.8
+Description-Content-Type: text/markdown
+License-File: LICENSE.md
+Requires-Dist: importlib-metadata >=4.4 ; python_version < "3.10"
+Provides-Extra: docs
+Requires-Dist: mkdocs >=1.5 ; extra == 'docs'
+Requires-Dist: mkdocs-nature >=0.6 ; extra == 'docs'
+Requires-Dist: mdx-gh-links >=0.2 ; extra == 'docs'
+Requires-Dist: mkdocstrings[python] ; extra == 'docs'
+Requires-Dist: mkdocs-gen-files ; extra == 'docs'
+Requires-Dist: mkdocs-section-index ; extra == 'docs'
+Requires-Dist: mkdocs-literate-nav ; extra == 'docs'
+Provides-Extra: testing
+Requires-Dist: coverage ; extra == 'testing'
+Requires-Dist: pyyaml ; extra == 'testing'
+
+[Python-Markdown][]
+===================
+
+[![Build Status][build-button]][build]
+[![Coverage Status][codecov-button]][codecov]
+[![Latest Version][mdversion-button]][md-pypi]
+[![Python Versions][pyversion-button]][md-pypi]
+[![BSD License][bsdlicense-button]][bsdlicense]
+[![Code of Conduct][codeofconduct-button]][Code of Conduct]
+
+[build-button]: https://github.com/Python-Markdown/markdown/workflows/CI/badge.svg?event=push
+[build]: https://github.com/Python-Markdown/markdown/actions?query=workflow%3ACI+event%3Apush
+[codecov-button]: https://codecov.io/gh/Python-Markdown/markdown/branch/master/graph/badge.svg
+[codecov]: https://codecov.io/gh/Python-Markdown/markdown
+[mdversion-button]: https://img.shields.io/pypi/v/Markdown.svg
+[md-pypi]: https://pypi.org/project/Markdown/
+[pyversion-button]: https://img.shields.io/pypi/pyversions/Markdown.svg
+[bsdlicense-button]: https://img.shields.io/badge/license-BSD-yellow.svg
+[bsdlicense]: https://opensource.org/licenses/BSD-3-Clause
+[codeofconduct-button]: https://img.shields.io/badge/code%20of%20conduct-contributor%20covenant-green.svg?style=flat-square
+[Code of Conduct]: https://github.com/Python-Markdown/markdown/blob/master/CODE_OF_CONDUCT.md
+
+This is a Python implementation of John Gruber's [Markdown][].
+It is almost completely compliant with the reference implementation,
+though there are a few known issues. See [Features][] for information
+on what exactly is supported and what is not. Additional features are
+supported by the [Available Extensions][].
+
+[Python-Markdown]: https://Python-Markdown.github.io/
+[Markdown]: https://daringfireball.net/projects/markdown/
+[Features]: https://Python-Markdown.github.io#Features
+[Available Extensions]: https://Python-Markdown.github.io/extensions
+
+Documentation
+-------------
+
+```bash
+pip install markdown
+```
+```python
+import markdown
+html = markdown.markdown(your_text_string)
+```
Last login: Mon Apr  8 00:19:20 on ttys006
cd%
(base) ~ cd CS-7643-O01
(base) ~/CS-7643-O01 ls
A3                             Assignment 2, Question 2.pdf   Assignment 2.pdf               Group_Project 2.zip            assignment2-spring2024         transposed_convolution.py
A3.zip                         Assignment 2, Question 2.tex   Assignment 2.synctex.gz        Group_Project.zip              assignment4_spring24
Assignment 2, Question 2.aux   Assignment 2.aux               Assignment 2.tex               assignment1                    assignment4_spring24.zip
Assignment 2, Question 2.log   Assignment 2.log               Group_Project                  assignment1-theory-problem-set assignment_4_testing
(base) ~/CS-7643-O01 cd Group_Project
(base) ~/CS-7643-O01/Group_Project (main ✔) git show
(base) ~/CS-7643-O01/Group_Project (main ✔) git filter-branch --force --index-filter \
"git rm -rf --cached --ignore-unmatch myprojectenv" \
--prune-empty --tag-name-filter cat -- --all

WARNING: git-filter-branch has a glut of gotchas generating mangled history
	 rewrites.  Hit Ctrl-C before proceeding to abort, then use an
	 alternative filtering tool such as 'git filter-repo'
	 (https://github.com/newren/git-filter-repo/) instead.  See the
	 filter-branch manual page for more details; to squelch this warning,
	 set FILTER_BRANCH_SQUELCH_WARNING=1.
Proceeding with filter-branch...

Rewrite 336648dde63bdb59429c6247a578b9e21aada5ca (8/8) (1 seconds passed, remaining 0 predicted)
WARNING: Ref 'refs/heads/main' is unchanged
WARNING: Ref 'refs/remotes/origin-github/main' is unchanged
WARNING: Ref 'refs/remotes/origin/main' is unchanged
(base) ~/CS-7643-O01/Group_Project (main ✔) cd Data\ Zenodo
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✔) git filter-branch --force --index-filter \
"git rm -rf --cached --ignore-unmatch myprojectenv" \
--prune-empty --tag-name-filter cat -- --all

WARNING: git-filter-branch has a glut of gotchas generating mangled history
	 rewrites.  Hit Ctrl-C before proceeding to abort, then use an
	 alternative filtering tool such as 'git filter-repo'
	 (https://github.com/newren/git-filter-repo/) instead.  See the
	 filter-branch manual page for more details; to squelch this warning,
	 set FILTER_BRANCH_SQUELCH_WARNING=1.
Proceeding with filter-branch...

You need to run this command from the toplevel of the working tree.
(base) ~/CS-7643-O01/Group_Project/Data Zenodo (main ✔) cd ..
(base) ~/CS-7643-O01/Group_Project (main ✔)






