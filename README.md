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
