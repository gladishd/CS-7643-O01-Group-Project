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











