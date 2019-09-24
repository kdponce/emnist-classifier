This is a basic EMNIST Letters Classifier that I made with the primary goal of identifying the effect
of each layer type in a CNN model to the evaluation accuracy and loss. It is a tutorial project of
sorts with the purpose of allowing me to gain an in-depth understanding of the topics that I have 
mostly glazed over during my implementation of another project, a facial expression recognizer using a modified VGG16 network.

___

How to run:

Make sure the following packages are installed:
> keras (or keras-gpu)

> opencv

> numpy

For keras-gpu users, additional packages might be needed for it to run properly.
However, I have observed that Conda should install all required packages automatically during the
installation of keras-gpu.

As of now, the "master" branch is the current implementation of the classifier 
with data augmentation (zoom, width shift, height shift) enabled. 
A separate branch, "feature/data-augmentation", contains the same architecture 
but with the data augmentation code removed.

Generate the .h5 model file by running "train.py". To predict, run "test.py" to evaluate the
model on a test.jpg file placed in the root directory.

___

Features to add/try:

1. More complex architectures (e.g. VGG16, Inception-v4).
2. Train for more epochs; and
3. Implement early stopping to mitigate overfitting.
4. Host a live version (i.e. mini-app)
