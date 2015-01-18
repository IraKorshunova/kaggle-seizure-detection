Kaggle Seizure Detection Challenge
=========
Solution for [Kaggle Seizure Detection Challenge](http://www.kaggle.com/c/seizure-detection).
Convolutional neural networks were applied to raw EEG data, but in a bad way:
features from different channels were combined only in the hidden layer.
Presumably, good working architecture should do the convolution across all channels in
the first layer.
