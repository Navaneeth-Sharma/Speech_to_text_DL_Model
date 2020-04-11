# Speech Recognition Using 1D convnet

Here the input data is the samples of word like cat,dog,etc.I have created my own data set containing 11 labels and 85 sounds for each label.The duration is 1 sec for each wav file.

I am using time series data to train and predict the model.Since the data contains the amplitude and the time for perticular word, I am using the samples of the sound data of the word.

I am using Librosa package(which is popular in sound data analysis) to convert the wav audio file into samples.This is in preprocess.py.check out.I am using a sampling rate of 16000 as it is generally used for speech analysis.

The input shape is of (8000,1) (the half of sampling rate) for model.This model contains 2 1D convnet and 2 Dense network as in main_sr.py.I have  used K-flod method since I had few data.you can use any data, but make sure you remove croped  wave in preprocess.py since i had to crop it for 1 sec.

I got around 96.5% of training accuracy and around 85% of validation accuracy.Since I am using my own data which is very less, I got less accuracy.



The required packeges are Librosa,matplotlib,numpy,'keras'.
