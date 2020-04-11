import os
import numpy as np 
import matplotlib.pyplot as plt 
from glob import glob

import librosa
from preprocess import *

import keras
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D, GRU, Bidirectional
from keras.models import Model
from keras import regularizers
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()

def bulid_model():

	inputs = Input(shape=(8000,1))

	# conv = Bidirectional(layers.GRU(256))(inputs)

	# First Conv1D layer
	conv = Conv1D(128,15,padding='valid', activation='relu', strides=1)(inputs)
	conv = MaxPooling1D(3)(conv)
	conv = Dropout(0.4)(conv)

	# Second Conv1D layer
	conv = Conv1D(256,11, padding='valid', activation='relu', strides=1)(conv)
	conv = MaxPooling1D(3)(conv)
	conv = Dropout(0.4)(conv)

	# # Third Conv1D layer
	# conv = Conv1D(512, 9,padding='valid', activation='relu', strides=1)(conv)
	# conv = MaxPooling1D(3)(conv)
	# conv = Dropout(0.2)(conv)

	# Dense Layer 1
	conv = Dense(512,activation='relu')(conv)
	conv = Dropout(0.4)(conv)

	#Dense Layer 6
	conv = Dense(1024, activation='relu')(conv)
	conv = Dropout(0.4)(conv)

	# # Third Conv1D layer
	# conv = Conv1D(32, 11,padding='valid', activation='relu', strides=1)(conv)
	# conv = MaxPooling1D(3)(conv)
	# conv = Dropout(0.2)(conv)

	# # Third Conv1D layer
	# conv = Conv1D(32, 11,padding='valid', activation='relu', strides=1)(conv)
	# conv = MaxPooling1D(3)(conv)
	# conv = Dropout(0.2)(conv)

	# #Fourth Conv1D layer
	# conv = Conv1D(32, 11,padding='valid', activation='relu', strides=1)(conv)
	# conv = MaxPooling1D(3)(conv)
	# conv = Dropout(0.2)(conv)

	#Fifth Conv1D layers
	# conv = Conv1D(64, 9,padding='valid', activation='relu', strides=1)(conv)
	# conv = Conv1D(64, 7,padding='valid', activation='relu', strides=1)(conv)
	# conv = MaxPooling1D(3)(conv)
	# conv = Dropout(0.5)(conv)

	# #Sixth Conv1D layer
	# conv = Conv1D(64, 5,padding='valid', activation='relu', strides=1)(conv)
	# conv = MaxPooling1D(3)(conv)
	# conv = Dropout(0.2)(conv)

	# # Seventh Conv1D layer
	# conv = Conv1D(32, 5,padding='valid', activation='relu', strides=1)(conv)
	# conv = MaxPooling1D(3)(conv)
	# conv = Dropout(0.6)(conv)

	

	

	# # Dense Layer 2
	# conv = Dense(64,activation='relu')(conv)
	# conv = Dropout(0.6)(conv)

	

	# # Dense Layer 3
	# conv = Dense(128,activation='relu')(conv)
	# conv = Dropout(0.6)(conv)

	

	# #Dense Layer 4
	# conv = Dense(256, activation='relu')(conv)
	# conv = Dropout(0.2)(conv)

	

	# #Dense Layer 5
	# conv = Dense(128, activation='relu')(conv)
	# conv = Dropout(0.2)(conv)

	

	

	

	#Flatten layer
	conv = Flatten()(conv)

	outputs = Dense(len(a), activation='softmax')(conv)

	model = Model(inputs, outputs)
	model.summary()

	# rmsprop = keras.optimizers.RMSprop(lr=0.000001)
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


	return model


k=2
num_val_samples = len(x_train)//k
num_val_samples2 = len(y_train)//k
num_epochs = 100
all_scores = []

for i in range(k):
	print('processing fold #',i+1)
	val_data = x_train[:i * num_val_samples:(i+1)*num_val_samples]
	val_targets = y_train[:i*num_val_samples:(i+1)*num_val_samples]

	partial_x_train = np.concatenate(
		[x_train[:i*num_val_samples*1:],
		x_train[(i+1)*num_val_samples::]])
	partial_y_train = np.concatenate(
		[y_train[:i*num_val_samples2],
		y_train[(i+1)*num_val_samples2:]]
		)

	model = bulid_model()

	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.00001) 
	mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

	history=model.fit(partial_x_train, partial_y_train ,epochs=30,callbacks=[es,mc],batch_size=1, validation_data=(x_val,y_val))

def smoothed_curve(points,factor=0.8):
	smothed_points = []
	for point in points:
		if smothed_points:
			previous = smothed_points[-1]
			smothed_points.append(previous*factor+point*(1-factor))
		else:
			smothed_points.append(point)
	return smothed_points

history_dict = history.history
loss_values = history_dict['loss']
val_loss_val = history_dict['val_loss']


epochs = range(1,len(loss_values)+1)

plt.plot(epochs,smoothed_curve(loss_values),'bo',label='Training Loss')
plt.plot(epochs,smoothed_curve(val_loss_val),'b',label='Validation Loss')
plt.title('Train and valid loss')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.legend()
plt.show()

acc_values = history_dict['acc']
val_acc_val = history_dict['val_acc']


plt.plot(epochs,smoothed_curve(acc_values),'bo',label='Training acc')
plt.plot(epochs,smoothed_curve(val_acc_val),'b',label='Validation acc')
plt.title('Train and valid acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()