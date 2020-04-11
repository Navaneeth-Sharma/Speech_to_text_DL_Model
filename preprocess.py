import os
import numpy as np 
import librosa

data_dir = '.../train/audio/'


a = os.listdir(data_dir)
print(a)

all_wave = []
all_label = []

audio_files = np.asarray(a)
for j in audio_files:

	folder_dir = data_dir+str(j)
	list_dirs = os.listdir(folder_dir)

	for k in list_dirs:

		samples ,sample_rate= librosa.load(folder_dir+'/'+str(k),16000)

		time = np.arange(0,len(samples))/sample_rate

		
			# croping the audio
		def cut(data, freq, start, end):
			end = int(end*freq)
			if end > len(data):
				return data[int(start*freq):]
			return data[int(start*freq):end]
		croped_wave = cut(samples,sample_rate,0.5,1.5)
		time = np.arange(0,len(croped_wave))/sample_rate

		samples = librosa.resample(croped_wave,sample_rate,8000)

		if(len(samples)== 8000):
			all_wave.append(samples)
			all_label.append(j)

from sklearn.preprocessing import LabelEncoder
lab_enc = LabelEncoder()
y = lab_enc.fit_transform(all_label)
classes = list(lab_enc.classes_)

from keras.utils import to_categorical
y = to_categorical(y,num_classes=len(a))


all_wave = np.array(all_wave).reshape(-1,8000,1)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.25,shuffle=True)



