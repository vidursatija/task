import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa.display

df_samples = pd.read_csv('samples_short.csv')
df_samples = df_samples[df_samples['n_coughs'] == 1]

df = pd.read_csv('ground_truth_short_fixed.csv')
df['duration'] = df.apply(lambda row: row['end'] - row['start'], axis=1)

df_samples['start'] = df.start.tolist()
df_samples['end'] = df.end.tolist()

sosHigh = signal.butter(10, 100, 'hp', fs=16000, output='sos')
sosLow = signal.butter(10, 2000, 'lp', fs=16000, output='sos')
# High pass and low pass filters
# The frequency of coughs is between 100 Hz and 2000 Hz.

for i, (idx, row) in enumerate(df_samples.iterrows()):
	# print(idx)
	y, sr = librosa.load('sounds/'+row['file_path'], sr=16000)
	y1 = signal.sosfilt(sosHigh, y)
	y2 = signal.sosfilt(sosLow, y1)
	y2 = y2[int(row['start']*16000-512):int(row['end']*16000+512)]
	# now only frequencies between 100 and 2000 Hz are present
	stft = np.abs(librosa.stft(y2, n_fft=512, hop_length=512//4))**2
	pdb = librosa.power_to_db(stft, ref=np.max)
	plt.subplot(5, 2, i+1)
	librosa.display.specshow(pdb, fmax=8000, hop_length=512//4)
plt.show()