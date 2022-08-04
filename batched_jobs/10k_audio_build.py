import sys
sys.path.append('/home/ci411/volume_estimation/')
import numpy as np
import pandas as pd
import os
import librosa as lr
from librosa import display

import featurization_funcs as featurize

MAIN_DIR = '/home/ci411/volume_estimation/'
AUDIO_OUT = '/scratch/ci411/sonos_rirs/reverb_audio'

speech_dir = "/scratch/ci411/sonos_rirs/voice_ace/Speech"

audioset_name = '080322_10K'

target_rate = 16000

out_path = os.path.join(AUDIO_OUT, audioset_name)

if not os.path.exists(out_path):
    os.makedirs(out_path)

speech_df = pd.read_csv(os.path.join(MAIN_DIR, 'ace_speech.csv'))
rir_df = pd.read_csv(os.path.join(MAIN_DIR, '0803_rir_df.csv'))

#figure out how many samples per room we want
n_rooms = rir_df.nunique(axis=0)['room']
n_samples = 2000 #will be multiplied bt 5 by snr augmentation
n_per_room = (n_samples//n_rooms) + 1 #a bit high

#sample each room equally
rir_sample = rir_df.groupby('room').sample(n=n_per_room, replace=True)

snr_aug = [np.infty, 30, 20, 10, 0]

print("GENERATING AUDIO")
featurize.generate_rir_audio(n_samples, speech_df, rir_sample, sr=target_rate, snr_list=snr_aug,\
                             verbose=True, out_path=out_path)                