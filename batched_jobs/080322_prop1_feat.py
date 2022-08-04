import sys
sys.path.append('/home/ci411/volume_estimation/')
import numpy as np
import pandas as pd
import os
import librosa as lr
from librosa import display

from featurization_funcs import *

AUDIO_OUT = '/scratch/ci411/sonos_rirs/reverb_audio'
FEATURES_OUT = '/scratch/ci411/sonos_rirs/features'



feature_name = '080322_10k_prop1'


#route to audio csv and load
audio_name = '080322_10K'
audio_path = os.path.join(AUDIO_OUT, audio_name)
audio_df = pd.read_csv(os.path.join(audio_path, 'audio_df.csv'))

#load audio to read input dimensionality
sr = 16000
path_audio = audio_df.iloc[0]['file_clip']
y, sr= lr.load(path_audio, sr=sr)
n_samples = len(y)

#define featurizer
input_shape = n_samples

mag_fb = gammatone_featurizer(input_shape, 'mag')

phase_fb = gammatone_featurizer(input_shape, 'phase')

phase_lf = gammatone_featurizer(input_shape, 'phase', num_freq=5,
                               low_freq=50, high_freq=500)

cont_lf = gammatone_featurizer(input_shape, 'continuity', num_freq=5,
                               low_freq=50, high_freq=500)

out_len = mag_fb.out_shape[1]

dft_f = dft_feat(input_shape, out_len=out_len)
dfts_f = dft_feat(input_shape, is_sorted=True, out_len=out_len)
cepstrum_f = cepstrum_feat(input_shape, out_len=out_len) 
rms_f = envelope_feat(input_shape, out_len=out_len)

lf_feats = [dft_f, dfts_f, cepstrum_f, rms_f] 
lf_feats_block = featurizer_block(input_shape, featurizations=lf_feats,
                                  out_len=out_len)


baseline = [mag_fb, lf_feats_block]

prop1 = [phase_fb, lf_feats_block]
prop2 = [phase_fb, cont_lf]
prop3 = [mag_fb, phase_lf, cont_lf]
prop4 = [mag_fb, phase_lf, lf_feats_block]

#baseline_block = featurizer_block(input_shape, featurizations=baseline,
#                                  out_len=out_len)
prop1_block = featurizer_block(input_shape, featurizations=prop1,
                                  out_len=out_len)
#prop2_block = featurizer_block(input_shape, featurizations=prop2,
#                                  out_len=out_len)
#prop3_block = featurizer_block(input_shape, featurizations=prop3,
#                                  out_len=out_len)
#prop4_block = featurizer_block(input_shape, featurizations=prop4,
#                                  out_len=out_len)

featurizer = prop1_block    

#set output route
feature_path = os.path.join(FEATURES_OUT, feature_name)
if not os.path.exists(feature_path):
    os.makedirs(feature_path)

#process features
eval_rooms = ['air_booth', 'air_office','ace_leture_Room_1', 'ace_Meeting_Room_1', 'Hotel_SkalskyDvur_ConferenceRoom2']
print("Processing features to " + feature_path)
feat_df = process_features(audio_df, featurizer, feature_path, sr=sr, eval_rooms=eval_rooms)
