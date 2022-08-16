import numpy as np

import scipy as sp
from scipy import fft
from scipy.signal import convolve


import librosa as lr
from librosa import feature

import gammatone
from gammatone import gtgram, filters, fftweight

from tqdm import tqdm

import os

import pandas as pd
import soundfile as sf
import pickle

class _featurizer:
    def __init__(self, input_len, sr=16000):
        self.input_len = input_len
        self.sr = sr

        self.out_shape = None
        
        #run process to get out_shape
        _ = self.process(np.ones(input_len))
        print("Feature output shape: {}".format(self.out_shape))

    def process(self, wave):
        pass

class featurizer_block(_featurizer):
    def __init__(self, input_len, featurizations=None, sr=16000,
                 out_len=None, normed=True):
        self.featurizations = featurizations
        self.input_len = input_len
        self.sr = sr
        self.normed=normed
        self.out_len = out_len
        self.out_shape = None
        
        #run process to get out_shape
        _ = self.process(np.ones(input_len))
        print("Featurization block output shape: {}".format(self.out_shape))
        
    def process(self, wave):
        out_feature = None
        for feat_func in self.featurizations:
            feature = feat_func.process(wave)
            feature = feature[:,:self.out_len]
            if self.normed:
                feature = feature/max(feature.flatten())
            if out_feature is None:
                out_feature = feature
            else:
                out_feature = np.vstack((out_feature,feature))
        
        if self.out_shape is None:
            self.out_shape = out_feature.shape
        return out_feature
        
    
from gammatone import gtgram, filters

class gammatone_feat(_featurizer):
    def __init__(self, input_len, sr=16000, low_freq=50, high_freq=2000, num_freq=20,
                 window_s=64, hop_s=32):
        
        self.input_len = input_len
        self.sr = sr
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_freq = num_freq
        self.window_s = window_s
        self.hop_s = hop_s
        
        self.out_shape = None

        #compute filterbank
        self.center_freqs = gammatone.filters.erb_space(low_freq=self.low_freq,
                                                   high_freq=self.high_freq,
                                                   num=self.num_freq)
        
        self.fcoefs = np.flipud(gammatone.filters.make_erb_filters(self.sr, self.center_freqs))

        #run process to get out_shape
        _ = self.process(np.ones(input_len))
        print("Gammatone output shape: {}".format(self.out_shape))
        
    def process(self, wave):
        #apply filterbank to signal
        xf = gammatone.filters.erb_filterbank(wave, self.fcoefs)
        xe = np.power(xf, 2)

        #apply windowing to signal
        filterbank_cols = xe.shape[1]
        ncols = (1 + int(np.floor((filterbank_cols - self.window_s)/ self.hop_s)))

        y = np.zeros((self.num_freq, ncols))

        for cnum in range(ncols):
            segment = xe[:, cnum * self.hop_s + np.arange(self.window_s)]
            y[:, cnum] = np.sqrt(segment.mean(1))

        features = y
        
        if self.out_shape is None:
            self.out_shape = features.shape
        return features

class dft_feat(_featurizer):
    def __init__(self, input_len, sr=16000, out_len=None, is_sorted=False):
        self.input_len = input_len
        self.sr = sr
        self.out_len = out_len
        self.is_sorted = is_sorted
        
        self.out_shape = None
        #run process to get out_shape
        _ = self.process(np.ones(input_len))
        print("Dft output shape: {}".format(self.out_shape))

    
    def process(self, wave):
        dft = sp.fft.fft(wave)
        if self.out_len is not None:
            dft = dft[:self.out_len]
        if self.is_sorted:
            dft = np.sort(dft)
            
        if self.out_shape is None:
            self.out_shape = (1,len(dft))
        dft = dft.reshape(self.out_shape)
        return dft

    
class cepstrum_feat(_featurizer):
    
    def __init__(self, input_len, sr=16000, out_len=None):
        self.input_len = input_len
        self.sr = sr
        self.out_len = out_len

        self.out_shape = None
        
        #run process to get out_shape
        _ = self.process(np.ones(input_len))
        print("Ceptrum output shape: {}".format(self.out_shape))

    def process(self, wave):
        dft = sp.fft.fft(wave)
        if self.out_len is not None:
            dft = dft[:self.out_len]
        #compute logarithm of spectral amplitude of dft
        log_spec_amp = np.log(np.power(np.abs(dft),2))        
        #compute ifft of log spectral amplitude
        ifft_lsp = sp.fft.ifft(log_spec_amp)
        #compute magnitude of ifft
        cepstrum = np.power(np.abs(ifft_lsp),2)
        
        if self.out_shape is None:
            self.out_shape = (1,len(cepstrum))
        cepstrum = cepstrum.reshape(self.out_shape)            
        return cepstrum
    
class envelope_feat(_featurizer):
    def __init__(self, input_len, sr=16000, window_s=64, hop_s=32,
                 out_len=None):
        self.input_len = input_len
        self.sr = sr
        self.window_s = window_s
        self.hop_s = hop_s
        self.out_len = out_len

        self.out_shape = None
        
        #run process to get out_shape
        _ = self.process(np.ones(input_len))
        print("Envelope output shape: {}".format(self.out_shape))

    def process(self, wave):
        log_energy = lr.feature.rms(y=wave, frame_length=self.window_s,
                                    hop_length=self.hop_s)

        if self.out_len is not None:
            log_energy = log_energy[:,:self.out_len]
        
        if self.out_shape is None:
            self.out_shape = log_energy.shape
        return log_energy

class gammatone_featurizer:
    def __init__(self, input_len, feat_type, sr=16000, num_freq=20, window_s=64,\
                 hop_s=32, low_freq=50, high_freq=2000, deriv_seq=None):
        
        self.feat_type = feat_type #can be 'mag', 'phase'
        self.input_len = input_len
        self.sr = sr
        self.num_freq = num_freq
        self.window_s = window_s
        self.hop_s = hop_s
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.nfft = int(2 ** (np.ceil(np.log2(2 * self.window_s))))
        self.deriv_seq = deriv_seq #list of derivatives 
                                   #(along either bin axis [0] or time axis [1])

        self.out_shape = None

        #run process to get out_shape
        _ = self.process(np.ones(input_len))
        
        print("{} features output shape: {}".format(self.feat_type, self.out_shape))

    def process(self, wave):
        weights, _ = gammatone.fftweight.fft_weights(self.nfft, self.sr, self.num_freq, 1,\
                                                     self.low_freq, self.high_freq, self.nfft/2+1)

        sgram = gammatone.fftweight.specgram(wave, self.nfft, self.sr, self.window_s, self.hop_s)
        if self.feat_type == 'mag':
            result = weights.dot(np.abs(sgram))
        elif self.feat_type == 'phase':     
            result = weights.dot(np.angle(sgram))
        
        if self.deriv_seq is not None:
            for ax in self.deriv_seq:
                result = np.gradient(result)[ax]
             
        if self.out_shape is None:
            self.out_shape = result.shape
        
        return result

    
def get_power(y):
    rms = lr.feature.rms(y=y)
    power = np.average(rms)**2
    return power

def conv_signals(speech, rir, snr=np.infty, noise=None):
    if noise == None:
        noise = np.random.normal(size=len(speech))
    noise_ratio = get_power(speech) / np.exp(snr/20)
    noise_scaled = noise * np.sqrt(noise_ratio)
    
    speech_signal = convolve(speech, rir, mode='same')
    noise_signal = convolve(noise_scaled, rir, mode='same')
    out_signal = speech_signal + noise_signal
    out_signal = out_signal / np.max(out_signal.flatten())
    return out_signal

def window_audio(speech, n_samples):
    offset_max = len(speech) - (n_samples+1)
    offset = np.random.randint(0, offset_max)
    return speech[offset:offset+n_samples]

def generate_rir_audio(n_samples, speech_df, rir_df, len_clip=4, sr=22050, return_audio=False,\
                       snr_list=[np.infty], verbose=False, out_path=None):
    #len_clip - length of clip in seconds
    valid_speech = speech_df[speech_df['length (s)']>=len_clip]
    speech_sample = valid_speech.sample(n_samples, replace=True).reset_index()
    rir_sample = rir_df.sample(n_samples, replace=True).reset_index()
        
    sample_df = speech_sample.join(rir_sample, lsuffix='_speech', rsuffix='_rir')\
                             .drop(columns=['index_speech','index_rir'])
    audio_clips = []
    
    output_df_list = []
    for row in tqdm(sample_df.iterrows()):
        speech, _ = lr.load(row[1]['file_speech'], sr=sr)
        rir, _ = lr.load(row[1]['file_rir'], sr=sr)
        windowed_speech = window_audio(speech, len_clip*sr)
        for snr in snr_list:
            out_row = row[1].copy()
            out_row['snr'] = snr
            conv_audio = conv_signals(windowed_speech, rir, snr=snr)
            if snr is np.infty:
                snr_tag = "snr_clean"
            else:
                snr_tag = "snr_{}db".format(snr)
            clipname = "clip{}_{}".format(row[0],snr_tag)
            out_row['clip_label'] = clipname
            if return_audio:
                audio_clips.append(conv_audio)
            if out_path is not None:
                file = os.path.join(out_path, '{}.wav'.format(clipname))
                out_row['file_clip'] = file
                sf.write(file, conv_audio, sr)
                if verbose:
                    print("Saved file to {}".format(file))
            output_df_list.append(out_row)
 
    out_df = pd.DataFrame(output_df_list).reset_index().drop(columns=['index'])
    df_path = os.path.join(out_path,'audio_df.csv')
    out_df.to_csv(df_path, index=False)
    print("Results stored to {}".format(df_path))
    if return_audio:
        return out_df, audio_clips
    else:
        return out_df
    
def process_features(pairs_df, featurizer, output_dir, sr=16000, verbose=False,\
                     return_features=False, eval_rooms=['air_office','ace_leture_Room_1']):
    features = []
    volumes = []
    
    featurizer_path = os.path.join(output_dir, 'featurizer.pickle')
    
    with open(featurizer_path, 'wb') as f:
        pickle.dump(featurizer, f)
        
    if verbose:
        print("Saved featurizer to {}".format(featurizer_path))
        
    feat_files = []    
    for row in tqdm(pairs_df.iterrows()):
        row_tup = row[1]
        label = row_tup['clip_label']
        audio, _ = lr.load(row_tup['file_clip'], sr=sr)
        feature = featurizer.process(audio)
        
        filename = os.path.join(output_dir, label + '.npz')
        np.savez(filename, feat=feature, vol=row_tup['vol'])
        feat_files.append(filename)
        if verbose:
            print("Saved feature to {}".format(filename))
        
        if return_features:
            features.append(feature)
            volumes.append(row_tup['vol'])
    
    feat_df = pairs_df.copy()
    feat_df['file_feature'] = feat_files
    
    test_df = feat_df[feat_df['room'].isin(eval_rooms)]
    train_df = feat_df.drop(test_df.index)
    val_df = train_df.sample(frac=.1)
    train_df = train_df.drop(val_df.index)

    feat_df['split'] = None
    feat_df.loc[train_df.index, 'split'] = 'train'
    feat_df.loc[val_df.index, 'split'] = 'val'
    feat_df.loc[test_df.index, 'split'] = 'test'
    feat_df['split'].value_counts()
    
    
    
    df_path = os.path.join(output_dir, 'feature_df.csv')
    feat_df.to_csv(df_path, index=False)
    print("Saved to {}".format(df_path))
    
    if return_features:
        return feat_df, zip(features, volumes)
    else:
        return feat_df