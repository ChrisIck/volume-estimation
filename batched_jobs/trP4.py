import sys
sys.path.append("..")

import model_funcs
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

MODELS_DIR = '/scratch/ci411/sonos_rirs/models/'
FEATURES_DIR = '/scratch/ci411/sonos_rirs/features/'

feature_set = '080322_10k_prop4'

model_dict = {}
model_dict['name'] = "080322_prop4"
model_dict['notes'] = "mag+ lf phase + lf 5 feats"
model_dict['data_path'] = os.path.join(FEATURES_DIR, feature_set, 'feature_df.csv')
model_dict['model_path'] = os.path.join(MODELS_DIR, model_dict['name'])

model_funcs.train_model(model_funcs.Baseline_Model, model_dict,\
                        overwrite=True, log=True)