import os
import json
import sys
sys.path.append("/home/ci411/volume_estimation/")

import model_funcs
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm 

from tqdm import tqdm

from eval_funcs import *

def compute_eval_metrics(dataloader, model, log=True):
    target_sum = 0
    pred_sum = 0
    n_steps = 0
    print("Computing sums...")
    for (x,y) in tqdm(dataloader):        
        (x, y) = (x.to(device), y.to(device))
        y = y.item()
        pred = model(x).item()
        target_sum += np.sum(y)
        pred_sum += np.sum(pred)
        n_steps += 1
        del x, y
    
    torch.cuda.empty_cache()

    target_mean = target_sum/n_steps
    pred_mean = pred_sum/n_steps
    
    mse = 0
    mean_error = 0
    cov = 0
    abs_log_ratio = 0
    
    var_pred = 0 #technically var * N but gets cancelled out in Pearson calculation
    var_target = 0 
    
    print("Computing metrics...")
    for (x,y) in tqdm(dataloader):        
        (x, y) = (x.to(device), y.to(device))
        pred = model(x)
        mse += MSE(pred, y).item()
        mean_error += Bias(pred, y).item()
        cov += CovStep(pred, y, pred_mean, target_mean).item()
        abs_log_ratio += MeanAbsLogStep(pred, y, log=log).item()
        
        var_pred += MSE(pred, pred_mean).item()
        var_target += MSE(y, target_mean).item()
        del x, y
        
    out_dict = {}
    out_dict['mse'] = (mse / n_steps)
    out_dict['bias'] = (mean_error / n_steps)
    out_dict['pearson_cor'] = (cov/(np.sqrt(var_pred) * np.sqrt(var_target)))
    out_dict['mean_mult'] = (np.exp(abs_log_ratio/n_steps))
    out_dict['var_ratio'] = (np.sqrt(var_pred) / np.sqrt(var_target))
    
    return out_dict
