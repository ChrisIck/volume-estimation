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


MODELS_DIR = '/scratch/ci411/sonos_rirs/models/'

def get_model_hist_spec_state(model_name, experiment_name):
    hist_dir = os.path.join(MODELS_DIR, experiment_name, model_name, 'hist.json')
    json_spec = os.path.join(MODELS_DIR, experiment_name, model_name, model_name+'_spec.json')
    model_state = os.path.join(MODELS_DIR, experiment_name, model_name, 'model_state.pt')
    
    if os.path.exists(hist_dir):
        with open(hist_dir) as f:
            hist = json.load(f)
    else:
        print("No history file at {}".format(hist_dir))
        hist = None

    with open(json_spec) as f:
        spec = json.load(f)
        
    return hist, spec, model_state

def plot_experiment_metrics(experiment_name, model_names=None):
    experiment_dir = os.path.join(MODELS_DIR, experiment_name)
    
    if model_names is None:
        model_names = os.listdir(experiment_dir)
        model_names.sort()
            
    n = len(model_names)
    width = 0.5/n
    
    fig, axs = plt.subplots(1,5, figsize=(12,6))
    fig.tight_layout()

    for i, model in enumerate(model_names):
        json_metric = os.path.join(experiment_dir, model, 'test_metrics.json')
        if os.path.exists(json_metric):
            with open(json_metric) as f:
                metric_dict = json.load(f)
        else:
            print("No metric file at {}".format(json_metric))
            continue
        
        offset = i/(2*n) - 0.25
        for j, (key, val) in enumerate(metric_dict.items()):
            axs[j].bar(offset, val, width, label=model)
            axs[j].set_title(key)
            axs[j].set_xticks([])
    plt.legend(loc='lower right')
    return fig

def plot_experiment_curves(experiment_name, model_names=None, offset=100):
        
    experiment_dir = os.path.join(MODELS_DIR, experiment_name)
    
    if model_names is None:
        model_names = os.listdir(experiment_dir)
        model_names.sort()
            
    n = len(model_names)
    width = 0.5/n
    cmap = cm.ScalarMappable(cmap='coolwarm')
    colors = cmap.to_rgba(np.arange(n))
    
    fig, axs = plt.subplots(7,1, figsize=(12,16))
    fig.tight_layout()
    axs_lengths = np.ones(7)

    for i, model in enumerate(model_names):
        hist_json = os.path.join(experiment_dir, model, 'hist.json')
        if os.path.exists(hist_json):
            with open(hist_json) as f:
                hist = json.load(f)
        else:
            print("No metric file at {}".format(hist_json))
            continue
        
        for j, (key, vals) in enumerate(hist.items()):
            axs[j].plot(vals, color=colors[i], label=model)
            axs[j].set_title(key)
            seq_len = len(vals)
            if seq_len>axs_lengths[j]:
                axs_lengths[j] = seq_len
                axs[j].set_xlim([offset, axs_lengths[j]])
            
    plt.legend()             
    return fig
    
def generate_confusion_plot(experiment_name, model_name, split='test', savepath=None, bins=50, figside=8,\
                            bounds=[1,5], log=False, verbose=True):
    if verbose:
        print("Loading model info {}/{}...".format(experiment_name,model_name))
    model_hist, model_spec, model_state = get_model_hist_spec_state(model_name, experiment_name)
    feature_df = pd.read_csv(model_spec['data_path'])
    
    if verbose:
        print("Building {} dataloader from {}...".format(split, model_spec['data_path']))
    dataloader = model_funcs.create_dataloader(feature_df[feature_df['split']==split], log=log)

    features, labels = next(iter(dataloader))
    input_height = features.size()[2]
    input_width = features.size()[3]
    
    if verbose:
        print("Loading model weights from {}...".format(model_state))
    model = model_funcs.Baseline_Model((input_height, input_width)).to(device)
    model.load_state_dict(torch.load(model_state, map_location=device))

    
    
    results = {}
    if verbose:
        print("Running predictions...")
        
    truth_list = []
    pred_list = []
    for (x, y) in dataloader:
        (x, y) = (x.to(device), y.to(device))
        pred = model(x)
        
        
        if log:
            true = y.item()
            pred = pred.item()
        else:
            true = np.log10(y.item())
            pred = np.log10(pred.item())
                
        truth_list.append(true)
        pred_list.append(pred)

    if verbose:
        print("Plotting {} points...".format(len(truth_list)))
    
    
    fig = plt.figure(figsize=(figside,figside))
    bounds_pair = [bounds, bounds]
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    axHistx.xaxis.set_label_position("top")
    axHistx.set_xlabel('Ground Truths')
    
    axHisty.yaxis.set_label_position("right")
    axHisty.set_ylabel('Predictions', rotation=270)

    axHistx.set_xticks([])
    axHistx.set_yticks([])
    axHisty.set_xticks([])
    axHisty.set_yticks([])
    
    axHistx.set_xlim(bounds)
    axHisty.set_ylim(bounds)

    _, xedge, yedge, _ = axScatter.hist2d(truth_list, pred_list, bins=bins, range=bounds_pair)
    axScatter.plot(bounds, bounds, 'r--')

    _ = axHistx.hist(truth_list, bins=bins, range=bounds)
    _ = axHisty.hist(pred_list, bins=bins, range=bounds, orientation='horizontal')
    plt.show()
    
    return fig

#metrics functions
def MSE(output, target):
    loss = torch.mean((output - target)**2)
    return loss

def Bias(output, target):
    loss = torch.mean(output - target)
    return loss

def CovStep(output, target, output_mean, target_mean):
    loss = torch.mean(((output - output_mean) * (target - target_mean)))
    return loss

def MeanAbsLogStep(output, target, log=True):
    #convert out of log
    if log:
        vol_pred = 10**output
        vol_target = 10**target
    else:
        vol_pred = output
        vol_target = target
    loss = torch.mean(torch.abs(torch.log(torch.abs(vol_pred/vol_target))))
    return loss

def compute_eval_metrics(dataloader, model, log=True):
    target_sum = 0
    pred_sum = 0
    n_steps = 0
    
    for (x,y) in dataloader:        
        (x, y) = (x.to(device), y.to(device))
        pred = model(x)
        target_sum += y.sum()
        pred_sum += pred.sum()
        n_steps += 1
    
    target_mean = target_sum/n_steps
    pred_mean = pred_sum/n_steps
    
    mse = 0
    mean_error = 0
    cov = 0
    abs_log_ratio = 0
    
    var_pred = 0 #technically var * N but gets cancelled out in Pearson calculation
    var_target = 0 
    
    for (x,y) in dataloader:        
        (x, y) = (x.to(device), y.to(device))
        pred = model(x)
        mse += MSE(pred, y)
        mean_error += Bias(pred, y)
        cov += CovStep(pred, y, pred_mean, target_mean)
        abs_log_ratio += MeanAbsLogStep(pred, y, log=log)
        
        var_pred += MSE(pred, pred_mean)
        var_target += MSE(y, target_mean)    
    
    out_dict = {}
    out_dict['mse'] = (mse / n_steps).item()
    out_dict['bias'] = (mean_error / n_steps).item()
    out_dict['pearson_cor'] = (cov/(torch.sqrt(var_pred) * torch.sqrt(var_target))).item()
    out_dict['mean_mult'] = (torch.exp(abs_log_ratio/n_steps)).item()
    out_dict['var_ratio'] = (torch.sqrt(var_pred) / torch.sqrt(var_target)).item()
    
    return out_dict