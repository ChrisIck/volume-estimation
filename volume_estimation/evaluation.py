import os
import json
import sys

from volume_estimation import modeling
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm 

from tqdm import tqdm


MODELS_DEFAULT = '/scratch/ci411/sonos_rirs/models/'
FIG_DEFAULT = '/home/ci411/volume_estimation/figures/'

def get_model_hist_spec_state(model_name, experiment_name, models_dir=MODELS_DEFAULT):
    hist_dir = os.path.join(models_dir, experiment_name, model_name, 'hist.json')
    json_spec = os.path.join(models_dir, experiment_name, model_name, model_name+'_spec.json')
    model_state = os.path.join(models_dir, experiment_name, model_name, 'model_state.pt')
    
    if os.path.exists(hist_dir):
        with open(hist_dir) as f:
            hist = json.load(f)
    else:
        print("No history file at {}".format(hist_dir))
        hist = None

    with open(json_spec) as f:
        spec = json.load(f)
        
    return hist, spec, model_state

def plot_experiment_metrics(experiment_name, model_names=None, split='test',models_dir=MODELS_DEFAULT):
    experiment_dir = os.path.join(models_dir, experiment_name)
    
    if model_names is None:
        model_names = os.listdir(experiment_dir)
        model_names.sort()
            
    n = len(model_names)
    width = 0.5/n
    
    fig, axs = plt.subplots(1,5, figsize=(12,6))
    fig.tight_layout()

    for i, model in enumerate(model_names):
        json_metric = os.path.join(experiment_dir, model, '{}_metrics.json'.format(split))
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

def plot_experiment_curves(experiment_name, model_names=None, offset=100, cmap='coolwarm',\
                           verbose=True, max_len=1000, models_dir=MODELS_DEFAULT):
        
    experiment_dir = os.path.join(models_dir, experiment_name)
    
    if model_names is None:
        model_names = os.listdir(experiment_dir)
        model_names.sort()
            
    n = len(model_names)
    width = 0.5/n
    cmap = cm.ScalarMappable(cmap=cmap)
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
            print("No history file at {}".format(hist_json))
            continue
        
        n_epochs = len(hist['duration'])
        if verbose:
            print("Model {} has completed {} epochs".format(model, n_epochs))
        for j, (key, vals) in enumerate(hist.items()):
            sub_vals = vals[offset:] #subset to plot
            epochs = np.linspace(offset, len(vals)-1, len(sub_vals))
            axs[j].plot(epochs, sub_vals, color=colors[i], label=model)
            axs[j].set_title(key)
            #if seq_len>axs_lengths[j]:
            #    axs_lengths[j] = seq_len

    for i in range(7):          
        axs[i].set_xlim([offset, max_len])
            
    plt.legend()             
    return fig
    
def generate_confusion_plot(experiment_name, model_name, dataloader=None, split='test', savepath=None, bins=50, figside=8,\
                            bounds=None, log=False, verbose=True, target='vol', logscale=True):
    model_hist, model_spec, model_state = get_model_hist_spec_state(model_name, experiment_name)
    
    if dataloader is None:
        if verbose:
            print("Loading model info {}/{}...".format(experiment_name,model_name))
        feature_df = pd.read_csv(model_spec['data_path'])

        if verbose:
            print("Building {} dataloader from {}...".format(split, model_spec['data_path']))
        dataloader = modeling.create_dataloader(feature_df[feature_df['split']==split], log=log, target=target)
        
    features, labels = next(iter(dataloader))
    input_height = features.size()[2]
    input_width = features.size()[3]
    
    if verbose:
        print("Loading model weights from {}...".format(model_state))
    model = modeling.Baseline_Model((input_height, input_width)).to(device)
    model.load_state_dict(torch.load(model_state, map_location=device))

    
    
    results = {}
    if verbose:
        print("Running predictions...")
        
    truth_list = []
    pred_list = []
    for (x, y) in tqdm(dataloader):
        (x, y) = (x.to(device), y.to(device))
        pred = model(x)
        
        
        if log:
            true = y.item()
            pred = pred.item()
        else:
            if logscale:
                true = np.log10(y.item())
                pred = np.log10(pred.item())
            else:
                true = y.item()
                pred = pred.item()
                
        truth_list.append(true)
        pred_list.append(pred)

    if verbose:
        print("Plotting {} points...".format(len(truth_list)))
    
    
    fig = plt.figure(figsize=(figside,figside))
    if bounds is None:
        if target == 'vol':
            bounds = [1,5]
        elif target == 'rt60':
            bounds = [-.5,1.5]
    
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
    
    print("Plotting...")
    _, xedge, yedge, _ = axScatter.hist2d(truth_list, pred_list, bins=bins, range=bounds_pair)
    axScatter.plot(bounds, bounds, 'r--')

    _ = axHistx.hist(truth_list, bins=bins, range=bounds)
    _ = axHisty.hist(pred_list, bins=bins, range=bounds, orientation='horizontal')
    
    
    if savepath is not None:
        print("Saving fig to {}".format(savepath))
        plt.savefig(savepath)
    
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

def compute_eval_metrics(dataloader, model, log=True, verbose=False):
    target_sum = 0
    pred_sum = 0
    n_steps = 0
    if verbose:
        print("Computing sums...")
        dataloader_iter = tqdm(dataloader)
    else:
        dataloader_iter = dataloader
    for (x,y) in dataloader_iter:        
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
    
    if verbose:
        print("Computing metrics...")
        dataloader_iter = tqdm(dataloader)
    else:
        dataloader_iter = dataloader
    for (x,y) in dataloader_iter:          
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

def evaluate_experiment(experiment_name, model_list=None, log=False, same_features=True,\
                        gen_plots=False, target='val', models_dir=MODELS_DEFAULT, fig_path=FIG_DEFAULT):
    
    if model_list is None:
        model_list = os.listdir(os.path.join(models_dir, experiment_name))
        model_list.sort()
    
    if same_features:
        _, model_spec, _ = get_model_hist_spec_state(model_list[0], experiment_name)
        feature_df = pd.read_csv(model_spec['data_path'])
        test_dataloader = modeling.create_dataloader(feature_df[feature_df['split']=='test'], log=log, target=target)

        features, labels = next(iter(test_dataloader))
        input_height = features.size()[2]
        input_width = features.size()[3]

        model = modeling.Baseline_Model((input_height, input_width)).to(device)
    
    for model_name in model_list:
        print("Loading {}".format(model_name))
        _, model_spec, model_state = get_model_hist_spec_state(model_name, experiment_name)
        
        if not same_features:
            print("Loading features from {}".format(model_spec['data_path']))
            feature_df = pd.read_csv(model_spec['data_path'])
            test_dataloader = modeling.create_dataloader(feature_df[feature_df['split']=='test'], log=log, target=target)
            features, labels = next(iter(test_dataloader))
            print('labels')
            input_height = features.size()[2]
            input_width = features.size()[3]
            model = modeling.Baseline_Model((input_height, input_width)).to(device)
        
        if os.path.exists(model_state):
            model.load_state_dict(torch.load(model_state, map_location=device))
        else:
            print("No weights found for {}".format(model_name))
            continue
        
        print("Computing metrics on {}".format(model_name))
        
        print("Loading test dataloader...")
        test_metrics = compute_eval_metrics(test_dataloader, model, log=log)
        test_path = os.path.join(models_dir, experiment_name, model_name, 'test_metrics.json')
        print('Saving metrics to {}'.format(test_path))
        with open(test_path, 'w') as f:
            json.dump(test_metrics, f)
        
        print("Loading val dataloader...")
        val_dataloader = modeling.create_dataloader(feature_df[feature_df['split']=='val'], log=log, target=target)
        val_metrics = compute_eval_metrics(val_dataloader, model, log=log)
        val_path = os.path.join(models_dir, experiment_name, model_name, 'val_metrics.json')
        print('Saving metrics to {}'.format(test_path))
        with open(val_path, 'w') as f:
            json.dump(val_metrics, f)   
        
        print("Loading train dataloader...")
        train_dataloader = modeling.create_dataloader(feature_df[feature_df['split']=='train'], log=log, target=target)
        train_metrics = compute_eval_metrics(train_dataloader, model, log=log)
        train_path = os.path.join(models_dir, experiment_name, model_name, 'train_metrics.json')
        print('Saving metrics to {}'.format(train_path))
        with open(train_path, 'w') as f:
            json.dump(train_metrics, f)
            
        
        if gen_plots:
            print("Generating confusion matrices...")
            model_fig_path = os.path.join(fig_path, experiment_name, model_name)
            if not os.path.exists(model_fig_path):
                os.makedirs(model_fig_path)
            _ = generate_confusion_plot(experiment_name, model_name, split='test', log=log, target=target,\
                                        verbose=False, savepath=os.path.join(model_fig_path, 'test_cm.pdf'))
            _ = generate_confusion_plot(experiment_name, model_name, split='val', log=log, target=target,\
                                        verbose=False, savepath=os.path.join(model_fig_path, 'val_cm.pdf'))
            _ = generate_confusion_plot(experiment_name, model_name, split='train', log=log,target=target,\
                                        verbose=False, savepath=os.path.join(model_fig_path, 'train_cm.pdf'))

    return None