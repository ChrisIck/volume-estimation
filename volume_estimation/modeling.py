import numpy as np
import pandas as pd
import torch
import os
import json
from torch.utils import data
from torch.nn import Conv2d, AvgPool2d, ReLU, Dropout, Flatten, Linear, Sequential, Module, MSELoss
from torch.optim import Adam 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from time import time


from volume_estimation import evaluation

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

torch.set_default_dtype(torch.float64)

#dataloader generator
def create_dataloader(feature_df, batch_size=1, targets=['vol']):
    dataset = []
    for row in feature_df.iterrows():
        feat_file = row[1]['file_feature']
        loaded = np.load(feat_file)

        feature = loaded['feat']
        feature = feature.reshape((1, feature.shape[0], feature.shape[1]))
        feature = np.real(feature)
        
        outputs = np.empty(len(targets))
        for i, target in enumerate(targets):
            val = row[1][target]
            outputs[i] = val
        dataset.append((feature, outputs))
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

#training loop
def train_model(model_func, model_dict, targets=['vol'], batch_size=16, lr_init=1e-3, l2_reg=1e-3, overwrite=False,
                epochs=1000, log=True, sched_thres=1e-4, normalize_targets=False):

    model_path = model_dict['model_path']
    if not os.path.exists(model_path):
        print("Saving model at ", model_path)
        os.makedirs(model_path)
    else:
        if overwrite:
            print("Overwriting model at ", model_path)
        else:
            print("Model exists at ", model_path)

    with open(os.path.join(model_path, model_dict['name']+'_spec.json'), 'w') as f:
        json.dump(model_dict, f)        
    
    print("\nLoading data from ", model_dict['data_path'])
    feat_df = pd.read_csv(model_dict['data_path'])
    
    print("Dataset size: {}".format(feat_df.shape))
    
    if log:
        for target in targets:
            feat_df[target] = np.log10(feat_df[target])
            
    if normalize_targets:
        for target in targets:
            feat_df[target] = feat_df[target]/feat_df[target].max()
            
    
    train_df = feat_df[feat_df['split']=='train']
    val_df = feat_df[feat_df['split']=='val']
    test_df = feat_df[feat_df['split']=='test']
    
    print("Creating training dataloader")
    train_dataloader = create_dataloader(train_df, batch_size=batch_size, targets=targets)

    print("Creating validation dataloader")
    val_dataloader = create_dataloader(val_df, targets=targets)

    print("Creating test dataloader")
    test_dataloader = create_dataloader(test_df, targets=targets)
    
    del feat_df
    
    features, labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {features.size()}")
    print(f"Labels batch shape: {labels.size()}")
    
    input_height = features.size()[2]
    input_width = features.size()[3]
    
    model_state_path = os.path.join(model_path, 'model_state.pt')
    opt_state_path = os.path.join(model_path, 'opt_state.pt')
    hist_path = os.path.join(model_path,'hist.json')
    
    n_out = len(targets)
    model = model_func((input_height, input_width), n_out=n_out).to(device)
    opt = Adam(model.parameters(),lr=lr_init, weight_decay=l2_reg)
    scheduler = ReduceLROnPlateau(opt, 'min', threshold=sched_thres)
    hist = {
        "duration": [],
        "train_loss": [],
    }
    
    for target in targets:
        hist['val_'+target+ "_loss"] = []
        hist['val_'+target+ "_bias"] = []
        hist['val_'+target+ "_pearson_cor"] = []
        hist['val_'+target+ "_mean_mult"] = []
        hist['val_'+target+ "_var_ratio"] = []
    
    lr = lr_init
    print("Beginning training for {} epochs...".format(epochs))
    for ep in range(epochs):
        t_start = time()
        model.train()

        train_loss = 0
        val_loss = 0
        train_steps = 0
        val_steps = 0

        for (x, y) in train_dataloader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            loss = evaluation.MSE(pred, y, is_loss=True)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss
            train_steps += 1

        with torch.no_grad():
            model.eval()

            val_metrics = evaluation.compute_eval_metrics(val_dataloader, model, log=log)
        
        #update LR scheduler
        scheduler.step(np.sum(val_metrics['mse']))
        
        t_end = time()
        t_elapsed = t_end - t_start
        
        hist['duration'].append(t_elapsed)
        hist['train_loss'].append(train_loss.item()/train_steps)
        for i, target in enumerate(targets):
            hist['val_'+target + '_loss'].append(val_metrics['mse'][i])
            hist['val_'+target + '_bias'].append(val_metrics['bias'][i])
            hist['val_'+target + '_pearson_cor'].append(val_metrics['pearson_cor'][i])
            hist['val_'+target + '_mean_mult'].append(val_metrics['mean_mult'][i])
            hist['val_'+target + '_var_ratio'].append(val_metrics['var_ratio'][i])

        
        print("E: {}\tD: {:.2f}s\tTL: {}\tVL: {}\tVB:{}\tVP: {}\tVM: {}\tVV:{}"\
              .format(ep, t_elapsed, (train_loss/train_steps).detach()[0], val_metrics['mse'],\
                      val_metrics['bias'], val_metrics['pearson_cor'],\
                      val_metrics['mean_mult'], val_metrics['var_ratio']))
        
        #save stuff
        torch.save(model.state_dict(), model_state_path)
        torch.save(opt.state_dict(), opt_state_path)
        with open(hist_path, 'w') as f:
            json.dump(hist, f)
            
    torch.cuda.empty_cache()
    print("Training complete, computing evaluation on datasets")
    test_metrics = evaluation.compute_eval_metrics(test_dataloader, model, log=log)
    with open(os.path.join(model_path, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f)
        
    val_metrics = evaluation.compute_eval_metrics(val_dataloader, model, log=log)
    with open(os.path.join(model_path, 'val_metrics.json'), 'w') as f:
        json.dump(val_metrics, f)
        
    train_dataloader = create_dataloader(train_df, batch_size=1, target=target)
    train_metrics = evaluation.compute_eval_metrics(train_dataloader, model, log=log)
    with open(os.path.join(model_path, 'train_metrics.json'), 'w') as f:
        json.dump(train_metrics, f)
    
    
    model_name = model_dict['model_path'].split('/')[-1]
    experiment_name = model_dict['model_path'].split('/')[-2]
    print("Generating confusion matrices...")
    _ = evaluation.generate_confusion_plot(experiment_name, model_name, dataloader=test_dataloader, log=log,\
                                           verbose=False, savepath=os.path.join(model_dict['model_path'], 'test_cm.pdf'))
    _ = evaluation.generate_confusion_plot(experiment_name, model_name, dataloader=val_dataloader, log=log,\
                                           verbose=False, savepath=os.path.join(model_dict['model_path'], 'val_cm.pdf'))
    _ = evaluation.generate_confusion_plot(experiment_name, model_name, dataloader=train_dataloader, log=log,\
                                           verbose=False, savepath=os.path.join(model_dict['model_path'], 'train_cm.pdf'))
                

#model definitions
class Baseline_Model(Module):
    def __init__(self, input_shape, n_out=1):
        #accepts a tuple with the height/width of the feature
        #matrix to set the FC layer dimensions
        super(Baseline_Model, self).__init__()
        
        #block1
        Conv1 = Conv2d(1, 30, kernel_size=(1,10), stride=(1,1))
        Avgpool1 = AvgPool2d((1,2), stride=(1,2))

        #block2
        Conv2 = Conv2d(30, 20, kernel_size=(1,10), stride=(1,1))
        Avgpool2 = AvgPool2d((1,2), stride=(1,2))

        #block3
        Conv3 = Conv2d(20, 10, kernel_size=(1,10), stride=(1,1))
        Avgpool3 = AvgPool2d((1,2), stride=(1,2))

        #block4
        Conv4 = Conv2d(10, 10, kernel_size=(1,10), stride=(1,1))
        Avgpool4 = AvgPool2d((1,2), stride=(1,2))

        #block5
        Conv5 = Conv2d(10, 5, kernel_size=(3,9), stride=(1,1))
        Avgpool5 = AvgPool2d((1,2), stride=(1,2))

        #block6
        Conv6 = Conv2d(5, 5, kernel_size=(3,9), stride=(1,1))
        Avgpool6 = AvgPool2d((2,2), stride=(2,2))

        #dropout
        dropout_layer = Dropout(p=0.5)
        height5 = input_shape[0] - 2
        height6 = (height5 - 2) // 2

        time1 = (input_shape[1] - 9) // 2
        time2 = (time1 - 9) // 2
        time3 = (time2 - 9) // 2
        time4 = (time3 - 9) // 2
        time5 = (time4 - 7) // 2
        time6 = (time5 - 7) // 2

        flat_dims = 5 * height6 * time6
        fc_layer = Linear(flat_dims, n_out)
        
        self.net = Sequential(
                    Conv1, ReLU(), Avgpool1,
                    Conv2, ReLU(), Avgpool2,
                    Conv3, ReLU(), Avgpool3,
                    Conv4, ReLU(), Avgpool4,
                    Conv5, ReLU(), Avgpool5,
                    Conv6, ReLU(), Avgpool6,
                    dropout_layer, Flatten(),
                    fc_layer, Flatten()
                )
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x