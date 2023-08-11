import pandas as pd
import torch 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score as acc
import numpy as np
from lime import explanation
from lime import lime_base
import math
from time_lime import LimeTimeSeriesExplainer
from data import load_data
from resnet import resnet34
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset, random_split
import torch.optim as optim 
import copy
import argparse

import os
import sklearn 
import scikitplot as skplt

torch.set_num_threads(32)
torch.manual_seed(911) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser() 
parser.add_argument('--classification_model_pth', type=str, default="/home/smjo/xai_timeseries/classification_models/toydata2/time/resnet.pt")
parser.add_argument('--dataset', type=str, choices=['toydata', 'toydata2', 'toydata3'])
parser.add_argument('--num_slices', type=int, choices=[12,24,48,96])
args = parser.parse_args()
batch_size=1

if args.dataset =='toydata2':
    train_ds = load_data(args.dataset, task='xai',set='train', domain='time')
    test_ds = load_data(args.dataset, task='xai',set='test', domain='time')
    val_ds = load_data(args.dataset, task='xai',set='val', domain='time')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle=False, pin_memory=True)

else:
    ds = load_data(args.dataset,task='xai',set=None,domain='time')
    train_size = int(0.8 * len(ds))
    val_size = int(0.1 * len(ds))
    test_size = len(ds) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])
    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

#load classification args 
classification_model = torch.load(args.classification_model_pth)
class_args = classification_model['args']

net = resnet34(class_args).to(device)
net.load_state_dict(classification_model['model_state_dict'])

for param in net.parameters():
    param.requires_grad = False 
    
idx = 5 # explained instance
num_slices = args.num_slices# split time series
num_features = int(192/num_slices) # how many feature contained in explanation


index = list(range(num_slices))    
num_indices = len(index)

weight_dict={ i:0 for i in range(num_slices)}

for k, (data, labels) in enumerate(test_loader):
    #codebook
    net.eval()
    net = net.to(device)
    data = data.unsqueeze(0).to(device)
    y_hat,prob, _,_,_ = net(data)
    # Explain ECG Dataset
    len_ts = data.shape[2]
    
    #Number of perturb indices
    
    

    explainer = LimeTimeSeriesExplainer(class_names =['Class0', 'Class1'])
    exp = explainer.explain_instance(data, net, num_features=num_slices, num_samples=5000, num_slices=num_slices,len_ts=192,
                                    replacement_method='total_mean')
    max = -1e-10
    top_important = 0
    for i in range(num_slices):
        feature, weight = exp.as_list()[i]
        if weight >  max:
            max = weight
            top_important = feature
    weight_dict[top_important] +=1
            

position_rank = sorted(weight_dict.items(), key=lambda x:x[1], reverse=True)
print(dict(position_rank).keys())




####### DRAW picture of result ###############3
os.makedirs(f'/home/smjo/xai_timeseries/lime_result/{args.dataset}', exist_ok=True)
values_per_slice = values_per_slice = math.ceil(192 / num_slices)
class0=[]
class1=[]
for k, (data, labels) in enumerate(test_loader):
    labels = torch.argmax(labels)
    if labels==0:
        class0.append(data)
    else:
        class1.append(data)
class0 = torch.stack(class0)
class1 = torch.stack(class1)
class0_mean = torch.mean(class0, 0)
class1_mean = torch.mean(class1, 0)

plt.plot(class0_mean.reshape(193,-1), color='blue', label='Mean of class 0')
plt.plot(class1_mean.reshape(193,-1), color='red', label='Mean of class 1')
for pos,cnt in weight_dict.items():
    start = pos * values_per_slice
    end = start + values_per_slice
    plt.axvspan(start , end, color='green', alpha=abs(cnt*0.001))
plt.legend(loc='lower left')

plt.savefig(f'/home/smjo/xai_timeseries/lime_result/{args.dataset}/{num_slices}.png')