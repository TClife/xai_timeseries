import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from torch.utils.data import Dataset, random_split
import numpy as np
from scipy.io.arff import loadarff
import argparse
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt 
import os
import wandb


def makedata(data_name):
    print(f"dataset is {data_name}")
    if data_name=='peak':
        class0_pth = '/home/smjo/xai_v0/timeseries_xai/data/made_data/class0.csv'
        class1_pth = '/home/smjo/xai_v0/timeseries_xai/data/made_data/class1.csv'
    elif data_name=='flat':
        class0_pth = '/home/smjo/xai/timeseries_xai/data/made_data/amplitude_class0.csv'
        class1_pth = '/home/smjo/xai/timeseries_xai/data/made_data/amplitude_class1.csv'
    
    elif data_name == 'flat4':
        class0_pth = '/home/smjo/xai/timeseries_xai/data/made_data/flat1_class0.csv'
        class1_pth = '/home/smjo/xai/timeseries_xai/data/made_data/flat2_class1.csv'
        
    elif data_name=='hard_mitbih':
        
        class0_pth= "/home/hschung/xai_workshop/timeseries/data/new_data/new_n_data.pt"
        class0_label_pth = '/home/hschung/xai_workshop/timeseries/data/new_data/new_n_labels.pt'
        class1_pth = "/home/hschung/xai_workshop/timeseries/data/new_data/new_s_data.pt"
        class1_label_pth = "/home/hschung/xai_workshop/timeseries/data/new_data/new_s_labels.pt"

    elif data_name =='mitbih cherry_pick':
        class0_pth = '/home/smjo/xai/timeseries_xai/data/f_dataset.pt'
        class1_pth ='/home/smjo/xai/timeseries_xai/data/q_dataset.pt'
        class0_label_pth ='/home/smjo/xai/timeseries_xai/data/f_label.pt'
        class1_label_pth ='/home/smjo/xai/timeseries_xai/data/q_label.pt'       #label-3 
         
    elif data_name == "mitbih_answer_two":
         #make zero dataset except designated position
        new_data = torch.zeros_like(data)
        new_data[:, 32:49] = data[:, 32:49]
        data = new_data
    else:
        print("wrong data name")
    
    if data_name == 'peak' or data_name=='flat' or data_name=='flat4':
        class0_data = torch.tensor(pd.read_csv(class0_pth,skiprows=0).values)
        class1_data = torch.tensor(pd.read_csv(class1_pth,skiprows=0).values)
        class0_label = torch.zeros(len(class0_data))
        class1_label = torch.ones(len(class1_data))
    elif data_name =='hard_mitbih':
        class0_data = torch.tensor(torch.load(class0_pth))
        class1_data = torch.tensor(torch.load(class0_pth))
        class0_label = torch.tensor(torch.load(class0_label_pth))
        class1_label = torch.tensor(torch.load(class1_label_pth))
        
    
    data = torch.cat((class0_data,class1_data), dim=0)
    y = torch.cat((class0_label, class1_label), dim=0)

    X = data[:,:192]

    padding=(0,16)
    data = F.pad(X,padding,"constant",0)
    
    print(f"X shape:{data.shape}, y shape:{y.shape}")
    
    class ECGDataset(Dataset):
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
            
    ds = ECGDataset(data, y)
 
    return ds