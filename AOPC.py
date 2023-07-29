import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd
import argparse                 
import sklearn 
import numpy as np 
import logging 
import sklearn.metrics as metrics 
import scikitplot as skplt
import matplotlib.pyplot as plt                                  
from argparse import ArgumentParser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset, random_split
import torch.nn.functional as F
from collections import Counter, defaultdict
import itertools
import os 
import copy
import warnings
from dataload import makedata
from classifier import ClassifierTrainer
from models import VQ_Classifier
import itertools
import random
from torcheval.metrics.aggregation.auc import AUC
from data import load_data
torch.manual_seed(911) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 1
len_position=12
dataset = 'mitbih'
classifier = 'cnn'
classification_model = "/home/smjo/xai_timeseries/classification_models/ptb/8/cnn.pt"
vqvae_model = "/home/smjo/xai_timeseries/saved_models/mitbih/8/model_300.pt"

ds = load_data(dataset, task = 'classification')



test_size = 100
train_size = len(ds)-test_size
train_dataset,test_dataset = random_split(ds, [train_size,test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, pin_memory=True)

#Find masking token
end_tokens={}

net = VQ_Classifier(
    num_classes = 1,
    vqvae_model = vqvae_model,
    positions =0,
    mask = 0,
    auc_classification = False,
    model_type = classifier,
    len_position = len_position
).to(device)

a = torch.load(classification_model)
net.load_state_dict(a['model_state_dict'])

for param in net.parameters():
    param.requires_grad = False
    
    
pos_order_lime = [0,2,1,3,4,6,5,10,8,9,7,11]

pos_order_ours = [1,8,3,7,6,2,0,9,10,5,11,4]
pos_order_IG = [3,1,2,4,0,5,6,7,8,10,9,11]

pos_order_shap = [0,2,1,3,4,7,5,6,10,9,8,11]

net.eval()

for idx_xai,xai_model in enumerate((pos_order_lime,pos_order_ours, pos_order_IG, pos_order_shap)):
    total_delta = 0
    result=[]
    delta_result = {pos:0 for pos in range(len_position)}
    with torch.no_grad():    
        for _, (data, labels) in enumerate(test_loader):
            data = data.unsqueeze(1).float()
            labels = labels.type(torch.LongTensor)
            data, labels = data.to(device), labels.to(device)
            output, codebook_tokens, recon, input= net(data)
            codebook_tokens = codebook_tokens.cpu().detach().numpy()        #1,12,8
            
            #output position 순서대로 random vector로 바꾸기
            random_num = {idx:np.random.randint(128, size = (1,8)) for idx in range(len_position)}
            new_codebook = codebook_tokens.copy()
            for pos in xai_model:
                new_codebook[0,pos,:] = random_num[pos]
                new_output,_ = net.predict(new_codebook)
                delta = output.item() - new_output.item()
                delta_result[pos] += delta
                total_delta += delta
            
            #sum(original prediction - new prediction)
        total_delta  = total_delta / (len_position * test_size)
        for pos,delta in delta_result.items():
            result.append(delta/test_size)
        print(f"xai_model:{xai_model},total_delta:{total_delta}")
        
        plt.plot(result, label=str(idx_xai))
        plt.xlabel('num_position')
        plt.ylabel('AOPC')
        plt.savefig('./AOPC.png')
