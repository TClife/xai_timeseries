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
from data import load_data
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_num_threads(32) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(112)

if __name__ =='__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--labels', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--dataset', type =str,default='ptb')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--vqvae_model', default = "/home/hschung/xai/xai_timeseries/saved_models/hard_mitbih/8/model_290.pt")
    parser.add_argument('--classification_model', type=str, default="/home/hschung/xai/xai_timeseries/saved_models/classification/mitbih/8/cnn.pt")
    parser.add_argument('--model_type',type =str,default='cnn', help='cnn_transformer, transformer, cnn')
    parser.add_argument('--device', type =str,default= "7")
    parser.add_argument('--auc_classification', type=bool, default=True)
    parser.add_argument('--len_position', type=int, default=12)
    parser.add_argument('--num_quantizers', type=int, default=8)
    
    #selected position and quantizer 
    parser.add_argument('--selected_position', type=int, default=0)
    parser.add_argument('--selected_quantizer', type=int, default=7) 
    parser.add_argument('--decode', type=str, default="perturbations", help="Either save the dataset codes, or decode the perturbed codes") 
    
    args = parser.parse_args()
    
    len_position=args.len_position
    dataset =args.dataset
    classifier = args.model_type
    classification_model =  args.classification_model
    vqvae_model = args.vqvae_model
    ds = load_data(args.dataset, task = 'xai')
    
    train_size = int(0.8 * len(ds))
    val_size = int(0.1 * len(ds))
    test_size = len(ds) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, pin_memory=True)
    
    #Find masking token
    end_tokens={}

    net = VQ_Classifier(
        num_classes = 2,
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
        
    masked_regions = net.mask_region(ds.data[:64, :].to(device))
    
    net = VQ_Classifier(
    num_classes = 2,
    vqvae_model = vqvae_model,
    positions =0,
    mask = masked_regions,
    auc_classification = False,
    model_type = classifier,
    len_position = len_position
    ).to(device)
    
    a = torch.load(classification_model)
    net.load_state_dict(a['model_state_dict'])

    for param in net.parameters():
        param.requires_grad = False
    
    net.eval()
    
    
    with torch.no_grad():    
        for sample_num, (data, labels) in enumerate(test_loader):
            data = data.unsqueeze(1).float()
            data= data.to(device)
            if args.decode == "perturbations":
                selected_positions = net.masking_position(data, args.selected_position, args.selected_quantizer, sample_num)
            elif args.decode == "dataset":
                selected_positions = net.original_decode(data, sample_num, labels)
                
    
                
    



    