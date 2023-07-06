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
from argument import model_path


if __name__ =='__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--labels', type=int, default=1)
<<<<<<< HEAD
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type =str,default='mitbih, flat, peak')
=======
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--dataset', type =str,default='peak')
>>>>>>> f99e3e2aab7b23b4fe372d64b270fc26e9f1ed6c
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--vqvae_model', type=str, default ="/home/smjo/xai_timeseries/vqvae/saved_models/peak/2/model_280.pt") 
    parser.add_argument('--model_type',type =str,default='cnn', help='cnn_transformer, transformer, cnn')
    parser.add_argument('--device', type =str,default='3')
    parser.add_argument('--auc_classification', type=bool, default=True)
    parser.add_argument('--classification_model', type=str, default="/home/smjo/xai_timeseries/vqvae/saved_models/classification/peak/2/cnn.pt")
    parser.add_argument('--len_position', type=int, default=12)
    parser.add_argument('--num_quantizers', type=int, default=2)
    args = parser.parse_args()
    
    

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    len_position=args.len_position
    dataset =args.dataset
    classifier = args.model_type
    classification_model, vqvae_model = model_path(args.dataset, args.model_type, args.num_quantizers)
    ds = load_data(args.dataset, task = 'classification')
    
    train_size = int(0.8 * len(ds))
    val_size = int(0.1 * len(ds))
    test_size = len(ds) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, pin_memory=True)
    
    #Find masking token
    end_tokens={}
<<<<<<< HEAD
    
    conv_net = VQ_Classifier(
=======

    net = VQ_Classifier(
>>>>>>> f99e3e2aab7b23b4fe372d64b270fc26e9f1ed6c
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

<<<<<<< HEAD
            with torch.no_grad():    
                score=0
                for idx, (data, labels) in enumerate(test_loader):
                    data = data.unsqueeze(1).float()
                    labels = labels.type(torch.LongTensor)
                    labels = torch.argmax(labels, dim=1)
                    data, labels = data.to(device), labels.to(device)
                    output, codebook_tokens, recon, input= net(data)
                
                    score += sklearn.metrics.roc_auc_score(labels.cpu(), output[:,1].cpu())
                    
            score = score / (idx+1)
            if score > max_auc:
                answer[f"top_{cnt}"]=i
                max_auc = copy.deepcopy(score)
        roc_auc_scores.append(max_auc)        
        arr.remove(answer[f"top_{cnt}"])
        
    print(f"roc_auc_scores:{roc_auc_scores}")
    print(answer.values())
=======
    for param in net.parameters():
        param.requires_grad = False
    
    net.eval()
    selected_positions = []
    
    for i in range(len_position):
        with torch.no_grad():    
            for _, (data, labels) in enumerate(test_loader):
                data = data.unsqueeze(1).float()
                labels = labels.type(torch.LongTensor)
                labels = torch.argmax(labels, dim=1)
                data, labels = data.to(device), labels.to(device)
                selected_positions = net.position_answer(data, i, labels, selected_positions)
    
    print(selected_positions)
>>>>>>> f99e3e2aab7b23b4fe372d64b270fc26e9f1ed6c
                
    



    