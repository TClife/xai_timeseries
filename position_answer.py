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
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type =str,default='mitbih, flat, peak')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--vqvae_model', type=str, default ="/home/hschung/saved_models/timeseries/flat/model_360.pt") 
    parser.add_argument('--model_type',type =str,default='cnn_transformer', help='cnn_transformer, transformer, cnn')
    parser.add_argument('--device', type =str,default='3')
    parser.add_argument('--auc_classification', type=bool, default=True)
    parser.add_argument('--classification_model', type=str)
    parser.add_argument('--mask_code', default=50)
    parser.add_argument('--len_position', type=int, default=13)
    parser.add_argument('--num_quantizers', type=int, default=[1,2,4,8])
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
    
    conv_net = VQ_Classifier(
        num_classes = 2,
        vqvae_model = vqvae_model,
        positions =0,
        mask = 0,
        auc_classification = False,
        model_type = classifier
    ).to(device)

    a = torch.load(classification_model)
    conv_net.load_state_dict(a['model_state_dict'])

    for param in conv_net.parameters():
        param.requires_grad = False

    conv_net.eval()
        
    masked_regions = conv_net.mask_region(ds.data[:64, :].to(device))
    
    
    #Make position answer
    answer = {}         
    arr = [i for i in range(len_position)]
    roc_auc_scores=[]


    for cnt in range(len_position):
        max_auc=-1
        print("arr:", arr)
        for i in arr:
            #ECG Dataset
            positions = list(answer.values())+[i]
            net = VQ_Classifier(
                num_classes = 2,
                vqvae_model = vqvae_model,
                positions = positions,
                mask = masked_regions,
                auc_classification = True,
                model_type = classifier
            ).to(device)
            
            
            net.load_state_dict(torch.load(classification_model)["model_state_dict"])
            net = net.to(device)
            for param in net.parameters():
                param.requires_grad = False        
            net.eval()   

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
                
    



    