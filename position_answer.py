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
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
torch.set_num_threads(32) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(911)


if __name__ =='__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--labels', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
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
    classification_model =  args.classification_model
    vqvae_model = args.vqvae_model
    ds = makedata(dataset)
    
    '''
    if dataset == 'flat':
        vqvae_model = "/home/hschung/saved_models/timeseries/flat/model_360.pt"
        if classifier == 'cnn_transformer':
            classification_model ="/home/hschung/saved_models/timeseries/flat_clas_cnn_transf/classification_60.pt"
        elif classifier == 'transformer':
            classification_model="/home/hschung/saved_models/timeseries/flat_clas_transf/classification_90.pt"
        elif classifier == 'cnn':
            classification_model = "/home/hschung/saved_models/timeseries/flat_clas/classification_60.pt"
            
            
    elif dataset=='peak':
        vqvae_model="/home/hschung/saved_models/timeseries/peak2/model_2990.pt"
        if classifier == 'cnn_transformer':
            classification_model = "/home/hschung/saved_models/timeseries/peak_clas_cnn_transf/classification_80.pt"
        elif classifier =='transformer':
            classification_model = "/home/hschung/saved_models/timeseries/peak_clas_transf/classification_80.pt"
        elif classifier =='cnn':
            classification_model ="/home/hschung/saved_models/timeseries/peak_clas/classification_170.pt"
            
            
    elif dataset =='mitbih':
        vqvae_model ="/home/hschung/saved_models/timeseries/hard_mitbih_sf/model_1170.pt"
        if classifier == 'cnn_transformer':
            classification_model="/home/hschung/saved_models/timeseries/hard_mitbih_clas_cnn_transf/classification_190.pt"
        elif classifier == 'transformer':
            classification_model="/home/hschung/saved_models/timeseries/hard_mitbih_clas_transf/classification_40.pt" 
        elif classifier == 'cnn':
            classification_model ="/home/hschung/saved_models/timeseries/classification_model_hard_mitbih/classification_110.pt"

    elif dataset == 'ptb':
        vqvae_model =  "/home/hschung/saved_models/timeseries/ptb/model_780.pt"
        if classifier == 'cnn':
            classification_model="/home/hschung/saved_models/timeseries/ptb_cnn/classification_870.pt"
    else:
        print("wrong name")
    '''
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
        num_classes = args.num_classes,
        vqvae_model = args.vqvae_model,
        positions = args.positions,
        mask = args.mask,
        auc_classification = True,
        model_type = args.model_type
    ).to(device)
    
    a = torch.load(args.classification_model)
    conv_net.load_state_dict(a['model_state_dict'])
    
    for param in conv_net.parameters():
        param.requires_grad = False
    
    conv_net.eval()
    
    with torch.no_grad():
        for idx, (data, labels) in enumerate(test_loader):
            data = data.unsqueeze(1).float()
            labels = labels.type(torch.LongTensor)
            data, labels = data.to(device), labels.to(device)
            output, codebook_tokens, recon, input= conv_net(data)
            print(f"codebook indices:{codebook_tokens}")
            for token in codebook_tokens:
                try:
                    end_tokens[token[-1].item()]+=1
                except:
                    end_tokens[token[-1].item()]=1
                    
        mask_code = max(end_tokens, key=end_tokens.get)
        print(f"mask code is {mask_code} ")
    
    
    
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
            net = VQVAE_Conv(
                n_emb = 64,
                num_classes =args.num_classes,
                vqvae_model=vqvae_model,
                positions = positions,
                mask = mask_code,
                auc_classification=True,
                model_type = classifier
            )
            print(positions)
            classification_model = classification_model
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
                    data, labels = data.to(device), labels.to(device)
                    output, codebook_tokens, recon, input= net(data)
                    score += sklearn.metrics.roc_auc_score(labels.cpu(), F.softmax(output, dim=1)[:,1].cpu())
                    
            score = score / (idx+1)
            if score > max_auc:
                answer[f"top_{cnt}"]=i
                max_auc = copy.deepcopy(score)
        roc_auc_scores.append(max_auc)        
        arr.remove(answer[f"top_{cnt}"])
        
    
print("answer:", answer.values())
print(roc_auc_scores)
    
            
    



    