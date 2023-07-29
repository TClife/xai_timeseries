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
from models import VQ_Classifier, resnet34, resnet34_raw
import itertools
from data import load_data
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_num_threads(32) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(911)

if __name__ =='__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--labels', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--dataset', type=str, default="toy_dataset")
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--classification_model', type=str, default="/home/hschung/xai/xai_timeseries/classification_models/toy_dataset/8/resnet.pt")
    parser.add_argument('--vqvae_model', default = "/home/hschung/xai/xai_timeseries/saved_models/toy_dataset/8/model_2170.pt") 
    parser.add_argument('--model_type',type =str,default='resnet', help='cnn_transformer, transformer, cnn, resnet, raw')
    parser.add_argument('--device', type =str,default='3')
    parser.add_argument('--auc_classification', type=bool, default=False)
    parser.add_argument('--len_position', type=int, default=12)
    parser.add_argument('--num_quantizers', type=int, default=8)
    parser.add_argument('--positions', type=int, default=0)
    parser.add_argument('--mask', type=int, default=0)
    parser.add_argument('--num_features', type=int, default=1)
    args = parser.parse_args()
    
    

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    len_position=args.len_position
    dataset =args.dataset
    classifier = args.model_type
    classification_model = args.classification_model
    vqvae_model = args.vqvae_model
    num_features = args.num_features
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

    if args.model_type == "resnet":
        net = resnet34(
            num_classes = args.num_classes,
            vqvae_model = args.vqvae_model,
            positions = args.positions,
            mask = args.mask,
            auc_classification = args.auc_classification,
            model_type = args.model_type,
            len_position = args.len_position
        ).to(device)
        
        a = torch.load(classification_model)
        net.load_state_dict(a['model_state_dict'])

        for param in net.parameters():
            param.requires_grad = False
            
        masked_regions = net.mask_region(ds.data[:64, :].to(device))
        
        net = resnet34(
        num_classes = args.num_classes,
        vqvae_model = vqvae_model,
        positions =0,
        mask = masked_regions,
        auc_classification = False,
        model_type = classifier,
        len_position = len_position
        ).to(device)
        
    elif args.model_type == "raw":
        net = resnet34_raw(
            num_classes = args.num_classes,
            vqvae_model = args.vqvae_model,
            positions = args.positions,
            mask = args.mask,
            auc_classification = args.auc_classification,
            model_type = args.model_type,
            len_position = args.len_position
        ).to(device)

        for param in net.parameters():
            param.requires_grad = False
        
    else:
        net = VQ_Classifier(
            num_classes = args.num_classes,
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
        num_classes = args.num_classes,
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
    insertion_selected_positions = []
    deletion_selected_positions = []
    combined_selected_positions = []
    
    for i in range(len_position // num_features):
        with torch.no_grad():    
            for _, (data, labels) in enumerate(test_loader):
                data = data.unsqueeze(1).float()
                labels = labels.type(torch.LongTensor)
                data, labels = data.to(device), labels.to(device)
                insertion_selected_positions, insertion_combined_scores, insertion_combinations = net.position_answer(data, i, labels, insertion_selected_positions, num_features)
                deletion_selected_positions, deletion_combined_scores, deletion_combinations = net.position_answer_deletion(data, i, labels, deletion_selected_positions, num_features)
                
                #combined metric 
                combined_selected_position = int(torch.argmax(insertion_combined_scores - deletion_combined_scores))
                assert insertion_combinations == deletion_combinations
                combined_selected_positions.append(int(insertion_combinations[combined_selected_position][0]))
                
                #change both insertion and deletion selected positions to combined 
                insertion_selected_positions = list(combined_selected_positions)
                deletion_selected_positions = list(combined_selected_positions)
                
    
    print(combined_selected_positions)
    