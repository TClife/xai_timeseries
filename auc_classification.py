import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from models import VQ_Classifier, resnet34, resnet34_raw
from data import load_data
import argparse                 
import sklearn 
from sklearn import metrics
import numpy as np  
import os 
import matplotlib.pyplot as plt                                  
from argparse import ArgumentParser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_num_threads(32) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(911)

#Create args parser for labels and batch size 
parser = ArgumentParser() 
parser.add_argument('--labels', type=int, default=0)
parser.add_argument('--num_features', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=96)
parser.add_argument('--classification_model', type=str, default="/home/hschung/xai/xai_timeseries/classification_models/toy_dataset2/8/resnet.pt")
parser.add_argument('--vqvae_model', default = "/home/hschung/xai/xai_timeseries/saved_models/toy_dataset2/8/model_210.pt") 
parser.add_argument('--task', type=str, default='xai', help="Task being done")
parser.add_argument('--dataset', type=str, default="toy_dataset2")
parser.add_argument('--plot_dataset', type=bool, default=True)
parser.add_argument('--model_type', type=str, default="resnet")
parser.add_argument('--auc_classification', type=bool, default=False)
parser.add_argument('--auc_classification_eval', type=bool, default=True) 
parser.add_argument('--auc_classification_eval2', type=bool, default=False)
parser.add_argument('--len_position', type=int, default=12)
parser.add_argument('--method', default= ["LIME", "SHAP", "IG", "Ours"])
parser.add_argument('--position_ranking', default = [[1,2,7,6,4,5,0,9,10,3,8,11], [1,6,11,7,2,4,10,9,0,5,3,8], [0,1,2,3,8,7,9,11,10,4,6,5], [6, 2, 9, 4, 10, 0, 5, 11, 1, 8, 3, 7]])
args = parser.parse_args() 

#dataset split 
ds = load_data(args.dataset, args.task)

# Split the dataset into training, validation, and test sets
train_size = int(0.8 * len(ds))
val_size = int(0.1 * len(ds))
test_size = len(ds) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])
training_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

#loop over methods and position rankings
for method, position_ranking in zip(args.method, args.position_ranking):

    #get positions to be used as features
    new_list = []

    for i in range(len(position_ranking)):
        new_list.append(position_ranking[:i+1])

    if args.auc_classification_eval:
        new_list = new_list 
    elif args.auc_classification_eval2:
        new_list = [position_ranking[i:] for i in range(len(position_ranking))]

    print(new_list)
    #loop over new_list
    x_axis = range(1, len(new_list)+1)
    reverse_xaxis = range(1, len(new_list)+1)
    roc_auc_scores = []
    pr_auc_scores = []
    for i in range(len(new_list)):

        #load classification args 
        classification_model = torch.load(args.classification_model)
        class_args = classification_model['args']
        
        if args.model_type == "resnet":
            #ECG Dataset
            net = resnet34(
                num_classes = class_args.num_classes,
                vqvae_model = class_args.vqvae_model,
                positions = new_list[i],
                mask = class_args.mask,
                auc_classification = args.auc_classification,
                auc_classification_eval = args.auc_classification_eval,
                auc_classification_eval2 = args.auc_classification_eval2,
                model_type = class_args.model_type,
                len_position = args.len_position
            ).to(device)

            #find masking regions 
            masked_regions = net.mask_region(ds.data[:64,:].to(device))

            net = resnet34(
                num_classes = class_args.num_classes,
                vqvae_model = class_args.vqvae_model,
                positions = new_list[i],
                mask = masked_regions,
                auc_classification = args.auc_classification,
                auc_classification_eval = args.auc_classification_eval,
                auc_classification_eval2 = args.auc_classification_eval2,
                model_type = class_args.model_type,
                len_position = args.len_position
            ).to(device)

        elif args.model_type == "raw":
            net = resnet34_raw(
                num_classes = class_args.num_classes,
                vqvae_model = class_args.vqvae_model,
                positions = new_list[i],
                mask = 0,
                auc_classification = args.auc_classification,
                auc_classification_eval = args.auc_classification_eval,
                auc_classification_eval2 = args.auc_classification_eval2,
                model_type = class_args.model_type,
                len_position = args.len_position
            ).to(device)

            for param in net.parameters():
                param.requires_grad = False

        else: 
            #ECG Dataset
            net = VQ_Classifier(
                num_classes = class_args.num_classes,
                vqvae_model = class_args.vqvae_model,
                positions = new_list[i],
                mask = class_args.mask,
                auc_classification = args.auc_classification,
                auc_classification_eval = args.auc_classification_eval,
                auc_classification_eval2 = args.auc_classification_eval2,
                model_type = class_args.model_type,
                len_position = args.len_position
            ).to(device)

            #find masking regions 
            masked_regions = net.mask_region(ds.data[:64,:].to(device))

            net = VQ_Classifier(
                num_classes = class_args.num_classes,
                vqvae_model = class_args.vqvae_model,
                positions = new_list[i],
                mask = masked_regions,
                auc_classification = args.auc_classification,
                auc_classification_eval = args.auc_classification_eval,
                auc_classification_eval2 = args.auc_classification_eval2,
                model_type = class_args.model_type,
                len_position = args.len_position
            ).to(device)        

        #load classification model
        net.load_state_dict(classification_model['model_state_dict'])

        for param in net.parameters():
            param.requires_grad = False        

        net.eval()   


        with torch.no_grad():
            for _, (data, labels) in enumerate(test_loader):
                data = data.unsqueeze(1).float()
                labels = labels.type(torch.LongTensor)
                data, labels = data.to(device), labels.to(device)
                output, codebook_tokens, recon, input= net(data)
                
                score = metrics.roc_auc_score(labels.cpu(), output.cpu())
                score2 = metrics.average_precision_score(labels.cpu(), output.cpu())
                print("AUROC: ", score)
                # print("AUPRC: ", score2)
                roc_auc_scores.append(score)
                pr_auc_scores.append(score2)

    area = np.trapz(roc_auc_scores, list(x_axis)) / 11
    print(f"{method} Area under the curve:", area)                
    plt.plot(list(x_axis), roc_auc_scores, label=method)
    if args.auc_classification_eval:
        plt.xticks(list(x_axis))
    elif args.auc_classification_eval2:
            plt.xticks(list(reverse_xaxis), list(x_axis)[::-1])
    plt.title(f"{args.dataset} dataset {args.model_type} model")
    plt.ylabel('AUROC')
    plt.xlabel('Number of Features')
    plt.legend()
    os.makedirs("./auroc", exist_ok=True)
    os.makedirs(f"./auroc/{args.dataset}", exist_ok=True)
plt.savefig(f"./auroc/{args.dataset}/{args.dataset}_{args.model_type}_{args.model_type}quantizer.png")


            
      




