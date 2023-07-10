import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from models import VQ_Classifier
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
torch.manual_seed(114)


#Create args parser for labels and batch size 
parser = ArgumentParser() 
parser.add_argument('--labels', type=int, default=0)
parser.add_argument('--num_features', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=80)
parser.add_argument('--vqvae_model', default = "/home/smjo/xai_timeseries/vqvae/saved_models/hard_mitbih/8/model_290.pt")
parser.add_argument('--classification_model', type=str, default="/home/smjo/xai_timeseries/vqvae/saved_models/classification/mitbih/8/cnn.pt")
parser.add_argument('--task', type=str, default='xai', help="Task being done")
parser.add_argument('--dataset', type=str, default="mitbih")
parser.add_argument('--plot_dataset', type=bool, default=True)
parser.add_argument('--model_type', type=str, default="cnn")
parser.add_argument('--auc_classification', type=bool, default=False)
parser.add_argument('--auc_classification_eval', type=bool, default=True)
parser.add_argument('--len_position', type=int, default=12)
parser.add_argument('--method', default= ["LIME", "SHAP", "IG", "Ours"])
parser.add_argument('--position_ranking', default = [[2,3,1,0,4,5,6,7,8,9,11,10], [2,3,1,0,4,5,6,7,8,9,10,11], [3,0,4,1,2,5,8,6,7,9,10,11], [0,2,1,3,4,5,6,7,8,10,11,9]])

args = parser.parse_args() 

#loop over methods and position rankings
for method, position_ranking in zip(args.method, args.position_ranking):

    #get positions to be used as features
    new_list = []

    for i in range(len(position_ranking)):
        new_list.append(position_ranking[:i+1])

    print(new_list)

    #loop over new_list
    x_axis = range(1, len(new_list)+1)
    roc_auc_scores = []
    pr_auc_scores = []
    for i in range(len(new_list)):
        ds = load_data(args.dataset, args.task)

        # Split the dataset into training, validation, and test sets
        train_size = int(0.8 * len(ds))
        val_size = int(0.1 * len(ds))
        test_size = len(ds) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])
        training_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        #load classification args 
        classification_model = torch.load(args.classification_model)
        class_args = classification_model['args']
        #ECG Dataset
        net = VQ_Classifier(
            num_classes = class_args.num_classes,
            vqvae_model = class_args.vqvae_model,
            positions = new_list[i],
            mask = class_args.mask,
            auc_classification = args.auc_classification,
            auc_classification_eval = args.auc_classification_eval,
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
                labels = torch.argmax(labels, dim=1)
                data, labels = data.to(device), labels.to(device)
                output, codebook_tokens, recon, input= net(data)
                
                score = metrics.roc_auc_score(labels.cpu(), output[:,1].cpu())
                score2 = metrics.average_precision_score(labels.cpu(), output[:,1].cpu())
                print(score)
                print(score2)
                roc_auc_scores.append(score)
                pr_auc_scores.append(score2)

    area = np.trapz(roc_auc_scores, list(x_axis)) / 11
    print(f"{method} Area under the curve:", area)                
    plt.plot(list(x_axis), roc_auc_scores, label=method)
    plt.title(f"{args.dataset} dataset {args.model_type} model")
    plt.ylabel('AUROC')
    plt.xlabel('Number of Features')
    plt.legend()
    os.makedirs("./auroc", exist_ok=True)
    os.makedirs(f"./auroc/{args.dataset}", exist_ok=True)
plt.savefig(f"./auroc/{args.dataset}/{args.dataset}_{args.model_type}_{len(masked_regions)}quantizer.png")


            
      




