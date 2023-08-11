import torch.nn as nn 
import torch 
from torch import tensor, nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from einops import rearrange, repeat, pack, unpack
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset, random_split
import torch.optim as optim
import math 
from residual_vqvae import Residual_VQVAE
import numpy as np 
import os 
from tqdm import tqdm
import sklearn.metrics
import copy
import argparse
from data import load_data
#from x_transformers import TransformerWrapper, Encoder
import itertools
import wandb
torch.set_num_threads(32)
torch.manual_seed(112)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.init(project="resnet for raw time domain", reinit=True)


#Resnet
class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
    
    
class ResNet1d(nn.Module):
    def __init__(self,block ,layers, args):
        super(ResNet1d, self).__init__()
   
        self.args = args
        self.inplanes = self.args.inplanes
        self.conv1 = nn.Conv1d(in_channels=1, out_channels = self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(self.args.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock1d, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, 512, layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        
        self.fc1 = nn.Linear(512 * block.expansion * 2, 1)
        self.fc2 = nn.Linear(512 * block.expansion * 2, 2)
            
        self.dropout = nn.Dropout(0.2)
        
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    
    def forward(self, img):
        x = img.float()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)
        
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        
        x1 = torch.sigmoid(x1)

        return x1,x2, 0,0, img
    
    
def resnet34(*args):
    model = ResNet1d(BasicBlock1d, [3,4,6,3], *args)
    return model


class ClassifierTrainer():
    def __init__(self, args):
        self.args = args
        self.savedir = os.path.join(*[self.args.savedir, self.args.dataset, str(self.args.domain)])
        self.dataset = self.args.dataset
        self.task = self.args.task
        directory = self.savedir
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        #ds = load_data(self.dataset, self.task)
        if self.dataset=='toydata2':
            train_ds = load_data(self.dataset, self.task,set='train', domain=self.args.domain)
            test_ds = load_data(self.dataset, self.task,set='test', domain=self.args.domain)
            val_ds = load_data(self.dataset, self.task,set='val', domain=self.args.domain)
            self.train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
            self.val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
            self.test_loader = DataLoader(test_ds, batch_size = args.batch_size, shuffle=False, pin_memory=True)
        # Split the dataset into training, validation, and test sets
        
        else:
            ds = load_data(self.dataset, self.task,set=None, domain=self.args.domain)
            train_size = int(0.8 * len(ds))
            val_size = int(0.1 * len(ds))
            test_size = len(ds) - train_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])
            self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
            self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
            self.test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, pin_memory=True)
        
    def train(self, net):
        self.model = net.to(device)
        optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        
        criterion = torch.nn.BCELoss()
        
        best_val_loss = 1000
        best_model = None
        total_val_loss = []

        #Traning the Network  
        for epoch in range(self.args.n_epochs):
            self.model.train()
            
            training_loss = 0
            output_list, labels_list = [], []
            
            for _, (data, labels) in enumerate(self.train_loader):
                data = data.unsqueeze(1).float()
                labels = labels.float()
                data, labels = data.to(device), labels.to(device)
                output, prob,_,_, img= self.model(data)
            
                loss = criterion(output.squeeze(1), labels) 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss += loss.item() * data.size(0)
                output_list.append(output.data.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
                
            training_loss = training_loss / len(self.train_loader.sampler)
            wandb.log({"Training loss": training_loss})
            
            self.model.eval()
            validation_loss = 0
            for _, (data, labels) in enumerate(self.val_loader):
                data = data.unsqueeze(1).float()
                labels = labels.float()
                data, labels = data.to(device), labels.to(device)
                output,prob, _,_, img= self.model(data) 
                loss = criterion(output.squeeze(1), labels) 
                
            score = sklearn.metrics.roc_auc_score(labels.cpu().detach().numpy(), output.cpu().detach().numpy())
            validation_loss += loss.item() * data.size(0)
            validation_loss = validation_loss / len(self.val_loader.sampler)
            total_val_loss.append(validation_loss)
            wandb.log({"Validation loss": validation_loss, "Score": score})
            if epoch % 10 == 0:
                print(epoch, score)
            if best_val_loss > total_val_loss[-1]:
                best_val_loss = total_val_loss[-1]
                best_model = copy.deepcopy(self.model)
                
                savedict = {
                    'args': self.args,
                    'model_state_dict': best_model.state_dict(),
                        }
                
            if epoch % 10 == 0:
                savepath  = os.path.join(self.savedir,f"{self.args.model_type}.pt")
                print(savepath)
                torch.save(savedict, savepath) 
        
    def test(self, net):
        criterion = torch.nn.BCELoss()
        self.model = net.to(device)
        a = torch.load(self.args.classification_model)
        self.model.load_state_dict(a['model_state_dict'])
        
        for param in self.model.parameters():
            param.requires_grad = False        
        
        self.model.eval() 
        total_score = [] 

        with torch.no_grad():    
            for _, (data, labels) in enumerate(self.test_loader):
                data = data.unsqueeze(1).float()
                labels = labels.type(torch.LongTensor)
                data, labels = data.to(device), labels.to(device)
                output, prob,_,_, img=self.model(data)
                #calculate auroc curve
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels.cpu(), output.cpu())
                score = sklearn.metrics.roc_auc_score(labels.cpu(), output.cpu())
                total_score.append(score)

                
            print("final test score: ", np.mean(total_score))
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    
    
#block, layers, model_type, num_classes=1, auc_classification=False, auc_classification_eval=False, auc_classification_eval2=False, input_channels=64, inplanes=64    
    parser.add_argument('--classification_model', type=str, default="/home/smjo/xai_timeseries/classification_models/toydata2/time/resnet.pt")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--n_emb', type=int, default=64)
    parser.add_argument('--mode', type=str, default='train', choices=['test', 'train'])
    parser.add_argument('--task', type=str, default='classification', choices=['xai', 'classification'])
    parser.add_argument('--dataset', type=str, default='toydata', help="Dataset to train on")
    parser.add_argument('--model_type', type=str, default="resnet")
    parser.add_argument('--domain', type=str, default="time")
    parser.add_argument('--savedir', type=str, default="./classification_models")
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--auc_classification',type=str, default=False)
    parser.add_argument('--auc_classification_eval',type=str, default=False)
    parser.add_argument('--auc_classification_eval2',type=str, default=False)
    parser.add_argument('--input_channels', type=int, default=64)
    parser.add_argument('--inplanes', type=int, default=64)
    args = parser.parse_args()
    
    
    classifier_train = ClassifierTrainer(args)
    net = resnet34(args).to(device)
    
    if args.mode=='train':
        classifier_train.train(net)
    else:
        classifier_train.test(net)
    
