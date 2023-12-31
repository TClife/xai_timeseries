import torch 
import argparse
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim 
from models import VQ_Classifier
import copy
import numpy as np 
import wandb 
import matplotlib.pyplot as plt
import os 
from data import load_data
import sklearn 
import scikitplot as skplt
from dataload import makedata
from models import resnet18, resnet34, resnet34_raw
torch.set_num_threads(32)
torch.manual_seed(911) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.init(project="residual_VQVAE classificaation", reinit=True)
#Trainer
class ClassifierTrainer():
    def __init__(self, args):
        self.args = args
        self.savedir = os.path.join(*[self.args.savedir, self.args.dataset, str(self.args.num_quantizers), str(self.args.domain)])
        self.dataset = self.args.dataset
        self.task = self.args.task
        directory = self.savedir
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        #ds = load_data(self.dataset, self.task)
        ds = load_data(self.dataset, self.task, domain=self.args.domain)
        
        # Split the dataset into training, validation, and test sets
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
                output, indices, recon, quantized= self.model(data) 
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
                output, net_endoces, recon,_= self.model(data)
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
        
        total_input = []
        total_output = []
        total_labels = []
        total_data = []
        total_recon = []   
        total_tokens = []   
        total_score = [] 
        
        total_fpr = []
        total_tpr = []
        total_thresholds = [] 
        
        with torch.no_grad():    
            for _, (data, labels) in enumerate(self.test_loader):
                data = data.unsqueeze(1).float()
                labels = labels.type(torch.LongTensor)
                data, labels = data.to(device), labels.to(device)
                output, codebook_tokens, recon, input= self.model(data)
                #calculate auroc curve
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels.cpu(), output.cpu())
                score = sklearn.metrics.roc_auc_score(labels.cpu(), output.cpu())
                total_score.append(score)
                print(output, labels)
            
                total_fpr.append(fpr)
                total_tpr.append(tpr)
                total_thresholds.append(thresholds)
                total_input.append(input)
                total_output.append(output)
                total_labels.append(labels)
                total_data.append(data)
                total_recon.append(recon)
                total_tokens.append(codebook_tokens)
                '''
                skplt.metrics.plot_roc_curve(labels.cpu(), output.cpu())
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.savefig('auroc.png')
                plt.clf()
                
                skplt.metrics.plot_precision_recall_curve(labels.cpu(), output.cpu())
                plt.savefig('precision_recall.png')
                plt.clf()
            '''
            print("final test score: ", np.mean(total_score))
            total_input = torch.vstack(total_input)
            total_output = torch.vstack(total_output)
            total_labels = torch.concat(total_labels)
            total_data = torch.vstack(total_data)
            total_recon = torch.vstack(total_recon)
            total_tokens = torch.vstack(total_tokens)          
                
            # accuracy = correct / total
            # print(accuracy)
            
        misc = {
            "codebook_tokens": total_tokens, 
            "input": total_input,
            "output": total_output,
            "labels": total_labels,
            "data": total_data,
            "recon": total_recon
        }
        
        torch.save(misc, f"./{args.dataset}_dataset_IG.pt")
        
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    #Model Configuration

    parser.add_argument('--classification_model', type=str, default="/home/smjo/xai_timeseries/classification_models/toydata3/8/resnet.pt")
    parser.add_argument('--vqvae_model', default = "/home/smjo/xai_timeseries/saved_models/toydata3/8/model_280.pt")  

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=3000)
    parser.add_argument('--n_emb', type=int, default=64)
    parser.add_argument('--mode', type=str, default='train', choices=['test', 'train'])
    parser.add_argument('--task', type=str, default='classification', help="Task being done")
    parser.add_argument('--dataset', type=str, default='toy_dataset', help="Dataset to train on")
    parser.add_argument('--auc_classification', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default="resnet")
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--positions', type=int, default=0)
    parser.add_argument('--mask', type=int, default=0)
    parser.add_argument('--len_position', type=int, default=12)
    parser.add_argument('--num_quantizers', type=int, default=8)
    parser.add_argument('--domain', type=str, default='time', help='time or frequency')

    #directories 
    parser.add_argument('--savedir', type=str, default="./classification_models")


    args = parser.parse_args()
    
    classifier_train = ClassifierTrainer(args)
    if args.domain == 'time':
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
        
        else:
            net = VQ_Classifier(
                num_classes = args.num_classes,
                vqvae_model = args.vqvae_model,
                positions = args.positions,
                mask = args.mask,
                auc_classification = args.auc_classification,
                model_type = args.model_type,
                len_position = args.len_position
            ).to(device)
            
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

    else:
        if args.model_type == 'resnet':
            net = resnet34(
                
            )
            
    if args.mode=='train':
        classifier_train.train(net)
    else:
        classifier_train.test(net)
    






    

