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
torch.set_num_threads(32)
torch.manual_seed(911) 
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"]="3"
#Trainer
class ClassifierTrainer():
    def __init__(self, args):
        self.args = args
        self.savedir = os.path.join(*[self.args.savedir, self.args.dataset, str(self.args.num_quantizers)])
        self.dataset = self.args.dataset
        self.task = self.args.task
        directory = self.savedir
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        #ds = load_data(self.dataset, self.task)
        ds = load_data(self.dataset, self.task)
        
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
                loss = criterion(output, labels) 
                
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
            correct = 0
            total = 0
            for _, (data, labels) in enumerate(self.val_loader):
                data = data.unsqueeze(1).float()
                labels = labels.float()
                data, labels = data.to(device), labels.to(device)
                output, net_endoces, recon,_= self.model(data)
                _,predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
                loss = criterion(output, labels) 
            accuracy = correct / total

            validation_loss += loss.item() * data.size(0)
            validation_loss = validation_loss / len(self.val_loader.sampler)
            total_val_loss.append(validation_loss)
            wandb.log({"Validation loss": validation_loss, "Accuracy": accuracy})
            if epoch % 10 == 0:
                print(epoch, accuracy)
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
                labels = torch.argmax(labels, dim=1)
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels.cpu(), output[:,1].cpu())
                score = sklearn.metrics.roc_auc_score(labels.cpu(), output[:,1].cpu())
                total_score.append(score)
                print(score)
            
                total_fpr.append(fpr)
                total_tpr.append(tpr)
                total_thresholds.append(thresholds)
                total_input.append(input)
                total_output.append(output)
                total_labels.append(labels)
                total_data.append(data)
                total_recon.append(recon)
                total_tokens.append(codebook_tokens)

                skplt.metrics.plot_roc_curve(labels.cpu(), output.cpu())
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.savefig('auroc.png')
                plt.clf()
                
                skplt.metrics.plot_precision_recall_curve(labels.cpu(), output.cpu())
                plt.savefig('precision_recall.png')
                plt.clf()
            
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
    parser.add_argument('--classification_model', type=str, default="/home/hschung/xai/xai_timeseries/classification_models/flat_conv_transf_nonoverlap_128_8/model_290.pt")
    parser.add_argument('--vqvae_model', default = "/home/hschung/xai/xai_timeseries/vqvae_models/flat_vqvae_nonoverlap_16_8/model_300.pt")  
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--n_emb', type=int, default=64)
    parser.add_argument('--mode', type=str, default='train', choices=['test', 'train'])
    parser.add_argument('--task', type=str, default='classification', help="Task being done")
    parser.add_argument('--dataset', type=str, default='flat', help="Dataset to train on")
    parser.add_argument('--auc_classification', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default="cnn_transformer")
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--positions', type=int, default=0)
    parser.add_argument('--mask', type=int, default=0)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--num_quantizers', type=int, default=[1,2,4,8])

    #directories 
    parser.add_argument('--savedir', type=str, default="/home/smjo/xai_timeseries/vqvae/saved_models/classification/")


    args = parser.parse_args()
    
    classifier_train = ClassifierTrainer(args)

    net = VQ_Classifier(
        num_classes = args.num_classes,
        vqvae_model = args.vqvae_model,
        positions = args.positions,
        mask = args.mask,
        auc_classification = args.auc_classification,
        model_type = args.model_type
    ).to(device)

    if args.mode == "train":
        wandb.login()
        wandb.init()
        classifier_train.train(net)
    
    if args.mode =="test":
        classifier_train.test(net)

        

