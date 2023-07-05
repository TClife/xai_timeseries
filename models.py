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
import torch.optim as optim
import math 
from residual_vqvae import Residual_VQVAE
import numpy as np 
from tqdm import tqdm

torch.set_num_threads(32)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VQ_Classifier(nn.Module):
    def __init__(self, *, num_classes, vqvae_model, positions, mask, auc_classification, model_type):
        super().__init__()
        self.model_type = model_type
        self.positions = positions
        self.mask = mask
        self.auc_classification = auc_classification
        #load vqvae model 
        vae_path = vqvae_model
        load_dict = torch.load(vae_path)["model_state_dict"]
        self.args= torch.load(vae_path)["args"]

        #vqvae model
        self.vae = Residual_VQVAE(image_size=self.args.ecg_size-16, num_layers=self.args.num_layers, num_tokens=self.args.num_tokens, 
                                  codebook_dim=self.args.codebook_dim, hidden_dim=self.args.hidden_dim, num_resnet_blocks=self.args.num_resnet_blocks, 
                                  temperature=self.args.temperature, straight_through=self.args.straight_through, num_quantizers=self.args.num_quantizers).to(device)
        self.vae.load_state_dict(load_dict)

        for param in self.vae.parameters():
            param.requires_grad = False    
        
        self.to_hidden = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )
        self.embedding = nn.Embedding(self.args.num_tokens, self.args.codebook_dim)

        if self.model_type == "cnn":
            self.mlp_head = nn.Sequential(
                nn.Linear(16, num_classes),
                nn.ReLU()
            )
            
        elif self.model_type == "transformer":
            self.encoderlayer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
            self.transformer = nn.TransformerEncoder(self.encoderlayer, num_layers=2)
            
            self.mlp_head = nn.Sequential(
                nn.Linear(16, num_classes),
                nn.ReLU()
            )
            self.mlp_head2 = nn.Sequential(
                nn.Linear(64, num_classes),
                nn.ReLU()
            )      
        elif self.model_type == "cnn_transformer":
            self.cnnencoderlayer = nn.TransformerEncoderLayer(d_model=16, nhead=8)
            self.cnn_transformer = nn.TransformerEncoder(self.cnnencoderlayer, num_layers=1)
            
            self.mlp_head2 = nn.Sequential(
                nn.Linear(64, num_classes),
                nn.ReLU()
            )
            self.mlp_head = nn.Sequential(
                nn.Linear(16, num_classes),
                nn.ReLU()
            )

        self.softmax = nn.Softmax(dim=1)
    
    def ig(self, encoding_indices):
        quantized = self.embedding(encoding_indices.long())
        quantized = quantized.view(quantized.shape[0], -1, quantized.shape[-1])
            

        if self.model_type == "cnn":
            quantized_t = quantized.transpose(2,1)
            x = self.to_hidden(quantized_t)
            x = x.transpose(2,1) 
            x = x.mean(dim = 1)
            x = self.mlp_head(x)
        elif self.model_type =="transformer":
            x = self.transformer(quantized)
            x = x.mean(dim = 1)
            x = self.mlp_head2(x)
        elif self.model_type =="cnn_transformer":
            quantized_t = quantized.transpose(2,1)
            x = self.to_hidden(quantized_t)
            x = x.transpose(2,1)      
            x = self.cnn_transformer(x)
            x = x.mean(dim = 1)
            x = self.mlp_head(x)
    
        
        return x, encoding_indices, quantized

    
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(device)
            X = X.unsqueeze(1)
        X = tensor(X, dtype=torch.float32)
        X = X.cuda()
        ds = TensorDataset(X)
        dl = DataLoader(ds, batch_size=256, shuffle=False, drop_last=False)

        """Evaluate the given data loader on the model and return predictions"""
        result = None
        recon = None
        with torch.no_grad():
            for x in dl:
                original = x[0].squeeze(1).long()
                x = self.embedding(original)
                quantized = x.view(x.shape[0], -1, x.shape[-1])
                
                if self.model_type == "cnn":
                    quantized_t = quantized.transpose(2,1)
                    x = self.to_hidden(quantized_t)
                    x = x.transpose(2,1) 
                    x = x.mean(dim = 1)
                    x = self.mlp_head(x)
                elif self.model_type =="transformer":
                    x = self.transformer(quantized)
                    x = x.mean(dim = 1)
                    x = self.mlp_head2(x)
                elif self.model_type =="cnn_transformer":
                    quantized_t = quantized.transpose(2,1)
                    x = self.to_hidden(quantized_t)
                    x = x.transpose(2,1)      
                    x = self.cnn_transformer(x)
                    x = x.mean(dim = 1)
                    x = self.mlp_head(x)
                
                x = x.cpu().detach().numpy()
                #decode indices
                x_recon = self.vae.indices_to_recon(original)
                x_recon = x_recon.cpu().detach().numpy()        

                recon = x_recon if recon is None else np.concatenate((recon, x_recon), axis=0)
                result = x if result is None else np.concatenate((result, x), axis=0)
                
        return result, recon

    
    def mask_region(self, img):
     
        #Zero Padding        
        padding = (0, 16)
        img = F.pad(img, padding, "constant", 0)
        
        img = img.unsqueeze(1)
         #convert numpy to tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).to(device)
            quantized = self.embedding(img)
            quantized = quantized.transpose(2,1)
            x = self.to_hidden(quantized)
            x = x.transpose(2,1) 
            x = x.mean(dim = 1)
            
            return self.mlp_head(x)

        encoded = self.vae.encoder(img.float()) 

        t = encoded.shape[-1] 

        encoded = rearrange(encoded, 'b c t -> b t c') 
        quantized, encoding_indices, commit_loss = self.vae.vq(encoded)
        masked_regions = []
        
        for i in range(encoding_indices.shape[2]):
            mask_indices,_ = torch.mode(encoding_indices[:,-1,i])
            masked_regions.append(mask_indices)

        return masked_regions
    
    def forward(self, img):
        #convert numpy to tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).to(device)
            quantized = self.embedding(img)
            quantized = quantized.transpose(2,1)
            x = self.to_hidden(quantized)
            x = x.transpose(2,1) 
            x = x.mean(dim = 1)
            
            return self.mlp_head(x)

        encoded = self.vae.encoder(img.float()) 

        t = encoded.shape[-1] 

        encoded = rearrange(encoded, 'b c t -> b t c') 
        quantized, encoding_indices, commit_loss = self.vae.vq(encoded)

        if self.auc_classification:
            new_tensor=[]
            #create a mask tensor
            for idx in tqdm(range(len(img))):
                tmp = torch.empty(encoding_indices[idx].shape, dtype = int)          #12*num_quantizers
                for num in range(encoding_indices[idx].shape[1]):
                    #tmp = torch.full_like(encoding_indices[idx][:,num], self.mask[num], device=device)    #12
                    tmp[:,num] = self.mask[num]
                    for i in range(len(self.positions)):
                        #encoding indices 80*12*2
                        try:
                            tmp[:,self.positions[i]] = encoding_indices[idx,:,self.positions[i]] 
                        except:
                            continue
                new_tensor.append([tmp.cpu().numpy()])
                sleep(0.1)
          
            new_tensor = torch.LongTensor(new_tensor).to(device)
            quantized = self.embedding(new_tensor)
            quantized = quantized.reshape(len(img),-1,64)

        else:
            #embed codebooks
            #print(f"else case:{encoding_indices[0]}")
            quantized = self.embedding(encoding_indices)
            quantized = quantized.view(quantized.shape[0], -1, quantized.shape[-1])
            #print(f"quantized:{quantized.shape}")
            

        if self.model_type == "cnn":
            quantized_t = quantized.transpose(2,1)
            x = self.to_hidden(quantized_t)
            x = x.transpose(2,1) 
            x = x.mean(dim = 1)
            x = self.mlp_head(x)
        elif self.model_type =="transformer":
            x = self.transformer(quantized)
            x = x.mean(dim = 1)
            x = self.mlp_head2(x)
        elif self.model_type =="cnn_transformer":
            quantized_t = quantized.transpose(2,1)
            x = self.to_hidden(quantized_t)
            x = x.transpose(2,1)      
            x = self.cnn_transformer(x)
            x = x.mean(dim = 1)
            x = self.mlp_head(x)
        
        #sigmoid of output
        x = torch.sigmoid(x)
        
        x_recon = self.vae(img)[0]
        
        return x, encoding_indices, x_recon, quantized