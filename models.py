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
import os 
from tqdm import tqdm

torch.set_num_threads(32)
torch.manual_seed(112)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, block, layers, vqvae_model, positions, mask, model_type, len_position, num_classes=1, auc_classification=False, auc_classification_eval=False, input_channels=64, inplanes=64):
        super(ResNet1d, self).__init__()

        self.model_type = model_type
        self.positions = positions
        self.len_position = len_position
        self.mask = mask
        self.auc_classification = auc_classification
        self.auc_classification_eval = auc_classification_eval
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

        self.embedding = nn.Embedding(self.args.num_tokens, self.args.codebook_dim)
        self.inplanes = inplanes
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock1d, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, 512, layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
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
        encoded = self.vae.encoder(img.float()) 

        t = encoded.shape[-1] 

        encoded = rearrange(encoded, 'b c t -> b t c') 
        quantized, encoding_indices, commit_loss = self.vae.vq(encoded) 

        if self.auc_classification_eval:
            #create a mask tensor
            tmp = torch.zeros(encoding_indices.shape, dtype=int, device=device)
            for i in range(len(self.mask)):
                tmp[:,:,i] = self.mask[i]
            
            new_tensor = tmp
            
            for i in range(len(self.positions)):
                new_tensor[:,self.positions[i], :] = encoding_indices[:,self.positions[i], :] 
            
            quantized = self.embedding(new_tensor)
            quantized = quantized.reshape(len(img),-1,64)
            
        else:
            #embed codebooks
            #print(f"else case:{encoding_indices[0]}")
            quantized = self.embedding(encoding_indices)
            quantized = quantized.view(quantized.shape[0], -1, quantized.shape[-1])
            #print(f"quantized:{quantized.shape}")
        quantized_t = quantized.transpose(2,1)
        x = quantized_t
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
        x = self.fc(x)
        x = torch.sigmoid(x)

        x_recon = self.vae(img)[0]

        return x, encoding_indices, x_recon, quantized


def resnet18(**kwargs):
    model = ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)
    return model

#Positional Encoding 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class VQ_Classifier(nn.Module):
    def __init__(self, *, num_classes, vqvae_model, positions, mask, auc_classification=False, auc_classification_eval=False, model_type, len_position):
        super().__init__()
        self.model_type = model_type
        self.positions = positions
        self.len_position = len_position
        self.mask = mask
        self.auc_classification = auc_classification
        self.auc_classification_eval = auc_classification_eval
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
            self.pos_encoder = PositionalEncoding(64)
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
    
    def original_decode(self, img, sample_num, labels):
        encoded = self.vae.encoder(img.float()) 
        
        t = encoded.shape[-1]
         
        encoded = rearrange(encoded, 'b c t -> b t c') 
        quantized, encoding_indices, commit_loss = self.vae.vq(encoded)   
        
        encoding_indices = encoding_indices.cpu().detach().numpy()
 
        #write to csv file 
        with open('/home/hschung/xai/xai_timeseries/xai/altered_recon/original_sample{}.csv'.format(sample_num), 'a') as f:
            np.savetxt(f, encoding_indices.reshape(encoding_indices.shape[1], encoding_indices.shape[2]), delimiter=',') 
            np.savetxt(f, labels.cpu().detach().numpy(), delimiter=',')      
             
    
    def masking_position(self, img, selected_position, selected_quantizer, sample_num):
        encoded = self.vae.encoder(img.float()) 
        
        t = encoded.shape[-1]
         
        encoded = rearrange(encoded, 'b c t -> b t c') 
        quantized, encoding_indices, commit_loss = self.vae.vq(encoded)    
        
        masked_tensor = torch.zeros(encoding_indices.shape, dtype=int, device=device)
        for i in range(len(self.mask)):
            masked_tensor[:,:,i] = self.mask[i]  
                  
        #original tensor for selected position
        original_tensor = masked_tensor.clone()
        original_tensor[:,selected_position,:] = encoding_indices[:,selected_position,:]
        
        #original tensor 
        original_recon = self.vae.indices_to_recon(original_tensor)
        plt.plot(img[0][0].cpu().detach().numpy())
        plt.plot(original_recon[0][0].cpu().detach().numpy())
        os.makedirs("/home/hschung/xai/xai_timeseries/altered_recon/original_samples/", exist_ok=True)
        plt.savefig("/home/hschung/xai/xai_timeseries/altered_recon/original_samples/sample{}_position{}.png".format(sample_num, selected_position))
        plt.clf() 
        
        #perturbed only the selected quantizer 
        print("original_token: {}".format(original_tensor[0,selected_position, selected_quantizer]))
        initial_recon = original_recon.clone()
        for i in range(self.args.num_tokens):
            plt.plot(initial_recon[0][0].cpu().detach().numpy())
            original_tensor[:,selected_position, selected_quantizer] = i
            original_recon = self.vae.indices_to_recon(original_tensor)
            plt.plot(original_recon[0][0].cpu().detach().numpy())
            os.makedirs("/home/hschung/xai/xai_timeseries/xai/altered_recon/sample{}_perturbed_position{}_quantizer{}".format(sample_num, selected_position, selected_quantizer), exist_ok=True)
            plt.savefig("/home/hschung/xai/xai_timeseries/xai/altered_recon/sample{}_perturbed_position{}_quantizer{}/code{}.png".format(sample_num, selected_position, selected_quantizer, i))
            plt.clf() 

            #write to csv file 
            with open('/home/hschung/xai/xai_timeseries/xai/altered_recon/altered_recon_sample{}_position{}_quantizer{}.csv'.format(sample_num, selected_position, selected_quantizer), 'a') as f:
                np.savetxt(f, original_recon[0][0].cpu().detach().numpy().reshape(1, 192), delimiter=',')
               
    
    def position_answer(self, img, unmasked_positions, labels, selected_positions):
        encoded = self.vae.encoder(img.float()) 
        
        t = encoded.shape[-1]
         
        encoded = rearrange(encoded, 'b c t -> b t c') 
        quantized, encoding_indices, commit_loss = self.vae.vq(encoded)    
        
        masked_tensor = torch.zeros(encoding_indices.shape, dtype=int, device=device)
        for i in range(len(self.mask)):
            masked_tensor[:,:,i] = self.mask[i]
        
        if not selected_positions:
            
            position_scores = []
            for i in range(self.len_position):
                new_tensor = masked_tensor.clone()
                new_tensor[:,i,:] = encoding_indices[:,i,:]
                result = None
                x = self.embedding(new_tensor)
                quantized = x.view(x.shape[0], -1, x.shape[-1])
                with torch.no_grad():
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
                    
                    x = torch.sigmoid(x)
                    
                    position_scores.append(torch.mean(x[torch.arange(labels.shape[0]), labels]))
            max_position = torch.argmax(torch.tensor(position_scores))

        else:
            
            #fix selected positions 
            new_tensor = masked_tensor.clone()
            for position in selected_positions:
                new_tensor[:,position,:] = encoding_indices[:,position,:]
            
            position_list = torch.arange(self.len_position)
            designated_positions = torch.tensor(selected_positions)
            
            created_mask = torch.isin(position_list, designated_positions)
            created_mask = ~created_mask
            positions_consider = position_list[created_mask]
            
            position_scores = []
            altered_recon = []
            for i in positions_consider:
                altered_tensor = new_tensor.clone()
                altered_tensor[:,i,:] = encoding_indices[:,i,:]
                result = None
                x = self.embedding(altered_tensor)
                quantized = x.view(x.shape[0], -1, x.shape[-1])
                with torch.no_grad():
                    if self.model_type == "cnn":
                        quantized_t = quantized.transpose(2,1)
                        x = self.to_hidden(quantized_t)
                        x = x.transpose(2,1) 
                        x = x.mean(dim = 1)
                        x = self.mlp_head(x)
                    elif self.model_type =="transformer":
                        x = self.transformer(quantized)
                        x = x[:, 0, :]
                        x = self.mlp_head2(x)
                    elif self.model_type =="cnn_transformer":
                        quantized_t = quantized.transpose(2,1)
                        x = self.to_hidden(quantized_t)
                        x = x.transpose(2,1)      
                        x = self.cnn_transformer(x)
                        x = x.mean(dim = 1)
                        x = self.mlp_head(x)
                    
                    x = torch.sigmoid(x)
        
                    position_scores.append(torch.mean(x[torch.arange(labels.shape[0]), labels]))
            max_position = torch.argmax(torch.tensor(position_scores))
            max_position = positions_consider[max_position]
            
        #plot x_recon 
        x_recon = self.vae.indices_to_recon(new_tensor)
        plt.plot(img[10][0].cpu().detach().numpy())
        plt.plot(x_recon[10][0].cpu().detach().numpy())
        plt.savefig("/home/hschung/xai/xai_timeseries/xai/altered_recon/{}_{}.png".format(self.model_type, len(selected_positions)))
        plt.clf()
        
        selected_positions.append(max_position)
        return selected_positions
                    
    
    def forward(self, img):
        encoded = self.vae.encoder(img.float()) 

        t = encoded.shape[-1] 

        encoded = rearrange(encoded, 'b c t -> b t c') 
        quantized, encoding_indices, commit_loss = self.vae.vq(encoded)            
        
        if self.auc_classification_eval:
            #create a mask tensor
            tmp = torch.zeros(encoding_indices.shape, dtype=int, device=device)
            for i in range(len(self.mask)):
                tmp[:,:,i] = self.mask[i]
            
            new_tensor = tmp
            
            for i in range(len(self.positions)):
                new_tensor[:,self.positions[i], :] = encoding_indices[:,self.positions[i], :] 
            
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
            quantized = self.pos_encoder(quantized)
            x = self.transformer(quantized.transpose(0,1))
            x = x[0,:,:]
            x = self.mlp_head2(x)
        elif self.model_type =="cnn_transformer":
            quantized_t = quantized.transpose(2,1)
            x = self.to_hidden(quantized_t)
            x = x.transpose(2,1)      
            x = self.cnn_transformer(x)
            x = x[:, -1, :]
            x = self.mlp_head(x)
        
        #sigmoid of output
        x = torch.sigmoid(x)
        
        x_recon = self.vae(img)[0]
        
        return x, encoding_indices, x_recon, quantized
