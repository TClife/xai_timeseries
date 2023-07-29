import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader, random_split
from residual_vqvae import Residual_VQVAE

import numpy as np
import argparse
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt 
import os
import wandb
from data import load_data
torch.set_num_threads(32) 
torch.manual_seed(911)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.init(project="residual_VQVAE", reinit=True)

#Trainer 
class VQTrainer():
    def __init__(self, args):
        self.args = args
         #directory 
        self.savedir = self.args.savedir
        self.dataset = self.args.dataset
        self.task = self.args.task
        directory = os.path.join(self.savedir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        ds = load_data(self.dataset, self.task)


        # Split the dataset into training, validation, and test sets
        train_size = int(0.8 * len(ds))
        val_size = int(0.1 * len(ds))
        test_size = len(ds) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True)

    def train(self, vae):
        self.model = vae.to(device)
        optimizer = optim.Adam(self.model.parameters(), self.args.lr, amsgrad=False)
        # scheduler = MultiStepLR(optimizer, milestones=[200], gamma=0.2)
        train_res_recon_error = []
        val_res_recon_error = []

        best_val_mse = 1000
        best_model = None

        #directory 
        self.savedir = os.path.join(*[self.args.savedir, args.dataset, str(args.num_quantizers)])
        wandb.run.name = self.args.dataset + f"num_quantizers{args.num_quantizers}"
        directory = os.path.join(self.savedir)
        if not os.path.exists(directory):
            os.makedirs(directory)
        savedict = {}
        for epoch in range(self.args.n_epochs):
            self.model.train()
            train_epoch_mse= 0.0

            for i,(data,_) in enumerate(self.train_loader):
                data = data.to(device).unsqueeze(1)
                data = data.float()

                optimizer.zero_grad()

                data_recon, recon_error, commit_loss, indices, _= self.model(data)

                loss = recon_error
                #backprop
                loss.backward()
                optimizer.step()

                train_epoch_mse += recon_error.item()

            train_res_recon_error.append(train_epoch_mse/len(self.train_loader))
            wandb.log({"Training recon loss": train_epoch_mse/len(self.train_loader)})

            # scheduler.step()
            with torch.no_grad():
                self.model.eval()
                val_epoch_mse= 0.0

                for j,(data,_) in enumerate(self.val_loader):
                    data = data.to(device).unsqueeze(1)
                    data = data.float()
                    data_recon, recon_error, commit_loss, indices, _ = self.model(data)
                    loss = recon_error

                    val_epoch_mse += recon_error.item()

                    if j==0:
                        plt.plot(data[j, 0, :].cpu().numpy(), label='original')
                        plt.plot(data_recon[j, 0, :].cpu().numpy(), label='recon')
                        plt.legend()
                        plt.savefig(f"{self.savedir}/recon.png")
                        plt.clf()

                val_res_recon_error.append(val_epoch_mse/len(self.val_loader))
                wandb.log({"Validation recon loss": val_epoch_mse/len(self.val_loader)})
                

            print(f'epoch[{epoch}]', f'train_loss: {train_res_recon_error[-1]:.6f}',
                  f'\tval_loss: {val_res_recon_error[-1]:.6f}')

            if best_val_mse > val_res_recon_error[-1]:
                best_val_mse = val_res_recon_error[-1]
                best_model = copy.deepcopy(self.model)

                savedict = {
                'args': self.args,
                'model_state_dict': best_model.state_dict(),
                'bpe_vocab': None
                    }
                
            if epoch % 10 == 0 and epoch != 0:
                torch.save(savedict, f'{self.savedir}/model_{epoch}.pt')                

        best_iter = np.argmin(val_res_recon_error)
        print('try to save in testdir')
        print(best_iter, f'save ... {self.savedir}/')
        print(f'train: {train_res_recon_error[best_iter]:.4f}\t')
        print(f'val: {val_res_recon_error[best_iter]:.4f}\t')
        print("save done!")
    
    def test(self, vae):
        #load model
        test_res_recon_error = []
        self.savedir = self.args.savedir
        self.model = vae
        a = torch.load(args.load_model)

        self.model.load_state_dict(a['model_state_dict'])

        self.model.eval()
        test_epoch_mse= 0.0

        for j,(data,label) in enumerate(self.test_loader):
            data = data.to(device).unsqueeze(1)    
            data = data.float()
            data_recon, recon_error, commit_loss, indices, _ = self.model(data)
            loss = recon_error

            #If we have indices, we can decode (reconstruct data) with this
            if args.decode: 
                data_recon2 = self.model.indices_to_recon(indices)
                    
            test_epoch_mse += recon_error.item()

            if j==0:
                plt.plot(data[j, 0, :].cpu().detach().numpy(), label='original')
                plt.plot(data_recon[j, 0, :].cpu().detach().numpy(), label='recon')
                plt.legend()
                plt.savefig(f"{self.savedir}/test_recon.png")
                plt.clf()
        
        test_res_recon_error.append(test_epoch_mse/len(self.test_loader))
        print(f'\ttest_loss: {test_res_recon_error[-1]:.10f}')
        
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    #Model Configuration

    parser.add_argument('--dataset', type=str, default='ptb', help="Dataset to train on")

    parser.add_argument('--ecg_size', type=int, default=208, help="Number of timesteps of ECG")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of convolutional layers")
    parser.add_argument('--num_tokens', type=int, default=128, help="Number of tokens in VQVAE")
    parser.add_argument('--codebook_dim', type=int, default=64, help="Dimension of codebook")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Dimension of hidden layer")
    parser.add_argument('--num_resnet_blocks', type=int, default=0, help="Number of resnet blocks")
    parser.add_argument('--temperature', type=float, default=0.9, help="Temperature for gumbel softmax")
    parser.add_argument('--straight_through', type=bool, default=False, help="Straight through estimator for gumbel softmax")
    parser.add_argument('--task', type=str, default='vqvae', help="Task being done")

    parser.add_argument('--savedir', type=str, default="./saved_models")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=301)
    parser.add_argument('--mode', type=str, default='train', choices=['test', 'train'])
    parser.add_argument('--num_quantizers', type=int, default=8)
    

    #test
    parser.add_argument('--load_model', type=str, default="/home/smjo/xai_timeseries/saved_models/toydata3/8/model_240.pt", help="Trained VQ-VAE Path")

    parser.add_argument('--decode', type=bool, default=False, help="Decode from latent space")
    
    args = parser.parse_args()

    vqtrain = VQTrainer(args)

    #load vqvae model 
    vae = Residual_VQVAE(
    image_size = args.ecg_size,
    num_layers = args.num_layers,                 # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
    num_tokens = args.num_tokens,                 # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
    codebook_dim = args.codebook_dim,             # codebook dimension
    hidden_dim = args.hidden_dim,                 # hidden dimension
    num_resnet_blocks = args.num_resnet_blocks,   # number of resnet blocks
    temperature = args.temperature,               # gumbel softmax temperature, the lower this is, the harder the discretization
    straight_through = args.straight_through,      # straight-through for gumbel softmax. unclear if it is better one way or the other
    num_quantizers = args.num_quantizers
    ).to(device)

    if args.mode == "train":
        wandb.login()
        wandb.init()
        vqtrain.train(vae)
    
    if args.mode =="test":
        vqtrain.test(vae)



    

