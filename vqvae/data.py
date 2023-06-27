import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
torch.set_num_threads(32) 
torch.manual_seed(911)

def load_data(data_name):
    print(f"dataset is {data_name}")
    if data_name=='ptb':
        class0_pth = './data/ptbdb_normal.csv'
        class1_pth = './data/ptbdb_abnormal.csv'
        class0_data = torch.tensor(pd.read_csv(class0_pth,skiprows=0).values)
        class1_data = torch.tensor(pd.read_csv(class1_pth,skiprows=0).values)
        class0_label = torch.zeros(len(class0_data))
        class1_label = torch.ones(len(class1_data))

        data = torch.cat((class0_data,class1_data), dim=0)
        labels = torch.cat((class0_label, class1_label), dim=0)
        
        #Zero Padding        
        padding = (0, 20)
        data = F.pad(data, padding, "constant", 0)
        
    elif data_name=='flat':
        class0_pth = "./data/amplitude_class0.csv"
        class1_pth = "./data/amplitude_class1.csv"
        class0_data = torch.tensor(pd.read_csv(class0_pth,skiprows=0).values)
        class1_data = torch.tensor(pd.read_csv(class1_pth,skiprows=0).values)
        class0_label = torch.zeros(len(class0_data))
        class1_label = torch.ones(len(class1_data))
        
        data = torch.cat((class0_data,class1_data), dim=0)
        labels = torch.cat((class0_label, class1_label), dim=0)
        
        #Zero Padding  
        padding = (0, 16)
        data = F.pad(data, padding, "constant", 0)
    
    elif data_name=='peak':
        class0_pth = './data/class0.csv'
        class1_pth = './data/class1.csv'
        class0_data = torch.tensor(pd.read_csv(class0_pth,skiprows=0).values)
        class1_data = torch.tensor(pd.read_csv(class1_pth,skiprows=0).values)
        class0_label = torch.zeros(len(class0_data))
        class1_label = torch.ones(len(class1_data))
        
        data = torch.cat((class0_data,class1_data), dim=0)
        labels = torch.cat((class0_label, class1_label), dim=0)
        
        #Zero Padding already done (make sure to erase zero-padding region during classification, xai)
        
    elif data_name=='mitbih':
        #data
        n_set = torch.tensor(torch.load('./mit_bih_dataset/n_data.pt'))
        s_set = torch.tensor(torch.load('./mit_bih_dataset/s_data.pt'))
        v_set = torch.tensor(torch.load('./mit_bih_dataset/v_data.pt'))
        f_set = torch.tensor(torch.load('./mit_bih_dataset/f_data.pt'))
        q_set = torch.tensor(torch.load('./mit_bih_dataset/q_data.pt'))
        
        #labels
        n_label = torch.tensor(torch.load('./mit_bih_dataset/n_labels.pt'))
        s_label = torch.tensor(torch.load('./mit_bih_dataset/s_labels.pt'))
        v_label = torch.tensor(torch.load('./mit_bih_dataset/v_labels.pt'))
        f_label = torch.tensor(torch.load('./mit_bih_dataset/f_labels.pt'))
        q_label = torch.tensor(torch.load('./mit_bih_dataset/q_labels.pt'))

        data = torch.cat((n_set,s_set,v_set,f_set,q_set), dim=0)
        labels = torch.cat((n_label,s_label,v_label,f_label,q_label), dim=0)
        
        #Zero Padding  
        padding=(0,16)
        data = F.pad(data,padding,"constant",0)
        
    else:
        print("Wrong data name")

    
    print(f"X shape:{data.shape}, y shape:{labels.shape}")
    
    class ECGDataset(Dataset):
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
            
    ds = ECGDataset(data, labels)
 
    return ds