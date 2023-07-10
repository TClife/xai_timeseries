import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
torch.set_num_threads(32) 
torch.manual_seed(911)

def load_data(data_name, task):
    print(f"task is {task}")
    print(f"dataset is {data_name}")
    if task == 'vqvae':
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
            n_set = torch.tensor(torch.load('./data/mit_bih_dataset/n_data.pt'))
            s_set = torch.tensor(torch.load('./data/mit_bih_dataset/s_data.pt'))
            v_set = torch.tensor(torch.load('./data/mit_bih_dataset/v_data.pt'))
            f_set = torch.tensor(torch.load('./data/mit_bih_dataset/f_data.pt'))
            q_set = torch.tensor(torch.load('./data/mit_bih_dataset/q_data.pt'))
            
            #labels
            n_label = torch.tensor(torch.load('./data/mit_bih_dataset/n_labels.pt'))
            s_label = torch.tensor(torch.load('./data/mit_bih_dataset/s_labels.pt'))
            v_label = torch.tensor(torch.load('./data/mit_bih_dataset/v_labels.pt'))
            f_label = torch.tensor(torch.load('./data/mit_bih_dataset/f_labels.pt'))
            q_label = torch.tensor(torch.load('./data/mit_bih_dataset/q_labels.pt'))

            data = torch.cat((n_set,s_set,v_set,f_set,q_set), dim=0)
            labels = torch.cat((n_label,s_label,v_label,f_label,q_label), dim=0)
            
            #Zero Padding  
            padding=(0,16)
            data = F.pad(data,padding,"constant",0)
            
        else:
            print("Wrong data name")
            
    elif task == 'classification':
        if data_name=='ptb':
            class0_pth = './data/ptbdb_normal.csv'
            class1_pth = './data/ptbdb_abnormal.csv'
            class0_data = torch.tensor(pd.read_csv(class0_pth,skiprows=0).values)[:400]
            class1_data = torch.tensor(pd.read_csv(class1_pth,skiprows=0).values)[:400]
            class0_label = torch.zeros(len(class0_data))
            class1_label = torch.ones(len(class1_data))

            data = torch.cat((class0_data,class1_data), dim=0)
            labels = torch.cat((class0_label, class1_label), dim=0)
            
            #Zero Padding        
            padding = (0, 4)
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
        
        elif data_name=='peak':
            class0_pth = './data/class0.csv'
            class1_pth = './data/class1.csv'
            class0_data = torch.tensor(pd.read_csv(class0_pth,skiprows=0).values)[:400]
            class1_data = torch.tensor(pd.read_csv(class1_pth,skiprows=0).values)[:400]
            class0_label = torch.zeros(len(class0_data))
            class1_label = torch.ones(len(class1_data))
            
            data = torch.cat((class0_data,class1_data), dim=0)[:,:192]
            labels = torch.cat((class0_label, class1_label), dim=0)
            
            #Zero Padding already done (make sure to erase zero-padding region during classification, xai)
            
        elif data_name=='mitbih':
            #data
            n_set = torch.tensor(torch.load('./data/mit_bih_dataset/n_data.pt'))
            s_set = torch.tensor(torch.load('./data/mit_bih_dataset/s_data.pt'))
            v_set = torch.tensor(torch.load('./data/mit_bih_dataset/v_data.pt'))
            f_set = torch.tensor(torch.load('./data/mit_bih_dataset/f_data.pt'))
            q_set = torch.tensor(torch.load('./data/mit_bih_dataset/q_data.pt'))
            
            #labels
            n_label = torch.tensor(torch.load('./data/mit_bih_dataset/n_labels.pt'))
            s_label = torch.tensor(torch.load('./data/mit_bih_dataset/s_labels.pt'))
            v_label = torch.tensor(torch.load('./data/mit_bih_dataset/v_labels.pt'))-2
            f_label = torch.tensor(torch.load('./data/mit_bih_dataset/f_labels.pt'))-2
            q_label = torch.tensor(torch.load('./data/mit_bih_dataset/q_labels.pt'))-3

            data = torch.cat((v_set,f_set), dim=0)
            labels = torch.cat((v_label,f_label), dim=0)

            #Binary classification 
        labels = F.one_hot(labels.long())
            
    elif task == "xai":
        if data_name=='ptb':
            class0_pth = './data/ptbdb_normal.csv'
            class1_pth = './data/ptbdb_abnormal.csv'
            class0_data = torch.tensor(pd.read_csv(class0_pth,skiprows=0).values)[:400]
            class1_data = torch.tensor(pd.read_csv(class1_pth,skiprows=0).values)[:400]
            class0_label = torch.zeros(len(class0_data))
            class1_label = torch.ones(len(class1_data))

            data = torch.cat((class0_data,class1_data), dim=0)
            labels = torch.cat((class0_label, class1_label), dim=0)
            
            #Zero Padding        
            padding = (0, 4)
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
        
        elif data_name=='peak':
            class0_pth = './data/class0.csv'
            class1_pth = './data/class1.csv'
            class0_data = torch.tensor(pd.read_csv(class0_pth,skiprows=0).values)[:400]
            class1_data = torch.tensor(pd.read_csv(class1_pth,skiprows=0).values)[:400]
            class0_label = torch.zeros(len(class0_data))
            class1_label = torch.ones(len(class1_data))
            
            data = torch.cat((class0_data,class1_data), dim=0)[:,:192]
            labels = torch.cat((class0_label, class1_label), dim=0)
            
            #Zero Padding already done (make sure to erase zero-padding region during classification, xai)
            
        elif data_name=='mitbih':
            #data
            mit_path = "/home/hschung/xai/xai_timeseries/data/mit_bih_dataset/"
            n_set = torch.tensor(torch.load(mit_path + 'n_data.pt'))
            s_set = torch.tensor(torch.load(mit_path + 's_data.pt'))
            v_set = torch.tensor(torch.load(mit_path + 'v_data.pt'))[:400]
            f_set = torch.tensor(torch.load(mit_path + 'f_data.pt'))[:400]
            q_set = torch.tensor(torch.load(mit_path + 'q_data.pt'))
            
            #labels
            n_label = torch.tensor(torch.load(mit_path + 'n_labels.pt'))
            s_label = torch.tensor(torch.load(mit_path + 's_labels.pt'))
            v_label = torch.tensor(torch.load(mit_path + 'v_labels.pt'))[:400]-2
            f_label = torch.tensor(torch.load(mit_path + 'f_labels.pt'))[:400]-2
            q_label = torch.tensor(torch.load(mit_path + 'q_labels.pt'))

            data = torch.cat((v_set,f_set), dim=0)
            labels = torch.cat((v_label,f_label), dim=0)
            
        else:
            print("Wrong data name")    

        #Binary classification 
        labels = F.one_hot(labels.long())
    
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

    #plot dataset    
    # if args.plot_dataset:
    #     label0 = []
    #     label1 = []
    #     label2 = []
    #     label3 = []

    #     for idx in range(0, len(data)):
    #         if labels[idx] == 0:
    #             label0.append(data[idx, :])
    #             # plt.plot(data[idx, :], 'r')
    #         elif labels[idx] == 1:
    #             label1.append(data[idx, :])
    #             # plt.plot(data[idx, :], 'orange')
    #         elif labels[idx] == 2:
    #             label2.append(data[idx, :])
    #             # plt.plot(data[idx, :], 'yellow')
    #         elif labels[idx] == 3:
    #             label3.append(data[idx, :])
    #             # plt.plot(data[idx, :], 'g')
        
    #     plot0 = torch.mean(torch.stack(label0),0)
    #     plot1 = torch.mean(torch.stack(label1),0)
    #     plot2 = torch.mean(torch.stack(label2),0)
    #     plot3 = torch.mean(torch.stack(label3),0)

    #     plt.plot(plot0, 'r', label='Class 0')
    #     plt.plot(plot1, 'orange', label='Class 1')
    #     plt.plot(plot2, 'yellow', label='Class 2')
    #     plt.plot(plot3, 'g', label='Class 3')
    #     plt.ylim(-5,5)
    #     plt.xticks(np.arange(0, 1633, 200))
    #     plt.title('ECG Torso Dataset') 
    #     plt.legend()
    #     plt.savefig("ECGtorso_dataset.png")