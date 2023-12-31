import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
torch.set_num_threads(32) 
torch.manual_seed(911)

def load_data(data_name, task,set, domain=None):
    print(f"task is {task}")
    print(f"dataset is {data_name}")
    if task == 'vqvae':
        if data_name=='ptb':
            class0_pth = './data/ptbdb_normal.csv'
            class1_pth = './data/ptbdb_abnormal.csv'
            class0_data = torch.tensor(pd.read_csv(class0_pth,skiprows=0).values)[:, :187]
            class1_data = torch.tensor(pd.read_csv(class1_pth,skiprows=0).values)[:, :187]
            class0_label = torch.zeros(len(class0_data))
            class1_label = torch.ones(len(class1_data))

            data = torch.cat((class0_data,class1_data), dim=0)
            labels = torch.cat((class0_label, class1_label), dim=0)
            
            #Zero Padding        
            padding = (0, 21)
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
        
        elif data_name=='toydata3':
            class0_data = torch.load('/home/smjo/xai_timeseries/data/toydata3_class0.pt')
            class1_data = torch.load('/home/smjo/xai_timeseries/data/toydata3_class1.pt')
            class0_label = torch.zeros(len(class0_data))
            class1_label = torch.ones(len(class1_data))
            
            data = torch.cat((class0_data,class1_data), dim=0)
            labels = torch.cat((class0_label, class1_label), dim=0)
        
        else:
            print("Wrong data name")
            
    elif task == 'classification':
            
            #Zero Padding already done (make sure to erase zero-padding region during classification, xai)  
        if data_name=='toydata':
            class0_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset/time_domain_class0.csv').values)
            class1_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset/time_domain_class1.csv').values)
            data = torch.cat([class0_data,class1_data], dim=0)[:,:192]
            
            class0_label = torch.zeros(len(class0_data))
            class1_label = torch.ones(len(class1_data))
        
            labels = torch.cat((class0_label, class1_label), dim=0)
            
        
        elif data_name=='toydata2':
            if set=='train':
                class0_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset2/train_random_dataset_class1.csv').values)
                class1_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset2/train_random_dataset_class2.csv').values)
                data = torch.cat([class0_data,class1_data], dim=0)[:,:192]
                
                class0_label = torch.zeros(len(class0_data))
                class1_label= torch.ones(len(class1_data))
            
                labels = torch.cat((class0_label, class1_label), dim=0)
            
            elif set=='test':
            
                class0_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset2/test_random_dataset_class1.csv').values)
                class1_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset2/test_random_dataset_class2.csv').values)
                data = torch.cat([class0_data,class1_data], dim=0)[:,:192]
                
                class0_label = torch.zeros(len(class0_data))
                class1_label = torch.ones(len(class1_data))
            
                labels = torch.cat((class0_label, class1_label), dim=0)
            else:
                class0_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset2/val_random_dataset_class1.csv').values)
                class1_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset2/val_random_dataset_class2.csv').values)
                data = torch.cat([class0_data,class1_data], dim=0)[:,:192]
                
                class0_label = torch.zeros(len(class0_data))
                class1_label = torch.ones(len(class1_data))
            
                labels = torch.cat((class0_label, class1_label), dim=0)
            
            
            
            
        elif data_name=='toydata3':
            class0_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset3/toy_dataset3_class0.csv').values)
            class1_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset3/toy_dataset3_class1.csv').values)
            data = torch.cat([class0_data,class1_data], dim=0)[:,:192]
            
            class0_label = torch.zeros(len(class0_data))
            class1_label = torch.ones(len(class1_data))
        
            labels = torch.cat((class0_label, class1_label), dim=0)
            
        elif data_name=='mitbih':
            #data
            mit_path = "/home/hschung/xai/xai_timeseries/data/mit_bih_dataset/"
            n_set = torch.tensor(torch.load(mit_path + 'n_data.pt'))
            s_set = torch.tensor(torch.load(mit_path + 's_data.pt'))
            v_set = torch.tensor(torch.load(mit_path + 'v_data.pt'))[:400]
            f_set = torch.tensor(torch.load(mit_path + 'f_data.pt'))[:400]
            q_set = torch.tensor(torch.load(mit_path + 'q_data.pt'))[:400]
            
            #labels
            n_label = torch.tensor(torch.load(mit_path + 'n_labels.pt'))
            s_label = torch.tensor(torch.load(mit_path + 's_labels.pt'))
            v_label = torch.tensor(torch.load(mit_path + 'v_labels.pt'))[:400]-2
            f_label = torch.tensor(torch.load(mit_path + 'f_labels.pt'))[:400]-2
            q_label = torch.tensor(torch.load(mit_path + 'q_labels.pt'))[:400]-3

            data = torch.cat((v_set,q_set), dim=0)
            labels = torch.cat((v_label,q_label), dim=0)

            #Binary classification 
        # labels = F.one_hot(labels.long())
            
    elif task == "xai":
        
            
        if data_name=='mitbih':
            #data
            mit_path = "/home/hschung/xai/xai_timeseries/data/mit_bih_dataset/"
            n_set = torch.tensor(torch.load(mit_path + 'n_data.pt'))
            s_set = torch.tensor(torch.load(mit_path + 's_data.pt'))
            v_set = torch.tensor(torch.load(mit_path + 'v_data.pt'))[:400]
            f_set = torch.tensor(torch.load(mit_path + 'f_data.pt'))[:400]
            q_set = torch.tensor(torch.load(mit_path + 'q_data.pt'))[:400]
            
            #labels
            n_label = torch.tensor(torch.load(mit_path + 'n_labels.pt'))
            s_label = torch.tensor(torch.load(mit_path + 's_labels.pt'))
            v_label = torch.tensor(torch.load(mit_path + 'v_labels.pt'))[:400]-2
            f_label = torch.tensor(torch.load(mit_path + 'f_labels.pt'))[:400]-2
            q_label = torch.tensor(torch.load(mit_path + 'q_labels.pt'))[:400]-3

            data = torch.cat((v_set,q_set), dim=0)
            labels = torch.cat((v_label,q_label), dim=0)
            
        elif data_name=='toydata':
            class0_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset/time_domain_class0.csv').values)
            class1_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset/time_domain_class1.csv').values)
            data = torch.cat([class0_data,class1_data], dim=0)[:,:192]
            
            class0_label = torch.stack((torch.ones(len(class0_data)), torch.zeros(len(class0_data))),dim=1)
            class1_label = torch.stack((torch.zeros(len(class1_data)), torch.ones(len(class1_data))),dim=1)
        
            labels = torch.cat((class0_label, class1_label), dim=0)
            
        elif data_name=='toydata2':
            if set=='train':
                class0_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset2/train_random_dataset_class1.csv').values)
                class1_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset2/train_random_dataset_class2.csv').values)
                data = torch.cat([class0_data,class1_data], dim=0)[:,:192]
                
                class0_label = torch.stack((torch.ones(len(class0_data)), torch.zeros(len(class0_data))),dim=1)
                class1_label = torch.stack((torch.zeros(len(class1_data)), torch.ones(len(class1_data))),dim=1)
            
                labels = torch.cat((class0_label, class1_label), dim=0)
            
            elif set=='test':
            
                class0_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset2/test_random_dataset_class1.csv').values)
                class1_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset2/test_random_dataset_class2.csv').values)
                data = torch.cat([class0_data,class1_data], dim=0)[:,:192]
                
                class0_label = torch.stack((torch.ones(len(class0_data)), torch.zeros(len(class0_data))),dim=1)
                class1_label = torch.stack((torch.zeros(len(class1_data)), torch.ones(len(class1_data))),dim=1)
            
                labels = torch.cat((class0_label, class1_label), dim=0)
            else:
                class0_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset2/val_random_dataset_class1.csv').values)
                class1_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset2/val_random_dataset_class2.csv').values)
                data = torch.cat([class0_data,class1_data], dim=0)[:,:192]
                
                class0_label = torch.stack((torch.ones(len(class0_data)), torch.zeros(len(class0_data))),dim=1)
                class1_label = torch.stack((torch.zeros(len(class1_data)), torch.ones(len(class1_data))),dim=1)
            
                labels = torch.cat((class0_label, class1_label), dim=0)
            
        elif data_name=='toydata3':
            class0_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset3/toy_dataset3_class0.csv').values)
            class1_data = torch.tensor(pd.read_csv('/home/smjo/xai_timeseries/data/toy_dataset3/toy_dataset3_class1.csv').values)
            data = torch.cat([class0_data,class1_data], dim=0)[:,:192] 

            
            class0_label = torch.stack((torch.ones(len(class0_data)), torch.zeros(len(class0_data))),dim=1)
            class1_label = torch.stack((torch.zeros(len(class1_data)), torch.ones(len(class1_data))),dim=1)
        
            labels = torch.cat((class0_label, class1_label), dim=0)
            
        else:
            print("Wrong data name")    

        #Binary classification 
        # labels = F.one_hot(labels.long())
    
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