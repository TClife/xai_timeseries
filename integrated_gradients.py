import torch 
from data import load_data
from models import VQ_Classifier
from torch.utils.data import DataLoader, Dataset, random_split 
import pandas as pd 
from argparse import ArgumentParser
from captum.attr import LayerIntegratedGradients

torch.set_num_threads(32) 
torch.manual_seed(911) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Create args parser for labels and batch size 
parser = ArgumentParser() 
parser.add_argument('--labels', type=int, default=0)
parser.add_argument('--num_features', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--classification_model', type=str, default="/home/hschung/xai/xai_timeseries/classification_models/ptb_conv_nonoverlap_128_8/model_290.pt")
parser.add_argument('--vqvae_model', default = "/home/hschung/xai/xai_timeseries/vqvae_models/ptb_residual_vqvae_nonoverlap_16_8/model_300.pt")
parser.add_argument('--task', type=str, default='xai', help="Task being done")
parser.add_argument('--dataset', type=str, default="ptb")
parser.add_argument('--plot_dataset', type=bool, default=True)
parser.add_argument('--feature_selection', type=str, default='highest_weights')
parser.add_argument('--model_type', type=str, default="cnn")
parser.add_argument('--auc_classification', type=bool, default=False)

args = parser.parse_args() 

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
    positions = class_args.positions,
    mask = class_args.mask,
    auc_classification = args.auc_classification,
    model_type = class_args.model_type
).to(device)

#load classification model
net.load_state_dict(classification_model['model_state_dict'])

#find masking regions 
masked_regions = net.mask_region(ds.data[:64,:].to(device))

#Functions for integrated gradients 

def output_function(inputs):
  return net.ig(inputs)[0][:,0] #[1,4] -> [4] get the fc-layer output from embedded input

#define model input and output
for _, (data, labels) in enumerate(test_loader):
    data = data.unsqueeze(1).float()
    labels = labels.type(torch.LongTensor)
    data, labels = data.to(device), labels.to(device)
    output, codebook_tokens, recon, input= net(data)

#instantiate integrated gradients
lig = LayerIntegratedGradients(output_function, net.embedding)

#construct original and baseline input
baseline_tokens = torch.zeros((codebook_tokens.shape)) 
for i in range(len(masked_regions)):
    baseline_tokens[:,:,i] = masked_regions[i]
    
baseline_tokens = baseline_tokens.to(device)
print(f'original input: {codebook_tokens}')
print(f'baseline input: {baseline_tokens}')

#Compute attributions 
attributions, delta = lig.attribute(inputs = codebook_tokens, baselines = baseline_tokens, return_convergence_delta=True)

#Compute attributions for each token 
def summarize_attributions(attributions):

    attributions = torch.sum(attributions, dim=2)
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    
    return attributions

attributions_sum = summarize_attributions(attributions) #[1, 6, 768] -> [6]
print(attributions_sum.size())
print(attributions_sum)

#find top-2 max values and indices 
top_2_values, top_2_indices = torch.topk(attributions_sum, 2, dim=1) #top-2 values and indices for each sample in testset 

#Final Count of important positions 
from collections import Counter
top_indices = [int(item) for sublist in top_2_indices for item in sublist]
count = Counter(top_indices).most_common()

print(f'Final Count: {count}')