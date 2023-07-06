import os 
from argparse import ArgumentParser


def model_path(dataset, model_type, num_quantizers):
    
    if dataset == 'mitbih':
        if model_type == 'cnn':
            if num_quantizer ==1:
                classification_pth = 
                vqvae_pth = '/home/smjo/xai_timeseries/vqvae/saved_models/hard_mitbih/1/model_280.pt'
            elif num_quantizers ==2:
                classification_pth=
                vqvae_pth = '/home/smjo/xai_timeseries/vqvae/saved_models/hard_mitbih/2/model_290.pt'
            elif num_quantizers ==4:
                classification_pth=
                vqvae_pth = '/home/smjo/xai_timeseries/vqvae/saved_models/hard_mitbih/4/model_290.pt'
            else:
                classification_pth=
                vqvae_pth = '/home/smjo/xai_timeseries/vqvae/saved_models/hard_mitbih/8/model_290.pt
'
                
        elif model_type == 'transformer':
            if 
        elif model_type == 'cnn_transformer':
            
    elif dataset == 'ptb':
    elif dataset  == 'peak':
    elif dataset == 'flat':
    else:
        print("wrong dataset")
    
    return classification_pth, vqvae_pth