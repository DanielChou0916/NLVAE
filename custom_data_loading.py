import torch
from torch import nn, optim
from torchvision import transforms, models
from tqdm.auto import tqdm
import random
import torch
import torch.nn.functional as F
import torch.distributions as dist
import os
import re

def create_folder_if_not_exists(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
        #print(f"Folder '{folder_path}' successfully created or already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

def sort_strings_by_last_component(file_paths):
    def extract_number(file_path):
        # Extract the number from the file name
        match = re.search(r'stress_(\d+).pt', file_path)
        return int(match.group(1)) if match else 0

    return sorted(file_paths, key=extract_number)

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

def get_sorted_subfolders(global_folder):
    if not os.path.isdir(global_folder):
        return "The specified folder does not exist."

    subfolders = [f.name for f in os.scandir(global_folder) if f.is_dir()]
    subfolders.sort(key=natural_sort_key)
    return subfolders
def batch_tensor_standardization(batch_data):
    # Assume 'batch_data' is the batch of stress maps with shape (batch_size, 1, 100, 100)
    # Compute mean and standard deviation for each stress map
    mean = torch.mean(batch_data, dim=(2, 3), keepdim=True)
    std = torch.std(batch_data, dim=(2, 3), keepdim=True)

    # Normalize each stress map individually
    normalized_data = (batch_data - mean) / (std+ 1e-6)
    return normalized_data,mean,std

def report_tensor_list(tensor_path):
    
    #UC_path='/kaggle/input/mixed-stress-path/Set_III_UCTensors'
    #tensor_path='/kaggle/input/scaled-stress-with-channel-regarding2dataset/Set3'
    file_list=[]
    for dirname, _, filenames in os.walk(tensor_path):
        for filename in filenames:
            file_list.append(os.path.join(dirname, filename))
    #print('The data set size:',len(file_list))
    #Remove the initial state

    for i in file_list:
    #j=0
        pt_name=i.split('\\')[-1]
        if re.findall(r'\d+', pt_name)[-1]=='0':
            file_list.remove(i)
    return file_list


def list_splitting(target_list,proportion=0.2,shuffle=True):
    if shuffle:
        random.shuffle(target_list)
    split_idx=int((1-proportion)*len(target_list))
    tarin_list=target_list[:split_idx]
    test_list=target_list[split_idx:]
    return tarin_list,test_list

def max_min_norm(a):
    return (a-a.min())/(a.max()-a.min())


def load_a_batch_tensor(sub_file_list,by_channel=False,channel_idx=0):
    '''
    This function is designed to load the batch tensor from a list
    1. The batch size=len(sub_file_list)
    2. by_channel: boolean param used to specify weather to just load the channels
                   E.g., only want to use specific stress components to train the model.
    3. channel_idx: which channel we want to use. Can be a single int or a list.
    '''
    try:
        assert type(sub_file_list) == list
    except:
        sub_file_list=[sub_file_list]
    batch_tensor=torch.tensor([])
    if not by_channel:
        for tensor in sub_file_list:
            stress=torch.load(tensor).unsqueeze(0)
            batch_tensor=torch.cat([batch_tensor,stress])
    else:
        for tensor in sub_file_list:
            stress=torch.load(tensor)#.unsqueeze(0)
            stress=stress[channel_idx,:,:].unsqueeze(0).unsqueeze(0)
            batch_tensor=torch.cat([batch_tensor,stress])
    #if load channels by a list, the dim will be 5, so squeeze the extra dim.
    #Since one of the unsqueeze operation in 'else:' is redundant 
    if len(batch_tensor.shape)>4:
        batch_tensor=batch_tensor.squeeze(1)
    return batch_tensor.type('torch.FloatTensor')

#0426 VAE_cov
def diagonal_decomposition(M):
    off_diag = M.clone()
    off_diag.diagonal(dim1=-1, dim2=-2).zero_()
    diag=M.clone()
    diag=diag.diagonal(dim1=-1, dim2=-2)
    return diag,off_diag