import torch
import numpy as np
from Latent_clustering_functions import *
from scipy.spatial.distance import squareform
#from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import wasserstein_distance
from collections import OrderedDict
def empirical_emd(x1,x2,joint_norm=True):
    """
    Input: the sklearn samples and x2 samples
    output: the empirical EMD 
    """
    #Joint normal
    if joint_norm:
        min_val = np.min([x1.min(0),x2.min(0)], axis=0)
        max1val = np.max([x1.max(0),x2.max(0)], axis=0)
        Y1=(x1-min_val)/(max1val-min_val)  
        Y2=(x2-min_val)/(max1val-min_val)
    else:
        Y1,Y2=[x1,x2]
    d = cdist(Y1, Y2)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / x1.shape[0]
def max_min_normalization(x):
    #x is NXT sequence. N is sample number and T is step.
    return (x-x.min())/(x.max()-x.min())

def EMD_within_series(x,ref_mode='Initial'):
    #x is NXT sequence. N is sample number and T is step.
    _,T=x.shape
    if ref_mode=='Initial':
        #empirical_emd(x[:,t-1],x[:,t],joint_norm=False)
        return torch.tensor([wasserstein_distance(x[:,0],x[:,t]) for t in range(1,T)])
    else:
        return torch.tensor([wasserstein_distance(x[:,t-1],x[:,t]) for t in range(1,T)])
    
def delta_EMD_array(x):
    #input x (N,T,D)
    #Return EMD_x (T-1,D)
    EMD_x=torch.tensor([])
    for j in range(x.shape[-1]):
        #x_j=max_min_normalization(x[:,:,j])
        x_j=x[:,:,j]
        EMD_xj=EMD_within_series(x_j,ref_mode='Dyn')
        EMD_x=torch.cat([EMD_x,EMD_xj.reshape(-1,1)],dim=-1)
    return EMD_x
def EMD_within_series_w(x, weights, ref_mode='Initial'):
    # x is an NXT sequence. N is the sample number and T is the time step.
    # weights is a T size array representing the weights at each time step for a single dimension
    N, T = x.shape
    #weights_array=torch.ones(N)
    if ref_mode == 'Initial':
        return torch.tensor([wasserstein_distance(weights[0]*x[:, 0], weights[t]*x[:, t]) for t in range(1, T)])
    else:
        return torch.tensor([wasserstein_distance(weights[t-1]*x[:, t-1], weights[t]*x[:, t]) for t in range(1, T)])
def delta_EMD_array_w(x, weights):
    # input x (N,T,D)
    # Return EMD_x (T-1,D)
    EMD_x = torch.tensor([])
    for j in range(x.shape[-1]):
        x_j = max_min_normalization(x[:,:,j])
        EMD_xj = EMD_within_series_w(x_j, weights[:, j])
        EMD_x = torch.cat([EMD_x, EMD_xj.reshape(-1,1)], dim=-1)
    return EMD_x


#Calculate the DTW tensor
def dtw_matrix_for_EMDarrays(latent, fabrics):
    #Input 2 tensors
    #1. EMD Latents, specify the BC and stress components. [T-1,Dz]
    #2. EMD fabrics, specify the BC. [T-1,Df]
    Dz,Df=[latent.shape[-1],fabrics.shape[-1]]
    dtw_distance=torch.zeros([Dz,Df])
    for i in range(Dz):
        for j in range(Df):
            z=latent[:,i]
            f=fabrics[:,j]
            dtw_distance[i,j]=Dynamic_Time_Warping(z,f)[0]
    return dtw_distance

def DTW_tensor_forEMDzf(BC_EMD_z,BC_EMD_f):
    #BC_EMD_z:(Stress=4,T-1=19,Dz=32)
    #BC_EMD_f:(T-1,Df=43)
    BC_S_EMDdtw=[]
    S,_,_=BC_EMD_z.shape
    for i in range(S):
        BC_S_EMDdtw.append(dtw_matrix_for_EMDarrays(BC_EMD_z[i],BC_EMD_f))
    return torch.stack(BC_S_EMDdtw,dim=0)


#UT_DTW in shape ()
def ranking_check(BC_DTW):
    mean_DTWwrt_zs=torch.mean(BC_DTW,dim=(1))
    rank_of_features=mean_DTWwrt_zs.sort(dim=-1).indices+1
    print(rank_of_features[:,:10])
    top_indices=[5,8,10]
    for top_idx in top_indices:
    #top_idx=10
        uniques,counts=rank_of_features[:,:top_idx].unique(return_counts=True)
        uniques[torch.argwhere(counts==4)].reshape(-1)
        print(f'For top {top_idx} fabrics at each channel:{uniques[torch.argwhere(counts==4)].reshape(-1)}')

def ranking_check_addWin_last_stage(BC_DTW,BC_weights_ave,top_indices=[5,8,10],only_check=True):
    mean_DTWwrt_zs=(BC_weights_ave.unsqueeze(-1)*BC_DTW).mean(dim=1) 
    rank_of_features=mean_DTWwrt_zs.sort(dim=-1).indices+1
    print(rank_of_features[:,:top_indices[-1]])
    top_fabric_dict={}
    for top_idx in top_indices:
    #top_idx=10
        uniques,counts=rank_of_features[:,:top_idx].unique(return_counts=True)
        print(f'For top {top_idx} fabrics at each channel:{uniques[torch.argwhere(counts==4)].reshape(-1)}')
        if not only_check:
            top_fabric_dict[top_idx]=uniques[torch.argwhere(counts==4)].reshape(-1)
    if not only_check:
        return top_fabric_dict