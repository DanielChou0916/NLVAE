import os
import numpy as np
import torch

import torch.distributions as dist


def ensure_positive_definite(matrix, eps=1e-4):
    # Perform eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    
    # Ensure all eigenvalues are positive
    positive_eigenvalues = torch.clamp(eigenvalues, min=eps)
    
    # Reconstruct the matrix with modified eigenvalues
    positive_definite_matrix = torch.matmul(
        eigenvectors, torch.matmul(torch.diag(positive_eigenvalues), eigenvectors.T)
    )
    
    return positive_definite_matrix

def ensure_positive_definite_batch(matrix, eps=1e-6):
    # Perform eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    
    # Ensure all eigenvalues are positive
    positive_eigenvalues = torch.clamp(eigenvalues, min=eps)
    
    # Reconstruct the matrices with modified eigenvalues
    positive_definite_matrices = torch.matmul(
        eigenvectors, torch.matmul(torch.diag_embed(positive_eigenvalues), eigenvectors.transpose(-2, -1))
    )
    
    return positive_definite_matrices

# Example usage:
#covariance = torch.rand(32, 32)  # Replace with your covariance matrix
#positive_definite_covariance = ensure_positive_definite(covariance)

def multi_vars_SkewNormal_density_estimate(values,check=False):
    '''
    Estimate the probability density by SkewNormal density function
    Input: A 2D tensor contains n samples and d features
    check: Used to visualize the density estimation in smooth curve
    '''
    mean = torch.mean(values,0)
    std = torch.std(values,0)
    skewness = torch.mean((values - mean) ** 3,0) / torch.pow(std, 3)
    residuals = values - mean
    standard_normal = dist.Normal(0, 1)
    kernel_values = 2 * standard_normal.cdf((residuals / std) *skewness) *(1/(std*(2*torch.pi)**0.5))* torch.exp(-0.5 * (residuals / std).pow(2))
    pdf = kernel_values / (torch.sum(kernel_values,0))
    #Check
    if check:
        bins=torch.tensor(np.linspace((values.min(0).values),(values.max(0).values),100))
        r_check=bins-mean
        k_check=2 * standard_normal.cdf((r_check / std) * skewness) *(1/(std*(2*torch.pi)**0.5))* torch.exp(-0.5 * (residuals / std).pow(2))
        pdf_check=k_check / (torch.sum(k_check,0))
        return pdf,kernel_values,[bins,k_check,pdf_check]
    else: 
        return pdf,kernel_values

def estimate_total_joint_density_skew_normal_training(values):
    mean = torch.mean(values, dim=0)
    residuals = values - mean
    covariance = torch.cov(residuals.T)
    skewness = torch.mean((residuals) ** 3, dim=0) / torch.pow((torch.diagonal(covariance))**0.5, 3)

    standard_normal = dist.Normal(0, 1)
    #cdf term
    cdf=standard_normal.cdf((residuals/torch.diagonal(covariance).sqrt())@ (skewness.unsqueeze(0)).T)
    #phi(x) term
    try:
        multinormal=dist.MultivariateNormal(mean, covariance)
    except:
        covariance=ensure_positive_definite(covariance)
        multinormal=dist.MultivariateNormal(mean, covariance)
    pdf=multinormal.log_prob(values).exp()

    joints_kernel=2*pdf.reshape(cdf.shape)*cdf
    normalized_joints=joints_kernel/(joints_kernel.sum())
    return normalized_joints,joints_kernel

def estimate_total_joint_density_skew_normal(values,mean,covariance,skewness):
    #mean = torch.mean(values, dim=0)
    residuals = values - mean
    #covariance = torch.cov(residuals.T)
    #skewness = torch.mean((residuals) ** 3, dim=0) / torch.pow((torch.diagonal(covariance))**0.5, 3)

    standard_normal = dist.Normal(0, 1)
    #cdf term
    cdf=standard_normal.cdf((residuals/torch.diagonal(covariance).sqrt())@ (skewness.unsqueeze(0)).T)
    #phi(x) term
    try:
        multinormal=dist.MultivariateNormal(mean, covariance)
    except:
        covariance=ensure_positive_definite(covariance)
        multinormal=dist.MultivariateNormal(mean, covariance)
    pdf=multinormal.log_prob(values).exp()

    joints_kernel=2*pdf.reshape(cdf.shape)*cdf
    normalized_joints=joints_kernel/(joints_kernel.sum())
    return normalized_joints,joints_kernel



def estimate_total_joint_density(data):
    # Calculate the mean and covariance of the data
    mean = torch.mean(data, dim=0)
    covariance = torch.cov(data.T)
    try:
        multinormal=dist.MultivariateNormal(mean, covariance)
    except:
        covariance=ensure_positive_definite(covariance)
        multinormal=dist.MultivariateNormal(mean, covariance)

    # Evaluate the log probability density at the data points
    log_density = multinormal.log_prob(data)

    # Convert log density to tensor
    log_density_tensor = log_density.reshape(-1,1)#.detach()

    # Reshape the density tensor to match the input data shape
    joint_density = torch.exp(log_density_tensor)#.reshape(data.shape)
    normalized_joints=joint_density/(joint_density.sum())
    return normalized_joints, joint_density

def total_correlation_estimation(marginal,joint,weight=True,eps=0):
    '''
    This function can be used for 2D arrays
    Or 3D batch distributions=> mainly applied in the pair-wise vectorized calculations
    '''
    
    joint_log=torch.log(joint+eps)
    marginal_log = torch.log(marginal+eps)
    marginal_log_sum=marginal_log.sum(dim=-1,keepdim=True)
    if weight:
        #total_corr = joint*(joint_log - marginal_log_sum)
        total_corr=joint*(joint/(marginal.prod(-1,keepdim=True))+eps).log()
    else:
        #total_corr = (joint_log - marginal_log_sum)
        total_corr=(joint/(marginal.prod(-1,keepdim=True))+eps).log()
    return total_corr

def pair_wise_MI_upper(z,full_cov=False):
    '''
    Input, z in (n,d)
    Output, mi_upper array, abs sum of array
    '''
    _,d=z.size()
    mi_upper=torch.zeros(d,d).to(z.device)
    _,marginal_kernel=multi_vars_SkewNormal_density_estimate(z)
    if full_cov:
        for i in range(d):
            for j in range(d):
                marginal_kernel_=marginal_kernel[:,[i,j]]
                _,joint_kernel=estimate_total_joint_density_skew_normal_training(z[:,[i,j]])
                mi=total_correlation_estimation(marginal_kernel_,joint_kernel).abs().sum()
                mi_upper[i,j]=mi
    else:        
        for i in range(d):
            for j in range(i+1,d):
                marginal_kernel_=marginal_kernel[:,[i,j]]
                _,joint_kernel=estimate_total_joint_density_skew_normal_training(z[:,[i,j]])
                mi=total_correlation_estimation(marginal_kernel_,joint_kernel).abs().sum()
                mi_upper[i,j]=mi
    return mi_upper, mi_upper.sum()

def pair_wise_MI_vectorized(z,full_cov=False):
    _,d=z.size()
    if full_cov:
        idx_list=[[i,j] for i in range(d) for j in range(d)]
    else:
        idx_list=[[i,j] for i in range(d) for j in range(i+1,d)]
    z_expand=z[:,idx_list]
    z_expand=z_expand.permute(1, 0, -1)
    #Batch statistics
    batch_mean = torch.mean(z_expand, dim=1,keepdim=True)
    residuals = z_expand - batch_mean
    batch_covariance=(residuals.mT@residuals/(residuals.shape[1]-1))
    batch_std=(torch.diagonal(batch_covariance,dim1=1,dim2=2)).unsqueeze(1)**0.5
    batch_skewness= torch.mean((residuals) ** 3, dim=1,keepdim=True) / torch.pow(batch_std, 3)

    #z_expand[300]
    # Cdf calculation
    standard_normal = dist.Normal(0, 1)
    pair_cdf=standard_normal.cdf(((residuals/batch_std)@(batch_skewness).mT))
    #pair_cdf[300]
    #check
    #standard_normal.cdf(((residuals[300]/batch_std[300])@(batch_skewness[300].unsqueeze(0)).mT))
    # Pdf calculation
    try:
        batch_multinormal=dist.MultivariateNormal(batch_mean.squeeze(1), batch_covariance)
    except:
        batch_covariance=ensure_positive_definite_batch(batch_covariance, eps=1e-6)
        batch_multinormal=dist.MultivariateNormal(batch_mean.squeeze(1), batch_covariance)
    pair_pd=batch_multinormal.log_prob(z_expand.permute(1, 0, 2)).exp()
    pair_pd=pair_pd.permute(-1,0).unsqueeze(-1)
    #pair_pd[400]
    #Check
    #multinormal=dist.MultivariateNormal(batch_mean[400], batch_covariance[400])
    #pd=multinormal.log_prob(z_expand[400]).exp().reshape(-1,1)
    #pair_joint_kernels
    pair_joints_kernel=2*pair_pd*pair_cdf
    #pair_joints_kernel[250]
    #Check
    #estimate_total_joint_density_skew_normal_training(z_expand[250])[-1]
    #pair_joints_kernel.shape
    # Pair marginal
    _,marginal_kernels=multi_vars_SkewNormal_density_estimate(z)
    pair_marginals=marginal_kernels[:,idx_list]
    pair_marginals=pair_marginals.permute(1, 0, -1)
    mi=total_correlation_estimation(pair_marginals,pair_joints_kernel)
    return mi
#pair_marginals.shape