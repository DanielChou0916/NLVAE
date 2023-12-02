import torch

#a = torch.Tensor([[0,0],[0,1],[1,0],[1,1]])

def pair_wise_DC(a,idx1,idx2,elp=1e-6):
    n,_=a.size()
    x=a[:,[idx1]]
    y=a[:,[idx2]]
    X=torch.cdist(x,x,p=1)
    Y=torch.cdist(y,y,p=1)
    ddx=X-X.mean(0, keepdim=True)-X.mean(1, keepdim=True)+X.mean()
    ddy=Y-Y.mean(0, keepdim=True)-Y.mean(1, keepdim=True)+Y.mean()
    dCovxy2=((ddx*ddy).sum()/(n**2))
    dCovxx2=((ddx*ddx).sum()/(n**2))
    dCovyy2=((ddy*ddy).sum()/(n**2))
    dCorxy2=dCovxy2/((dCovxx2*dCovyy2)**0.5+elp)
    return dCorxy2.sqrt(),dCovxy2.sqrt()

def Distance_CorrCorv(a, sum_offdiag_upper=False):
    "Return a upper triangular matrix"
    _,d=a.size()
    if sum_offdiag_upper:
        sDCor,sDCov=[0,0]
        for i in range(a.shape[-1]):
            for j in range(i+1,a.shape[-1]):
                dCorxy,dCovxy=pair_wise_DC(a,i,j)
                sDCor+=dCorxy
                sDCov+=dCovxy
        return sDCor,sDCov
    else:
        DCor=torch.zeros(d,d).to(a.device)
        DCov=torch.zeros(d,d).to(a.device)
        for i in range(d):
            for j in range(i+1,d):
                dCorxy,dCovxy=pair_wise_DC(a,i,j)
                DCor[i,j]=dCorxy
                DCov[i,j]=dCovxy
        return DCor,DCov
    
def Distance_CorrCorv_vectorized(a,correlation=True):
    '''
    It's designed to handle the multi-features matrix for the distance correlation
    The input should be a 2D torch tensor contains at least two features
    The outputs are square-root distance correlation and covariance, presented in Pearson correlation style.
    '''
    n,_=a.size()
    dist_a=(a.unsqueeze(0)-a.unsqueeze(1)).abs()
    a_til=dist_a-dist_a.mean(0,keepdim=True)-dist_a.mean(1,keepdim=True)+dist_a.mean((0,1),keepdim=True)

    Cov = (torch.einsum('ijm,ijk->mk',a_til,a_til)) / (n ** 2)
    if correlation:
        Cor=Cov/(Cov.diag().reshape(-1,1)*Cov.diag().reshape(1,-1)).sqrt()
        return Cor.sqrt(),Cov.sqrt()
    else:
        return Cov.sqrt()
    
