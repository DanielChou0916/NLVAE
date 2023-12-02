import torch

def spearman_correlation_vectorized(a,pair_wise=False):
    n,d=a.size()
    ranks=a.argsort(dim=0,descending=True).argsort(dim=0)+1
    diff=(ranks.unsqueeze(2)-ranks.unsqueeze(1))
    if pair_wise:
        pair_list=[[i,j] for i in range(d) for j in range(i+1,d)]
        pair_wise_rank=(ranks[:,pair_list]).permute(1, 0, -1)
        pair_wise_rank.shape
        pair_wise_diff=pair_wise_rank[:,:,0]-pair_wise_rank[:,:,1]
        return 1-6*(pair_wise_diff**2).sum(1)/(n*(n**2-1))
    else:
        return 1-6*(diff**2).sum(0)/(n*(n**2-1))
    

def spearman_correlation(X, Y):
    # Calculate ranks
    rank_X = X.argsort(descending=True).argsort()+1
    rank_Y = Y.argsort(descending=True).argsort()+1
    print(rank_X)
    # Calculate average ranks
    #avg_rank_X = (rank_X.float() + 1).mean()
    #avg_rank_Y = (rank_Y.float() + 1).mean()

    # Calculate differences in ranks
    diff = rank_X - rank_Y
    print(diff)
    # Calculate sum of squared differences
    sum_squared_diff = (diff ** 2).sum()

    # Calculate n (number of data points)
    n = float(len(X))

    # Calculate Spearman's rank correlation coefficient
    correlation = 1.0 - (6.0 * sum_squared_diff) / (n * (n ** 2 - 1.0))

    return correlation