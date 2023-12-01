import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from matplotlib import cm
import matplotlib.ticker as tick
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from matplotlib.colors import ListedColormap
import matplotlib
import re
def sort_list(list_):
   id=[i.split('/')[-1] for i in list_]
   id=[int(re.findall(r'\d+', i)[-1]) for i in id]
   return [x for _,x in sorted(zip(id,list_))]

def optimal_k_gmm_bic(x_cluster_target,kmin=2,kmax=12):
  try:
      assert len(x_cluster_target)>=kmax
  except:
      kmax=len(x_cluster_target)
  clusters_num=np.array([i for i in range(kmin,kmax+1)])
  bics,min_bic= [[],0]
  for i in range(kmin,kmax+1): # test the AIC/BIC metric between 1 and 10 components
    gmm = GaussianMixture(n_components = int(i),max_iter=1000)
    gmm.fit(x_cluster_target)
    bic = gmm.bic(x_cluster_target)
    bics.append(bic)
    if bic < min_bic or min_bic == 0:
      min_bic = bic
  opt_bic=clusters_num[bics.index(min(bics))]
  fig=plt.figure(figsize=(7.5,5),dpi=244,facecolor='white')
  plt.plot(clusters_num,bics)
  plt.xlabel('Number of clusters, k')
  plt.ylabel('Bayes Information Criterion')
  plt.scatter(opt_bic,min(bics),color='r',label='Optimal k')
  plt.legend()
  plt.close()
  return opt_bic,fig
def quatplot(y, z, quadrangles, values, ax=None, **kwargs):
    if not ax: ax = plt.gca()
    yz = np.c_[y,z]
    verts = yz[quadrangles]
    pc = matplotlib.collections.PolyCollection(verts, **kwargs)
    pc.set_array(values)
    ax.add_collection(pc)
    ax.autoscale()
    return pc
def showMeshPlot(nodes, elements, values, titlelabel=[r'$\sigma_{11}$ from FEM', r'Stress, $\sigma_{11}$ (MPa)'], discrete_bar=False,zero_white=False):
    y = nodes[:,0]
    z = nodes[:,1]

    fig, ax = plt.subplots(dpi=144, figsize=(10, 10))
    ax.set_aspect('equal')

    pc = quatplot(y, z, np.asarray(elements), values, ax=ax, edgecolor='face', cmap="rainbow")
    if discrete_bar:
        unique_labels = np.unique(values).astype(int)
        base_cmap = matplotlib.colormaps['rainbow']  # Updated line
        colors = base_cmap(np.linspace(0, 1, len(unique_labels)))
        if zero_white:
            colors[0] = (1, 1, 1, 1)  # Set the first entry to white
        cmap = ListedColormap(colors)  # Create a new ListedColormap
        pc.set_cmap(cmap)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        # Create a color bar with ticks at the center of each discrete color
        cbar = fig.colorbar(pc, cax=cax, boundaries=np.arange(unique_labels.min() - 0.5, unique_labels.max() + 1.5), ticks=np.arange(unique_labels.min(), unique_labels.max() + 1))
        cbar.set_label(titlelabel[-1], rotation=270, labelpad=20)
        cbar.set_ticklabels(unique_labels)  # Set tick labels to match the unique labels
    else:
        # Create the continuous color bar
        cbar = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
        ticks = np.linspace(values.min(), values.max(), num=8)
        cbar.set_ticks(ticks)
        cbar.set_label(titlelabel[-1], rotation=270, labelpad=20)
        cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])

    ax.set(title=titlelabel[0], xlabel=r'x, $mm$', ylabel=r'y, $mm$')
    plt.close(fig)
    return fig, ax
def GMM_1D_plot(nodes, elements, values, gmm1,titlelabel=[r'Heatmap of $\sigma_{11}$ Component from FEM in MPa',r'Gaussian Mixture Densities for $\sigma_{11}$', r'']):
    fig = plt.figure(figsize=(12, 5), dpi=244, facecolor='white')
    ax = fig.add_subplot(121)   # Original stress
    y = nodes[:,0]
    z = nodes[:,1]
    pc = quatplot(y, z, np.asarray(elements), values, ax=ax, edgecolor='face', cmap="rainbow")
    # Adjust the colorbar to show min, max, and six intervals
    
    ticks = np.linspace(values.min(), values.max(), num=8)  # Creates 8 ticks: vmin, vmax, and 6 intervals
    cbar = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.linspace(values.min(), values.max(), num=8)
    cbar.set_ticks(ticks)
    cbar.set_label(titlelabel[-1], rotation=270, labelpad=20)
    cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])
    ax.set(title=titlelabel[0], xlabel=r'x, $mm$', ylabel=r'y, $mm$')
    #fig.subplots_adjust(left=0.12, right=0.97,bottom=0.21, top=0.9, wspace=0.5)
    # plot 1: data + best-fit mixture
    ax = fig.add_subplot(122)
    M_best = gmm1
    X=values
    x = np.linspace(X.min(), X.max(), 1000)
    logprob = M_best.score_samples(x.reshape(-1, 1))
    responsibilities = M_best.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf=(pdf)#/(pdf.sum())
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    weights = np.ones_like(X) / len(X)
    ax.set_title(titlelabel[1])
    ax.hist(X, 100, density=True, weights=weights, histtype='stepfilled', alpha=0.3,orientation='horizontal')
    ax.plot(pdf,x, '-k',label='Mixture PDF')
    for i in range(pdf_individual.shape[-1]):
        ax.plot(pdf_individual[:,i],x, '--',label='Component PDF '+str(i+1))
    #ax.set_aspect('equal', 'box')
    ax.set_xlabel('Density, $p(\sigma_{11})$')
    ax.set_ylabel(r'$\sigma_{11}$, MPa')
    ax.set_yticks([])
    ax.legend()
    ax.set_xlim(0.9*pdf.min(),1.1*pdf.max())
    ax.set_ylim(X.min(),X.max())
    plt.close()
    return fig
def GMM2(cluster_target,kmin1=4,kmax1=10,kmin2=3,kmax2=12):
    phaseI_tar=cluster_target[:,[-1]]##
    #Primary stage
    k1,_=optimal_k_gmm_bic(phaseI_tar,kmin=kmin1,kmax=kmax1)##
    gmm1=GaussianMixture(n_components=k1,max_iter=1000,init_params='k-means++')##
    labels=gmm1.fit_predict(phaseI_tar)##
    #Clusters selection
    min_center_clusters=gmm1.predict(phaseI_tar.min().reshape(1, -1)).item()
    max_center_clusters=gmm1.predict(phaseI_tar.max().reshape(1, -1)).item()
    idices_min=np.where(labels==min_center_clusters)  
    idices_max=np.where(labels==max_center_clusters)
    #print(f'Min_idx:{min_center_clusters+1},Max_idx:{max_center_clusters+1}')
    if min_center_clusters==max_center_clusters:
        clusterCount = np.bincount(labels)
        min_size_clusters=np.argwhere(clusterCount==np.unique(clusterCount)[0])
        idices1=[]
        for i in min_size_clusters:
            idices1.append(np.where(labels==i))   
        idices1=np.hstack([idices1]).reshape(-1)
        min_2ed_size_clusters=np.argwhere(clusterCount==np.unique(clusterCount)[1])
        idices2=[]
        for i in min_2ed_size_clusters:
            idices2.append(np.where(labels==i))   
        idices2=np.hstack([idices2]).reshape(-1)           
        
        idices=np.hstack([idices1,idices2]).reshape(-1)
    else:
        idices=np.hstack([idices_max,idices_min]).reshape(-1)
    #Secondary stage
    phaseII_tar=cluster_target[idices,:].copy()
    #Normalized the stress dim
    last_column=phaseII_tar[:,-1]
    vmax,vmin,vscale=[last_column.max(),last_column.min(),phaseII_tar[:,:-1].max()]
    phaseII_tar[:,-1]=(last_column-vmin)/(vmax-vmin)*vscale#Normalize the final column
    k2,_=optimal_k_gmm_bic(phaseII_tar,kmin=kmin2,kmax=kmax2)
    #bic2_fig
    gmm2=GaussianMixture(n_components=k2,max_iter=1000)
    labels2=gmm2.fit_predict(phaseII_tar)
    #Reformulate the final labels
    final_labels=np.zeros([len(cluster_target),1])
    final_labels[idices]=(labels2+1).reshape(-1,1)

    return gmm1,gmm2,labels,idices,final_labels,[vmax,vmin,vscale]

def plot_projectionGMM(x_c,gmm2,retriClusterCenStress):
    input=x_c.copy()
    input=input[:,:3]
    input[:,-1]=(input[:,-1]-input[:,-1].min())/(input[:,-1].max()-input[:,-1].min())*input[:,0].max()
    xlim=[0,100]
    ylim=[0,100]
    spatial_means=gmm2.means_

    fig3d = plt.figure(figsize=(15,10),dpi=144,facecolor='white')
    #first
    ax = fig3d.add_subplot(121, projection='3d')   #Origianl stress
    fig3d.text(0.3, 0.75, '3D Scaled Stress Distribution across the FEM Domain', ha='center', va='center', fontsize=12)
    ax.scatter(spatial_means[retriClusterCenStress>0,0],spatial_means[retriClusterCenStress>0,1],spatial_means[retriClusterCenStress>0,-1], color='k', s=60,marker='+',label=r'Cluster Centers: Tensile Concentration Regions')
    ax.scatter(spatial_means[retriClusterCenStress<0,0],spatial_means[retriClusterCenStress<0,1],spatial_means[retriClusterCenStress<0,-1], color='k', s=60,marker='_',label=r'Cluster Centers: Compressive Concentration Regions')
    surf = ax.plot_trisurf( input[:,0],input[:,1], input[:,-1],alpha=0.4,cmap=cm.rainbow)
    #The local concentration of Gaussian via average projection
    N = 100
    col = np.linspace(xlim[0],xlim[1] , N)
    row = np.linspace(ylim[0],ylim[1] , N)
    Z = np.linspace(ylim[0],ylim[1] , N)
    colm, rowm, Zm= np.meshgrid(col, row, Z)

    samples = np.array([colm.ravel(),rowm.ravel(), Zm.ravel()]).T
    Z_score=gmm2.score_samples(samples)
    z_oper=Z_score.reshape(N,N,N)
    cset = ax.contourf(colm[:,:,0],rowm[:,:,0], z_oper.mean(-1), zdir='z', offset=-100, cmap=cm.rainbow)
    ax.set_zlim(xlim[0]-100,xlim[-1])
    ax.set_xlabel('$x$, $mm$')
    ax.set_ylabel('$y$, $mm$' )
    ax.set_zlabel('Scaled Stress Value')
    ax.legend(loc='upper right', fontsize="6", bbox_to_anchor=(0.5, 0.42, 0.5, 0.5)) 
    #ax.invert_yaxis()
    ticks=np.linspace(ylim[0],ylim[1],6)
    ticks=[int(i) for i in ticks]
    ax.set_yticks(ticks)

    ax.set_zticks(ticks)
    elev = 22 # Elevation angle in degrees
    azim = -76  # Azimuthal angle in degrees
    ax.view_init(elev=elev, azim=azim)
    #Second
    ax = fig3d.add_subplot(122)
    im3=ax.contourf(row,col,z_oper.mean(-1), cmap=cm.rainbow)
    ax.scatter(spatial_means[retriClusterCenStress>0,0],spatial_means[retriClusterCenStress>0,1], color='k',marker='+')
    ax.scatter(spatial_means[retriClusterCenStress<0,0],spatial_means[retriClusterCenStress<0,1], color='k',marker='_')

    fig3d.text(0.71, 0.75, r'Mean Projection of Log Likelihood, $\log(\mathbf{GMM}^2(\cdot))$', ha='center', va='center', fontsize=12)
    ax.set_aspect('equal', 'box')
    ax.set_yticks(ticks)

    ax.set_xlabel('$x$, $mm$')
    ax.set_ylabel('$y$, $mm$' )
    cbar3 = fig3d.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    plt.close()
    return fig3d,Z_score