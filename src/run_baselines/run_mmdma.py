"""
Modified from the PyTorch implementation of MMD-MA: https://bitbucket.org/noblelab/2020_mmdma_pytorch

Usage:
    cd src
    python run_baselines/run_mmdma.py --dataset t_branch
"""
import numpy as np
import math
import sys
import os
import argparse

import matplotlib.cm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

import torch
from torch.utils import data
import torch.optim
from torch.autograd import Variable
import torch.nn as nn
import torch.cuda

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from scipy.spatial.distance import cdist

import yaml
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from utils.utils import load_data, sorted_by_label
from utils.vis import plt_heatmap
from utils.metrics import transfer_accuracy, calc_domainAveraged_FOSCTTM

USAGE = """USAGE: manifold_align_mmd_pytorch.py <input_k1> <input_k2> <result_dir> <num_feat> <sigma> <lambda1> <lambda2>

Run MMD-MA algorithm training to align single-cell datasets:
<input_k1>: Input kernel for single-cell dataset 1
<input_k2>: Input kernel for single-cell dataset 2
<result_dir>: Directory for saving the alpha and beta weights learned by the algorithm
<num_feat>: Dimension size of the learned low-dimensional space [Recommended tuning values : 4,5,6]
<sigma>: Bandwidth paramteter for gaussian kernel calculation, set value to 0.0 to perform automatic calculation
<lambda1>: Parameter for penalty term [Recommended tuning values : 1e-03, 1e-04, 1e-05, 1e-06, 1e-07]
<lambda2>: Parameter for distortion term [Recommended tuning values : 1e-03, 1e-04, 1e-05, 1e-06, 1e-07]

The outputs of the code are alpha and beta weight matrices learned by the algorithm

To obtain the final embeddings:
Embeddings for single-cell dataset 1 = input_k1 x alpha matrix 
Embeddings for single-cell dataset 2 = input_k2 x beta matrix
"""



if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device ="cpu"

print("Running on",device)

def compute_pairwise_distances(x, y): #function to calculate the pairwise distances
  if not len(x.size()) == len(y.size()) == 2:
    raise ValueError('Both inputs should be matrices.')

  if list(x.size())[1] != list(y.size())[1]:
    raise ValueError('The number of features should be the same.')
  
  diff =  (x.unsqueeze(2) - y.t())
  diff = (diff ** 2).sum(1)
  return diff.t()

def gaussian_kernel_matrix(x, y, sigmas): #function to calculate Gaussian kernel
  beta = 1.0 / (2.0 * (sigmas.unsqueeze(1)))
  dist = compute_pairwise_distances(x, y)
  s = beta * (dist.contiguous()).view(1,-1)
  result =  ((-s).exp()).sum(0)
  return (result.contiguous()).view(dist.size())


def stream_maximum_mean_discrepancy(x, y,  sigmas, kernel=gaussian_kernel_matrix): #This function has been implemented  to caculate MMD value for large number of samples (N>5,000)
  n_x = x.shape[0]
  n_y = y.shape[0]
  
  n_small = np.minimum(n_x, n_y)
  n = (n_small // 2) * 2
  
  cost = (kernel(x[:n:2], x[1:n:2], sigmas)  + kernel(y[:n:2], y[1:n:2], sigmas)
          - kernel(x[:n:2], y[1:n:2], sigmas) - kernel(x[1:n:2], y[:n:2], sigmas)).mean()
  if cost.data.item()<0:
    cost = torch.FloatTensor([0.0]).to(device)
  return cost

def maximum_mean_discrepancy(x, y, sigmas, kernel=gaussian_kernel_matrix): #Function to calculate MMD value

  cost = (kernel(x, x, sigmas)).mean()
  cost += (kernel(y, y, sigmas)).mean()
  cost -= 2.0 * (kernel(x, y, sigmas)).mean()
  
  if cost.data.item()<0:
    cost = torch.FloatTensor([0.0]).to(device)
  
  return cost

def calc_sigma(x1,x2): #Automatic sigma calculation 
    const = 8
    mat = np.concatenate((x1,x2))
    dist = []
    nsamp = mat.shape[0]
    for i in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(mat[i,:], mat)), axis=1))
        dist.append(sorted(euc_dist)[1])
    
    sigma = np.square(const*np.median(dist))
    print("Calculated sigma:",sigma)
    return sigma

class manifold_alignment(nn.Module): #MMD objective function

  def __init__(self, nfeat, num_k1, num_k2, seed):
    super(manifold_alignment, self).__init__()
    #Initializing alpha and beta 
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
    else:
      torch.manual_seed(seed)
    self.alpha = Parameter(torch.FloatTensor(num_k1, nfeat).uniform_(0.0,0.1).to(device))
    self.beta = Parameter(torch.FloatTensor(num_k2, nfeat).uniform_(0.0,0.1).to(device))
    
    
  def forward(self, k1, k2, ip, sigmas, lambda1, lambda2):
    
    if sigmas == 0: #If the user does not specify sigma values for the kernel calculation, they will be caclulated automatically
      x1 = (torch.matmul(k1,self.alpha)).detach().cpu().numpy()
      x2 = (torch.matmul(k2,self.beta)).detach().cpu().numpy()
      
      sigma = calc_sigma(x1,x2)
      sigmas = torch.FloatTensor([sigma]).to(device)
    
    mmd = maximum_mean_discrepancy(torch.matmul(k1,self.alpha),torch.matmul(k2,self.beta), sigmas)
    #mmd = stream_maximum_mean_discrepancy(torch.matmul(k1,self.alpha),torch.matmul(k2,self.beta), sigmas) #remove comment and comment the previous line if number of samples are large (N>5,000)
    
    penalty = lambda1 * ((((torch.matmul(self.alpha.t(),torch.matmul(k1,self.alpha))) - ip).norm(2))
              + (((torch.matmul(self.beta.t(),torch.matmul(k2,self.beta))) - ip).norm(2)))
    
    distortion = lambda2 * ((((torch.matmul((torch.matmul(k1,self.alpha)),(torch.matmul(self.alpha.t(),k1.t()))))-k1).norm(2))
              + (((torch.matmul((torch.matmul(k2,self.beta)),(torch.matmul(self.beta.t(),k2.t()))))-k2).norm(2)))
    
    return mmd, penalty, distortion, sigmas


#Functions to plot function values
def plot_data(filename,k,i,epoch,obj,mmd,pen,dist,kernel,nfeat,sigma,lambda1,lambda2):
  plt.xlabel('Iteration')
  plt.ylabel('log(Function value)')
  plt.title('kernel:' + str(kernel)+', nfeat:'+str(nfeat)+',seed:'+str(k)+', sigma:'+str(sigma)+', lambda1:'+str(lambda1)+', lambda2:'+str(lambda2))
  
  plt.plot(obj, 'k--', label='Objective')
  plt.plot(mmd, 'r--', label='MMD')
  plt.plot(pen, 'b--', label='Penalty')
  plt.plot(dist, 'g--', label='Distortion')
  if i == epoch:
    plt.legend(loc='upper right')
  plt.savefig(filename)
  plt.close()

def nearestk_mapping(data1, data2, k=3):
    # search nearest neighbor by NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(data2)
    distances, indices = nbrs.kneighbors(data1)

    # construct mapping matrix
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    mapping = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(k):
            mapping[i, indices[i, j]] = 1 / k

    # row-wise normalize
    # mapping = mapping / mapping.sum(axis=1, keepdims=True)
    mapping = mapping / mapping.sum()
    return mapping

def nearestk_mapping_distance(data1, data2, k=3):
    # search nearest neighbor by NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(data2)
    distances, indices = nbrs.kneighbors(data1)

    # construct mapping matrix
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    mapping = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(k):
            mapping[i, indices[i, j]] = 1 / distances[i, j]

    # row-wise normalize
    # mapping = mapping / mapping.sum(axis=1, keepdims=True)
    mapping = mapping / mapping.sum()
    return mapping

def nearestk_mapping_symmetric(data1, data2, k=3):
    mapping1 = nearestk_mapping(data1, data2, k)
    mapping2 = nearestk_mapping(data2, data1, k)
    mapping = (mapping1 + mapping2.T) / 2
    return mapping

def nearestk_mapping_symmetric_distance(data1, data2, k=3):
    mapping1 = nearestk_mapping_distance(data1, data2, k)
    mapping2 = nearestk_mapping_distance(data2, data1, k)
    mapping = (mapping1 + mapping2.T) / 2
    return mapping
  
def input_kernel(data, type='linear'):
  if type == 'linear':
    return np.matmul(data, data.T)
  elif type == 'gaussian':
     return rbf_kernel(data)
  else:
    raise Exception("Invalid kernel type")

def norm(data, type='l2'):
  if type in ['l1', 'l2', 'max']:
    data = normalize(data, norm=type)
  elif type == 'zscore':
    data = StandardScaler().fit_transform(data)
  else:
    raise Exception("Invalid normalization type")

def run_mmdma(data1, data2, label1, label2, links, params, lambda_range, seed_range, acc_log_url, foscttm_log_url, save_path, save_data = True, save_fig = True, save_mmdma_log=True):  
    if save_data: 
      os.makedirs(os.path.join(save_path, 'mapping'), exist_ok=True)
      os.makedirs(os.path.join(save_path, 'integration'), exist_ok=True)   
    if save_mmdma_log:
      os.makedirs(os.path.join(save_path, 'mmdma_log'), exist_ok=True)

    print("Number of dimensions of latent space...",params["nfeat"]) # number features in joint embedding
    sigmas = torch.FloatTensor([params["sigma"]]).to(device)
    Ip = np.identity(params["nfeat"]).astype(np.float32)          #identity matrix of size nfeatxnfeat
    
    K1 = torch.from_numpy(input_kernel(data1, type=params["kernel"])).float().to(device)
    K2 = torch.from_numpy(input_kernel(data2, type=params["kernel"])).float().to(device)
    I_p = torch.from_numpy(Ip).to(device)
             
    for lbd in lambda_range:
      lbd1 = lbd[0]
      lbd2 = lbd[1]
      
      for seed in seed_range:
          print("Running seed = {}".format(seed))
                   
          obj_val=[]
          mmd_val=[]
          pen_val=[]
          dist_val=[]
          
          model = manifold_alignment(params["nfeat"], data1.shape[0], data2.shape[0], seed)
          model = model.to(device)

          optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,amsgrad=True)

          model.train()
      
          for i in range(params["epoch"] + 1): #Training takes place for 10,000 iterations
          
              optimizer.zero_grad()

              mmd, penalty, distortion, sigmas = model(K1, K2, I_p, sigmas, lbd1, lbd2)
              obj = mmd + penalty + distortion

              obj.backward()

              optimizer.step()
              
              obj_value = obj.data.item()
              mmd_value = mmd.data.item()
              pen_value = penalty.data.item()
              dist_value = distortion.data.item()
          
              if mmd_value > 0 : 
                  obj_val.append(math.log(obj_value))
                  mmd_val.append(math.log(mmd_value))
                  pen_val.append(math.log(pen_value))
                  dist_val.append(math.log(dist_value))

              if (i%200 == 0 or i==params["epoch"]): #the weights can be saved every 200 iterations
                  weights=[]
              
                  for p in model.parameters():
                      if p.requires_grad:
                          weights.append(p.data)
              
                  if save_mmdma_log:
                    plot_data(os.path.join(save_path, "mmdma_log", "Functions_{}.png".format(seed)), 
                              seed,i,params["epoch"],obj_val,mmd_val,pen_val,dist_val, params["kernel"], 
                              params["nfeat"], params["sigma"],lbd1,lbd2)

                    if (i==0 or i==params["epoch"]): #This saves the weights at the beginning and end of the training
                        np.savetxt(os.path.join(save_path, "mmdma_log", "alpha_hat_{}_{}.txt".format(seed, i)), weights[0].cpu().numpy())
                        np.savetxt(os.path.join(save_path, "mmdma_log", "beta_hat_{}_{}.txt".format(seed, i)), weights[1].cpu().numpy())  
          
          data1_new = torch.matmul(K1, weights[0]).cpu().numpy()
          data2_new = torch.matmul(K2, weights[1]).cpu().numpy()
          mapping = nearestk_mapping_symmetric(data1_new, data2_new, k=3)

          # save integration result
          if save_data: 
            np.savetxt(os.path.join(save_path, "mapping", "lbd1{}_lbd2{}_seed{}.txt".format(lbd1, lbd2, seed)), mapping)
            np.savetxt(os.path.join(save_path, "integration", "lbd1{}_lbd2{}_seed{}_data1.txt".format(lbd1, lbd2, seed)), data1_new)
            np.savetxt(os.path.join(save_path, "integration", "lbd1{}_lbd2{}_seed{}_data2.txt".format(lbd1, lbd2, seed)), data2_new)
          if save_fig:
            plt_heatmap(mapping, "lbd1{}_lbd2{}_seed{}".format(lbd1, lbd2, seed), show=False, 
                        save_url=os.path.join(save_path, "mapping_fig", "heatmap", "lbd1{}_lbd2{}_seed{}.png".format(lbd1, lbd2, seed)))

          # evaluate
          acc = transfer_accuracy(data1_new, data2_new, label1, label2)
          foscttm = calc_domainAveraged_FOSCTTM(data1_new, data2_new, links)

          # save acc & FOSCTTM error
          with open(acc_log_url, 'a') as f: f.write(f'lbd1{lbd1}_lbd2{lbd2}_seed{seed}\t{acc}\n')    
          with open(foscttm_log_url, 'a') as f: f.write(f'lbd1{lbd1}_lbd2{lbd2}_seed{seed}\t{foscttm}\n')   


def main(args):
    with open("./run_baselines/baseline.yaml", "r") as file:
        config = yaml.safe_load(file)
    params = config[args.dataset]
    print("Loading data...")
    assert os.path.exists(params["data_path"])

    data_url1 = os.path.join(params["data_path"], params["data_url1"])
    data_url2 = os.path.join(params["data_path"], params["data_url2"])
    assert os.path.isfile(data_url1) and os.path.isfile(data_url2)

    label_url1 = os.path.join(params["data_path"], params["label_url1"])
    label_url2 = os.path.join(params["data_path"], params["label_url2"])
    assert os.path.isfile(label_url1) and os.path.isfile(label_url2)

    data1 = load_data(data_url1, )
    data2 = load_data(data_url2, )
    print("data size: data1 = {}, data2 = {}".format(data1.shape, data2.shape))

    label1 = load_data(label_url1, ).astype(int)
    label2 = load_data(label_url2, ).astype(int)

    # create links for FOSCTTM, all datasets follow 1-1 correspondence
    links = np.array(list(zip([i for i in range(data1.shape[0])], [i for i in range(data2.shape[0])])))

    if params["norm"]:
        if params["norm"] == 'l2':
          data1 = normalize(data1, norm=params["norm"])
          data2 = normalize(data2, norm=params["norm"])

    if params["sort"]:
        data1, label1, sorted_indices1 = sorted_by_label(data1, label1)
        data2, label2, sorted_indices2 = sorted_by_label(data2, label2)

        # Create a mapping from original indices to sorted indices
        sorted_indices1_map = np.argsort(sorted_indices1)
        sorted_indices2_map = np.argsort(sorted_indices2)

        # Update the links array using the mapping
        updated_links = np.zeros_like(links)
        updated_links[:, 0] = sorted_indices1_map[links[:, 0]]
        updated_links[:, 1] = sorted_indices2_map[links[:, 1]]
        links = updated_links
    
    save_path = os.path.join(params["save_path"], 'mmdma')
    os.makedirs(os.path.join(save_path, 'metrics'), exist_ok=True)

    # create acc & FOSCTTM error files if not exists
    acc_log_url = os.path.join(save_path, 'metrics', 'accLT_log.txt')
    foscttm_log_url = os.path.join(save_path, 'metrics', 'FOSCTTM_log.txt')

    if not os.path.exists(acc_log_url):
        with open(acc_log_url, 'w') as f: f.write('Param\tltACC\n')

    if not os.path.exists(foscttm_log_url):
        with open(foscttm_log_url, 'w') as f: f.write(f'Param\tAVG_FOSCTTM\n') 

    # test parameters
    lambda_range = params["mmdma"]["lambda_range"]
    seed_range = range(0, 20)
    run_mmdma(data1, data2, label1, label2, links, params["mmdma"], lambda_range, seed_range, 
              acc_log_url, foscttm_log_url, save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset name')
    main(parser.parse_args())
