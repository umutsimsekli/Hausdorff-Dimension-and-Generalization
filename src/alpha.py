from models import LeNetFamily
from utils import alpha_estimator
from tqdm import tqdm

import glob
import os
import torch

import numpy as np
import click
import pickle

# Some divisibility BS
resolutions = { 1563: [36*36, [2,3,6,12,36,72]], 
                196: [196, [2, 4, 7, 14, 28]], 
                782: [756, [2,3,6,7,9,18,36]], 
                391: [324, [2,3,6,9,18,36]]}

net_sizes_div = {368640: [600*600,[4,10,60,120,300,600]],
                 10080: [100*100, [4,10,20,50,100]],
                 386914: [600*600, [4,10,60,120,200,600]],
                 450: [20*20 , [2,4,5,10,20]],
                 2400: [48*48, [2,4,12,24,48]],
                 4000: [60*60, [2,4,12,30,50]],
                 6882: [80*80, [2,4,20,40,80]],
                 48000: [200*200, [2,4,20,100,200]],
                 840: [28*28, [2,4,7,14,28]],
                 52202: [228*228, [2,4,57,114,228]],
                 97706: [312*312, [2,4,13,39,156,312]],
                 7056: [84*84, [2,7,21,42,84]],
                 1200: [34*34, [2,17,34]],
                 369970: [600*600, [4,10,60,120,300,600]],
                 415474: [644*644, [7, 23, 161, 322,644]],
                 69146: [260*260, [2,13,20,130,260]]}
                 
def peek_model_size(model):
    model_sizes = []
    # Input is mode
    full_size = 0
    for p in model.parameters():
        if(len(p.shape)<2):
            full_size += p.shape[0]
            continue
        model_sizes.append(np.prod(p.shape))
        full_size += np.prod(p.shape)
    model_sizes.append(full_size)
    return model_sizes

def get_ms(iter_size, mod_size):
    if iter_size in [1563, 196, 782, 391]:
        iter_set =  resolutions[iter_size]
    else:
        raise ValueError("Not a valid iteration size")
    if mod_size in net_sizes_div:
        mod_set = net_sizes_div[mod_size]
    else:
        raise ValueError("Not a valid model size")
    full_size = iter_set[0]*mod_set[0]
    all_ms = sorted([a*b for a in iter_set[1] for b in mod_set[1]])
    return full_size, all_ms

def estimator_vector_full(iterate_matrix):
    iter_nums = iterate_matrix.shape[0]
    dim = iterate_matrix.shape[1]
    sz,ms = resolutions[iter_nums]
    
    iterate_matrix_zm = iterate_matrix - torch.mean(iterate_matrix, axis=0).view(1,-1)
    
    est = [alpha_estimator(mm, iterate_matrix_zm[-1*sz:,:]).item() for mm in ms]

    return np.median(est)

def estimator_vector_projected(iterate_matrix):
    iter_nums = iterate_matrix.shape[0]
    dim = iterate_matrix.shape[1]
    sz,ms = resolutions[iter_nums]
    
    iterate_matrix_zm = iterate_matrix - torch.mean(iterate_matrix, axis=0).view(1,-1)    

    proj_alpha = []
    for i in range(10):
        rand_direction = np.random.randn(dim,1)
        rand_direction = rand_direction / np.linalg.norm(rand_direction)
        rand_direction_t = torch.from_numpy(rand_direction).float()
        
        projected = torch.mm(iterate_matrix_zm,rand_direction_t)
        
        cur_alpha_est = [alpha_estimator(mm, projected[-1*sz:,:]).item() for mm in ms]
        
        proj_alpha.append(np.median(cur_alpha_est))
    return np.median(proj_alpha), np.max(proj_alpha)

def estimator_vector_mean(iterate_matrix):
    iter_nums = iterate_matrix.shape[0]
    dim = iterate_matrix.shape[1]

    mean_over_iters = torch.mean(iterate_matrix, axis=0)
    mean_over_iters_zm = mean_over_iters - torch.mean(mean_over_iters)
    mean_over_iters_zm = mean_over_iters_zm.view(-1,1)
    sz, ms = net_sizes_div[dim]
    
    estimate_mean = [alpha_estimator(mm, mean_over_iters_zm[-1*sz:,:]).item() for mm in ms]
    return np.median(estimate_mean)

def estimator_scalar(iterate_matrix):
    iter_nums = iterate_matrix.shape[0]
    dim = iterate_matrix.shape[1]

    sz, ms = get_ms(iter_nums, dim)
    iterate_matrix_vec = iterate_matrix.view(-1,1)
    iterate_matrix_vec_zm = iterate_matrix_vec - torch.mean(iterate_matrix_vec)
    estimate = [alpha_estimator(mm, iterate_matrix_vec_zm[-1*sz:,:]) for mm in ms[::4]]
   
    return np.median(estimate)

def estimate_experiment(exp_name, ncl):
    device = torch.device("cuda:0")
    
    models = glob.glob('{}/*'.format(exp_name))
    models = sorted(models)
    iter_num = len(models)
    
    # Peek for the model size to be used later
    deep_model = LeNetFamily(ncl[0],ncl[1])
    deep_model = deep_model.to(device)
    deep_model.load_state_dict(torch.load(models[0]))
    ms = peek_model_size(deep_model)

    parameter_arrays = []
    for mod_size in ms:
        parameter_arrays.append(torch.zeros(iter_num, mod_size))
    
    for i, model in enumerate(models):
        deep_model = LeNetFamily(ncl[0],ncl[1])
        deep_model = deep_model.to(device)
        deep_model.load_state_dict(torch.load(model))
        
        read_mem_cnt = 0
        read_param_cnt = 0
        for p in deep_model.parameters():
            cpu_data = p.data.cpu().view(1,-1)
            if len(p.shape) < 2:
                # Not the weight mat, so append to full one and go on
                parameter_arrays[-1][i,read_param_cnt:read_param_cnt+p.shape[0]] = cpu_data
                read_param_cnt+=p.shape[0]
                continue
            parameter_arrays[read_mem_cnt][i,0:np.prod(p.shape)] = cpu_data
            read_mem_cnt +=1
            
    # All models are stored in the memory, so we need to call estimators
    alpha_full_est = []
    alpha_proj_med_est = []
    alpha_proj_max_est = []
    alpha_mean_est = []
    alpha_scalar_est = []
    for param in parameter_arrays:
        alpha_full = estimator_vector_full(param)
        alpha_full_est.append(alpha_full)

        alpha_proj_med, alpha_proj_max = estimator_vector_projected(param)
        alpha_proj_med_est.append(alpha_proj_med)
        alpha_proj_max_est.append(alpha_proj_max)

        alpha_mean = estimator_vector_mean(param)
        alpha_mean_est.append(alpha_mean)

        alpha_scalar = estimator_scalar(param)
        alpha_scalar_est.append(alpha_scalar) 
    return {"full": alpha_full_est, "proj_med": alpha_proj_med_est, "proj_max": alpha_proj_max_est,
            "mean": alpha_mean_est, "scalar": alpha_scalar_est}

@click.command()
@click.option('--exp_id')
def estimate(exp_id):
    exps = glob.glob('last_epoch_results/*/')
    exps = sorted(exps)
    current_exp = exps[int(exp_id)]
    
    ncl = current_exp.split('/')[1].split('__')[0:2]
    ncl = [int(ncl[0].split('_')[1]), int(ncl[1].split('_')[1])]

    alpha = estimate_experiment(current_exp, ncl)
    
    with open("alpha_{}.bn".format(exp_id), "wb") as f:
        pickle.dump({"alpha": alpha, "exp": current_exp}, f)

if __name__ == '__main__':
    estimate()
