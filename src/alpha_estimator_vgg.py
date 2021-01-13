from models import VGG, LeNetFamily
from utils import alpha_estimator
from tqdm import tqdm

import glob
import os
import torch

import numpy as np
import click
import pickle

# Some divisibility BS
resolutions = { 3136: [756, [2,3,6,18,36]], 
                1574: [756, [2,3,6,18,36]], 
                1563: [756, [2,3,6,18,36]], 
                793: [756, [2,3,6,7,9,18,36]],
                782: [756, [2,3,6,7,9,18,36]], 
                402: [324, [2,3,6,9,18,36]],
                391: [324, [2,3,6,9,18,36]],
                207: [196, [2, 4, 7, 14, 28]],
                196: [196, [2, 4, 7, 14, 28]]
              }

def get_5_good_fiv(x):
    rs = int(np.floor(np.sqrt(x)))
    div = []
    for i in range(2,rs):
        if x%i == 0:
            div.append(i)
    if len(div)>5:        
        return x, div[::int(np.ceil(len(div)/4))]+[div[-1]]
    else:
        return get_5_good_fiv(x-1)

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
    #model_sizes.append(full_size)
    return model_sizes

def get_ms(iter_size, mod_size):
    if iter_size in resolutions:
        iter_set =  resolutions[iter_size]
    else:
        raise ValueError("Not a valid iteration size")
    mod_set = get_5_good_fiv(mod_size)
    full_size = iter_set[0]*mod_set[0]
    all_ms = sorted([a*b for a in iter_set[1] for b in mod_set[1]])
    return full_size, all_ms

def estimator_vector_full(iterate_matrix):
    iter_nums = iterate_matrix.shape[0]
    dim = iterate_matrix.shape[1]
    sz,ms = resolutions[iter_nums]
    
    iterate_matrix_zm = iterate_matrix - torch.mean(iterate_matrix, axis=0).view(1,-1)
   
    print(sz,ms, iter_nums, dim)
    print(iterate_matrix_zm[-1*sz:,:].shape)
    print(len( iterate_matrix_zm[-1*sz:,:] ))
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
    sz, ms = get_5_good_fiv(dim)
    
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

def estimate_experiment(exp_name, mod_name):
    device = torch.device("cuda:0")
    
    models = glob.glob('{}/*'.format(exp_name))
    models = sorted(models, key=lambda x:int(x.split('_')[-1]))
    iter_num = len(models)
    
    # Peek for the model size to be used later
    deep_model = VGG(mod_name, True)
    deep_model = deep_model.to(device)
    deep_model.load_state_dict(torch.load(models[0]))
    ms = peek_model_size(deep_model)

    parameter_arrays = []
    for mod_size in ms:
        print("ozan", iter_num, mod_size)
        parameter_arrays.append(torch.zeros(iter_num, mod_size))
    
    for i, model in enumerate(models):
        deep_model = VGG(mod_name, True)
        deep_model = deep_model.to(device)
        deep_model.load_state_dict(torch.load(model))
        
        read_mem_cnt = 0
        read_param_cnt = 0
        for p in deep_model.parameters():
            cpu_data = p.data.cpu().view(1,-1)
            if len(p.shape) < 2:
                # Not the weight mat, so append to full one and go on
                #parameter_arrays[-1][i,read_param_cnt:read_param_cnt+p.shape[0]] = cpu_data
                #read_param_cnt+=p.shape[0]
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
@click.option('--model_folder')
def estimate(model_folder):
    exps = glob.glob('{}/*/*/*'.format(model_folder))
    exps = sorted(exps)

    alpha = {}
    for current_exp in exps:    
        mod_name = 'VGG'+current_exp.split('/')[1][1:]
        alpha[current_exp] = estimate_experiment(current_exp, mod_name)
    
    with open("vgg16_alpha_{}.bn".format(exp_id), "wb") as f:
        pickle.dump(alpha, f)

if __name__ == '__main__':
    estimate()
