"""
Train a diffusion model for recommendation with optional conditional similarity
"""

import argparse
from ast import parse
import os
import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models.gaussian_diffusion as gd
from models.DNN import DNN
import evaluate_utils
import data_utils
from copy import deepcopy

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

# yelp_clean_lr1e-05_wd0.0_bs400_dims[1000]_emb10_x0_steps5_scale0.01_min0.001_max0.01_sample0_reweight0_log.pth
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='yelp_clean', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='./datasets/', help='load data path')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

# params for the model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=5, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.01, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.01, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

# New params for conditional diffusion
parser.add_argument('--conditional', action='store_true', help='use conditional diffusion with similarity vectors')
parser.add_argument('--similarity_method', type=str, default='cosine', help='method to compute similarity: cosine or jaccard')
parser.add_argument('--top_k_similar', type=int, default=3, help='number of similar users to consider for conditioning')
parser.add_argument('--sim_normalization', type=str, default='none', help='normalization method for similarity vectors: none, min_max, l2')

args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
train_path = args.data_path + args.dataset + '/train_list.npy'
valid_path = args.data_path + args.dataset + '/valid_list.npy'
test_path = args.data_path + args.dataset + '/test_list.npy'

# Basic data loading
train_data, valid_y_data, test_y_data, n_user, n_item, _ = data_utils.data_load(train_path, valid_path, test_path)

# For conditional model, compute similarity matrix
similarity_matrix = None
if args.conditional:
    print(f"Computing {args.similarity_method} similarity matrix for {n_user} users...")
    # This function is now available in data_utils.py
    similarity_matrix = data_utils.compute_similarity_matrix(train_data, method=args.similarity_method)
    
    # Create dataset with similarity information
    train_dataset = data_utils.DataDiffusion(
        torch.FloatTensor(train_data.A), 
        similarity_matrix=similarity_matrix,
        normalization_method=args.sim_normalization
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        pin_memory=True, 
        shuffle=True, 
        num_workers=0, 
        worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.tst_w_val:
        tv_dataset = data_utils.DataDiffusion(
            torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A),
            similarity_matrix=similarity_matrix,
            normalization_method=args.sim_normalization
        )
        test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
else:
    # Original non-conditional data loading
    train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.tst_w_val:
        tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A))
        test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)

mask_tv = train_data + valid_y_data

print('data ready.')


### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

### Build MLP ###
out_dims = eval(args.dims) + [n_item]
in_dims = out_dims[::-1]

# For conditional model, ensure dimensions are correct
if args.conditional:
    model = DNN(in_dims, out_dims, args.emb_size, time_type=args.time_type, 
                norm=args.norm, conditional=args.conditional, n_items=n_item)
    print(f"Model input dimension: {in_dims[0]}, similarity vector dim: {n_item}")
    print("Using Diffusion model with conditional similarity vectors")
else:
    model = DNN(in_dims, out_dims, args.emb_size, time_type=args.time_type, 
                norm=args.norm)
    print("Using standard Diffusion model")

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("models ready.")

param_num = 0
mlp_num = sum([param.nelement() for param in model.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = mlp_num + diff_num
print("Number of all parameters:", param_num)

def evaluate(data_loader, data_te, mask_his, topN):
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            # Handle conditional vs non-conditional data format
            if args.conditional:
                batch, similarity_vec = batch_data
                batch = batch.to(device)
                similarity_vec = similarity_vec.to(device)
                his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
                
                # Call p_sample with conditional vector
                prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise, cond_vec=similarity_vec)
            else:
                batch = batch_data.to(device)
                his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
                prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise)
                
            prediction[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results

best_recall, best_epoch = -100, 0
best_test_result = None
print("Start training...")
for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= 20:
        print('-'*18)
        print('Exiting from training early')
        break

    model.train()
    start_time = time.time()

    batch_count = 0
    total_loss = 0.0
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Handle conditional vs non-conditional data format
        if args.conditional:
            batch, similarity_vec = batch_data
            batch = batch.to(device)
            similarity_vec = similarity_vec.to(device)
            
            batch_count += 1
            optimizer.zero_grad()
            
            # Call training_losses with conditional vector
            losses = diffusion.training_losses(model, batch, args.reweight, cond_vec=similarity_vec)
            loss = losses["loss"].mean()
            total_loss += loss
            loss.backward()
            optimizer.step()
        else:
            batch = batch_data.to(device)
            batch_count += 1
            optimizer.zero_grad()
            
            losses = diffusion.training_losses(model, batch, args.reweight)
            loss = losses["loss"].mean()
            total_loss += loss
            loss.backward()
            optimizer.step()
    
    if epoch % 5 == 0:
        valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN))
        if args.tst_w_val:
            test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
        else:
            test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
        evaluate_utils.print_results(None, valid_results, test_results)

        if valid_results[1][1] > best_recall: # recall@20 as selection
            best_recall, best_epoch = valid_results[1][1], epoch
            best_results = valid_results
            best_test_results = test_results

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            
            # Updated model filename to include conditional flag
            model_filename = '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_cond{}_{}.pth' \
                .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
                args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.conditional, args.log_name)
            
            torch.save(model, model_filename)
    
    print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
evaluate_utils.print_results(None, best_results, best_test_results)   
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))