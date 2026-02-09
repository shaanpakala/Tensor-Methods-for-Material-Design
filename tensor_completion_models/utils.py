# work_dir = "/home/spaka002/NSF_REU_2024/"

import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
import sys

from tensor_completion_models.ETC import *
from tensor_completion_models.CoSTCo import *
from tensor_completion_models.CPD import *
from tensor_completion_models.CPD_smooth import *
from tensor_completion_models.tuckER import *
from tensor_completion_models.Tensor_Train import *
from tensor_completion_models.NeAT.train import *

# from notebooks.tensor_completion_models.ETC import *

from sklearn.utils.validation import check_random_state
import tensorly as tl

from sklearn.model_selection import train_test_split, KFold


import re
def split_formula(formula):
    elements = re.findall(r'([A-Z][a-z]?)(\d+)', formula)
    return [e[0] for e in elements], [int(e[1]) for e in elements]


def get_sparse_tensor(X, Y, pad_indices = False, round_n = None, features = None, return_conversion = False):

    tX = pd.DataFrame(X, 
                columns = [f'Mode_{i+1}' for i in range(X.shape[1])],
                index = list(range(X.shape[0])))

    if return_conversion: convert_dict = dict()
    for c in range(tX.shape[1]):

        if 'num' in features[c]: start = 1
        else: start = 0

        if pad_indices: 

            if round_n is None: round_num = 1e-10
            else: round_num = round_n

            integers = np.arange(start, X[:, c].max()+1)
            fractions = np.round(np.round(X[:, c][X[:, c] % 1 != 0] / round_num) * round_num, 5)
            convert = sorted(set(np.append(integers, fractions)))

            tX[f'Mode_{c+1}'] = np.round(np.round(tX[f'Mode_{c+1}'] / round_num) * round_num, 5)
            del integers, fractions, round_num

        else:
            convert = sorted(set(X[:, c]))

        convert = {convert[i]:i for i in range(len(convert))}
        if return_conversion: convert_dict[f'Mode_{c+1}'] = convert

        tX[f'Mode_{c+1}'] = tX[f'Mode_{c+1}'].map(lambda x: convert[x])
        del convert

    tX = torch.tensor(np.array(tX)).long()

    if len(Y.shape) == 1: tY = torch.tensor(Y)
    else:
        new_indices = list()
        values = list()

        for xi in range(len(tX)):
            for yi in range(Y.shape[-1]):
                new_indices += [torch.cat((tX[xi], torch.tensor([yi])))]
                values += [Y[xi, yi]]

        tX, tY = torch.stack(new_indices), torch.tensor(values)
        
    tensor_shape = torch.Size(tX.max(axis = 0).values + 1)
    sparse_tensor = torch.sparse_coo_tensor(indices = tX.t(), values = tY, size = tensor_shape).coalesce()

    if return_conversion: return sparse_tensor, convert_dict
    else: return sparse_tensor
    
def convert_df_to_indices(df, conversions):
    
    return_df = df.copy()
    cols = list(return_df.columns)
    for i in range(len(cols)): return_df[cols[i]] = return_df[cols[i]].map(lambda x: conversions[f'Mode_{i+1}'][x])
    
    return torch.tensor(np.array(return_df))

def trunc_df(df, num_elements, max_num, min_freq):    
    
    return_df = df.copy()
    
    for i in range(num_elements): return_df = return_df[return_df[f'e{i+1}_num'].map(lambda x: x <= max_num)]
        
    for i in range(num_elements):
        comp = return_df[f'e{i+1}'].value_counts() > min_freq
        return_df = return_df[return_df[f'e{i+1}'].map(lambda x: comp[x])]
        del comp
        
    return return_df

def train_tc(model_type = 'CPD',
             rank = 5,
             train_loader = None,
             val_loader = None,
             tensor_size = None,
             num_epochs = 15_000, 
             batch_size = 256, 
             lr = 5e-3, 
             wd = 5e-4, 
             loss_p = 2,
             zero_lambda = 1,
             cpd_smooth_lambda = 2,
             cpd_smooth_window = 3,
             cpd_inverse_smooth_window = 3,
             cpd_inverse_smooth_lambda = 0,
             cpd_inverse_std_lambda = 0,
             non_smooth_modes = list(),
             non_inverse_smooth_modes = list(),
             NeAT_hidden_dim = 32,
             NeAT_drop = 0.1,
             NeAT_drop2 = 0.5,
             tucker_in_drop = 0.1,
             tucker_hidden_drop = 0.1,
             early_stopping = True,
             flags = 15,
             verbose = False, 
             epoch_display_rate = 1, 
             val_size = 0.2,
             reinitialize_count = 0,
             convert_to_cpd = False,
             device = "cuda" if torch.cuda.is_available() else "cpu"):

# _________________________ Configurations _____________________________________________________________________________________

    cfg = DotMap()
    
    # All
    cfg.nc = len(tensor_size)
    cfg.rank = rank
    cfg.sizes = tensor_size
    cfg.lr = lr
    cfg.wd = wd
    cfg.epochs = num_epochs
    cfg.random = 18
    cfg.device = device
    
    # CPD-S
    cfg.smooth_lambda = cpd_smooth_lambda
    cfg.window = cpd_smooth_window
    cfg.inverse_window = cpd_inverse_smooth_window
    cfg.inverse_smooth_lambda = cpd_inverse_smooth_lambda
    cfg.inverse_std_lambda = cpd_inverse_std_lambda
    
    
    # TuckER
    cfg.in_drop = tucker_in_drop
    cfg.hidden_drop = tucker_hidden_drop
    cfg.bs = batch_size
    bce_loss = nn.BCELoss()

    # NeAT
    cfg.layer_dims = [len(cfg.sizes), NeAT_hidden_dim, 1]
    cfg.depth = len(cfg.layer_dims)
    cfg.dropout = NeAT_drop
    cfg.dropout2 = NeAT_drop2
    cfg.epochs = num_epochs
    cfg.val_size = val_size
    
# ______________________________________________________________________________________________________________________________
    
    # pick Tensor Completion Model & Initialize
    model_dict = {
        'CPD':CPD,
        'CPD-S':CPD_Smooth,
        'NeAT':NeAT,
        'TuckER':TuckER
    }
    
    model = model_dict[model_type](cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    model._initialize()

    flag, flag_2 = 0, 0
    old_MAE = 1e+6
    err_list = list()
    
# ____________________ train the model _________________________________________________________________________________________
    
    for epoch in range(cfg.epochs):

        model.train()

        for batch in train_loader:

            inputs, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            model = model.to(device)

            rec = model(inputs)

# _________________________ loss_fn ____________________________________________________________________________________________

            if model_type == 'tuckER':
                loss = bce_loss(rec, targets.to(torch.float))

            else:

                errors = abs(rec - targets).pow(loss_p)

                zero_lambda_mask = ((targets == 0) * zero_lambda) + (targets != 0)
                errors = errors * zero_lambda_mask

                loss = errors.sum()

                if (model_type == 'cpd.smooth' or model_type == 'cpd.smooth.t'):

                    for n in range(len(tensor_size)):    
                        if n not in non_smooth_modes:                
                            loss = loss + (model.smooth_reg(n) * cfg.smooth_lambda)
                            
                        if n not in non_inverse_smooth_modes:
                            loss = loss + (model.inverse_smooth_reg(n) * cfg.inverse_smooth_lambda)
                            loss = loss + (model.inverse_std_error(n) * cfg.inverse_std_lambda)


            loss.backward()
            optimizer.step()

# ______________________________________________________________________________________________________________________________

        model.eval()
        if (epoch+1) % 1 == 0:
            with torch.no_grad():
                
                val_MAE, num_batches = 0, 0
                for batch in val_loader:
                    
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                    
                    val_rec = model(inputs)
                    val_MAE += abs(val_rec - targets).mean()
                    num_batches += 1

                val_MAE /= num_batches

                # for early stopping
                if (early_stopping):
                    
                    if (old_MAE < val_MAE):
                        flag +=1
                            
                    if flag == flags:
                        break
                    
                    if (old_MAE == val_MAE):
                        flag_2 +=1
                        
                if flag_2 == 25:
                    break
                
                old_MAE = val_MAE

                err_list += [old_MAE]
                
                if (verbose and ((epoch+1)%epoch_display_rate==0)): 
                    print(f"Epoch {epoch+1} Train Loss: {loss} Val_MAE: {val_MAE:.4f}\t")
                
    if (verbose): print()                       

# ______________________________________________________________________________________________________________________________
    
    # reinitialize model if it didn't converge!
    if (torch.tensor(err_list[10:]).std() < 1e-6): 
        new_model_type = model_type
        
        if (reinitialize_count >= 5) and convert_to_cpd:
            if (verbose): print(f"\nConverting {model_type} to cpd!\n")
            new_model_type = 'CPD'


        if (verbose): print(f"\nReinitializing {model_type}! Reinitialize Count: {reinitialize_count}.\n")     

        return train_tc(model_type = new_model_type,
                        rank = rank,
                        train_loader = train_loader,
                        val_loader = val_loader,
                        tensor_size = tensor_size,
                        num_epochs = num_epochs, 
                        batch_size = batch_size, 
                        lr = lr, 
                        wd = wd, 
                        loss_p = loss_p,
                        zero_lambda = zero_lambda,
                        cpd_smooth_lambda = cpd_smooth_lambda,
                        cpd_smooth_window = cpd_smooth_window,
                        cpd_inverse_smooth_window = cpd_inverse_smooth_window,
                        cpd_inverse_smooth_lambda = cpd_inverse_smooth_lambda,
                        cpd_inverse_std_lambda = cpd_inverse_std_lambda,
                        non_smooth_modes = non_smooth_modes,
                        non_inverse_smooth_modes = non_inverse_smooth_modes,
                        NeAT_hidden_dim = NeAT_hidden_dim,
                        NeAT_drop = NeAT_drop,
                        NeAT_drop2 = NeAT_drop2,
                        tucker_in_drop = tucker_in_drop,
                        tucker_hidden_drop = tucker_hidden_drop,
                        early_stopping = early_stopping,
                        flags = flags,
                        verbose = verbose, 
                        epoch_display_rate = epoch_display_rate, 
                        val_size = val_size,
                        reinitialize_count = reinitialize_count,
                        convert_to_cpd = convert_to_cpd,
                        device = device)

# ______________________________________________________________________________________________________________________________

    return model    # return final tensor completion model







# ______________________________________________________________________________________________________________________________
# ______________________________________________________________________________________________________________________________
# ______________________________________________________________________________________________________________________________
# ______________________________________________________________________________________________________________________________
# ______________________________________________________________________________________________________________________________



def train_tensor_completion(model_type, 
                            sparse_tensor,
                            rank = 5, 
                            num_epochs = 15_000, 
                            batch_size = 256, 
                            lr = 5e-3, 
                            wd = 5e-4, 
                            loss_p = 2,
                            zero_lambda = 1,
                            cpd_smooth_lambda = 2,
                            cpd_smooth_window = 3,
                            cpd_inverse_smooth_window = 3,
                            cpd_inverse_smooth_lambda = 0,
                            cpd_inverse_std_lambda = 0,
                            non_smooth_modes = list(),
                            non_inverse_smooth_modes = list(),
                            dataset_mode = None,
                            NeAT_hidden_dim = 32,
                            NeAT_drop = 0.1,
                            NeAT_drop2 = 0.5,
                            tucker_in_drop = 0.1,
                            tucker_hidden_drop = 0.1,
                            train_norm = None,
                            early_stopping = True,
                            flags = 15,
                            verbose = False, 
                            epoch_display_rate = 1, 
                            val_size = 0.2,
                            return_errors = False,
                            reinitialize_count = 0,
                            convert_to_cpd = False,
                            for_queries = False,
                            device = "cuda" if torch.cuda.is_available() else "cpu"):


    if (verbose): print(f"Rank = {rank}; lr = {lr}; wd = {wd}\n")

    train_indices = sparse_tensor.indices().t()
    train_values = sparse_tensor.values()
    tensor_size = sparse_tensor.size()

    training_indices = train_indices.to(device) # NNZ x mode
    training_values = train_values.to(device)   # NNZ
    training_values = training_values.to(torch.double)
    
    if val_size is not None:
        indices, val_indices, values, val_values = train_test_split(training_indices, training_values, 
                                                                    test_size = val_size, random_state = 18)
    else: indices, values = training_indices, training_values

    cfg = DotMap()

    cfg.norm = lambda x: x
    cfg.unnorm = lambda x: x

    if train_norm is not None:

        if train_norm == 'minmax':

            max_ = training_values.max()
            min_ = training_values.min()

            cfg.norm = lambda x: (x - min_)/(max_ - min_)
            cfg.unnorm = lambda x: (x*(max_ - min_)) + min_

        elif train_norm == 'standard':

            mean_ = training_values.mean()
            std_ = training_values.std()

            cfg.norm = lambda x: (x - mean_)/(std_)
            cfg.unnorm = lambda x: (x*std_) + mean_


    values = cfg.norm(values)
    if val_size is not None: val_values = cfg.norm(val_values)
    
    dataset = COODataset(indices, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    cfg.nc = training_indices.shape[0]
    cfg.rank = rank
    cfg.sizes = tensor_size
    cfg.lr = lr
    cfg.wd = wd
    cfg.epochs = num_epochs
    cfg.random = 18

    # create the model
    
    if (model_type == 'cpd'):
        model = CPD(cfg).to(device)
        
    elif (model_type == 'cpd.smooth'):
        
        cfg.smooth_lambda = cpd_smooth_lambda
        cfg.window = cpd_smooth_window
        cfg.inverse_window = cpd_inverse_smooth_window
        cfg.inverse_smooth_lambda = cpd_inverse_smooth_lambda
        cfg.inverse_std_lambda = cpd_inverse_std_lambda
        
        model = CPD_Smooth(cfg).to(device)
            
        for param in model.parameters():
            param.requires_grad = True

    elif (model_type == 'costco'):
        model = CoSTCo(cfg).to(device)

    elif (model_type == 'tuckER'):

        cfg.in_drop = tucker_in_drop
        cfg.hidden_drop = tucker_hidden_drop
        cfg.bs = batch_size
        cfg.device = device

        model = TuckER(cfg).to(device)
        
    
    elif (model_type == 'NeAT'):
        
        cfg.rank = rank
        cfg.sizes = tensor_size
        cfg.layer_dims = [len(cfg.sizes), NeAT_hidden_dim, 1]
        cfg.depth = len(cfg.layer_dims)
        cfg.lr = lr
        cfg.wd = wd
        cfg.dropout = NeAT_drop
        cfg.dropout2 = NeAT_drop2
        cfg.epochs = num_epochs
        cfg.batch_size = batch_size
        cfg.device = device
        cfg.val_size = val_size

        model = NeAT(cfg).to(device)

    
    elif (model_type == "tensor.train"):
        
        model = train_tensor_train(sparse_tensor = sparse_tensor,
                                   rank = rank,
                                   lr = lr,
                                   wd = wd,
                                   num_epochs = num_epochs,
                                   batch_size = batch_size,
                                   early_stopping = early_stopping,
                                   flags = flags,
                                   val_size = val_size,
                                   verbose = verbose,
                                   epoch_display_rate = epoch_display_rate,
                                   device = device)
        
        return model

    else:
        print("No Model Selected!")
        model = CPD(cfg).to(device)

    bce_loss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    model._initialize()

    flag = 0
    flag_2 = 0

    err_list = list()
    old_MAE = 1e+6

    # train the model
    for epoch in range(cfg.epochs):

        model.train()

        for batch in dataloader:

            inputs, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            model = model.to(device)

            rec = model(inputs)

# _________________________ loss_fn ____________________________________________________________________________________________

            if model_type == 'tuckER':

                loss = bce_loss(rec, targets.to(torch.float))

            else:

                errors = abs(rec - targets).pow(loss_p)
                
                zero_lambda_mask = ((targets == 0) * zero_lambda) + (targets != 0)
                errors = errors * zero_lambda_mask
                
                loss = errors.sum()
                
                if (model_type == 'cpd.smooth' or model_type == 'cpd.smooth.t'):
                    
                    for n in range(len(tensor_size)):    
                        if n not in non_smooth_modes:  
                            smooth_loss = 0
                            
                            smooth_loss = (model.smooth_reg(n) * cfg.smooth_lambda)
                            if dataset_mode is not None:
                                if n == dataset_mode[0]: smooth_loss*=dataset_mode[1]
                            loss = loss + smooth_loss
                            
                        # if n not in non_inverse_smooth_modes:
                        #     loss = loss + (model.inverse_smooth_reg(n) * cfg.inverse_smooth_lambda)
                        #     loss = loss + (model.inverse_std_error(n) * cfg.inverse_std_lambda)

# ______________________________________________________________________________________________________________________________

            loss.backward()
            optimizer.step()

        model.eval()
        if (epoch+1) % 1 == 0 and val_size is not None:
            with torch.no_grad():

                train_rec = model(indices)
                train_MAE = abs(train_rec - values).mean()

                val_rec = model(val_indices)
                val_MAE = abs(val_rec - val_values).mean()

                # for early stopping
                if (early_stopping):

                    if (old_MAE < val_MAE):
                        flag +=1

                    if flag == flags:
                        break

                    if (old_MAE == val_MAE):
                        flag_2 +=1

                if flag_2 == 25:
                    break

                old_MAE = val_MAE

                err_list += [old_MAE]

                if (verbose and ((epoch+1)%epoch_display_rate==0)): 
                    print(f"Epoch {epoch+1} Train_MAE: {train_MAE:.4f} Val_MAE: {val_MAE:.4f}\t")

    if (verbose): print()                       
    
    
    # reinitialize model if it didn't converge!
    if val_size is not None: cond = (torch.tensor(err_list[10:]).std() < 1e-6)
    else: cond = False

    if cond: 
        
        if (reinitialize_count >= 5) and convert_to_cpd:
            
            if (verbose): print(f"\nConverting {model_type} to cpd!\n")

            return train_tensor_completion(model_type = 'cpd', 
                                            sparse_tensor = sparse_tensor,
                                            rank = rank, 
                                            num_epochs = num_epochs, 
                                            batch_size = batch_size, 
                                            lr=lr, 
                                            wd=wd, 
                                            tucker_in_drop = tucker_in_drop,
                                            tucker_hidden_drop = tucker_hidden_drop,
                                            early_stopping = early_stopping, 
                                            flags = flags, 
                                            verbose = verbose, 
                                            epoch_display_rate = epoch_display_rate, 
                                            val_size = val_size,
                                            return_errors = return_errors,
                                            reinitialize_count=reinitialize_count+1)
        
        if (verbose): print(f"\nReinitializing {model_type}! Reinitialize Count: {reinitialize_count}.\n")     

        return train_tensor_completion(model_type = model_type, 
                            sparse_tensor = sparse_tensor,
                            rank = rank, 
                            num_epochs = num_epochs, 
                            batch_size = batch_size, 
                            lr=lr, 
                            wd=wd, 
                            tucker_in_drop = tucker_in_drop,
                            tucker_hidden_drop = tucker_hidden_drop,
                            early_stopping = early_stopping, 
                            flags = flags, 
                            verbose = verbose, 
                            epoch_display_rate = epoch_display_rate, 
                            val_size = val_size,
                            return_errors = return_errors,
                            reinitialize_count=reinitialize_count+1)
                            
    if (return_errors): return model, torch.tensor(err_list)
    return model