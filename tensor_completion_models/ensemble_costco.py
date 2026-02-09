import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

device = 'cpu'

class ensemble_costco(nn.Module):

    def __init__(self, rank, tensor_shape, activation = None, n_decompositions = 3, 
                 hidden_channels = None, dropout = [0, 0],
                 smooth_lambda = 0.5, dynamic_smooth_lambda = None, window = 3, non_smooth_modes = list()):
        super(ensemble_costco, self).__init__()

        self.current_training_iteration = 0
        self.current_training_epoch = 0
        self.losses = {'train':{'Regression':list(), 'Smooth':list(), 'Total':list()},
                       'val':  {'Regression':list(), 'Smooth':list(), 'Total':list()}}        

        self.rank = rank
        self.sizes = tensor_shape
        self.nmode = len(self.sizes)
        if type(dropout) == int or type(dropout) == float: self.dropout = [dropout] * 2
        else: self.dropout = dropout

        self.n_decompositions = n_decompositions
        if hidden_channels is None: self.hidden_channels = self.n_decompositions
        else: self.hidden_channels = hidden_channels
        
        self.fl = 0

        self.loss_fn = nn.L1Loss()
        self.window = window
        self.non_smooth_modes = non_smooth_modes
        self.smooth_lambda = smooth_lambda
        self.dynamic_smooth_lambda = dynamic_smooth_lambda

        if activation is None: self.activation = nn.Identity()
        elif activation.lower() == 'sigmoid': self.activation = nn.Sigmoid()   
        elif activation.lower() == 'relu': self.activation = nn.ReLU()
        elif activation.lower() == 'tanh': self.activation = nn.Tanh()
        elif activation.lower() == 'elu': self.activation = nn.ELU()
        else: activation = nn.Identity()
        
        self.embeds_list = nn.ModuleList([
            nn.ModuleList([nn.Embedding(self.sizes[i], self.rank) 
                        for i in range(self.nmode)])
            for _ in range(self.n_decompositions)
        ])

        hc = lambda x: int(self.hidden_channels * x)
        conv_block = [
            nn.Conv2d(in_channels = self.n_decompositions, out_channels = hc(1), kernel_size = (3, 3), stride = 1, padding = 1),
            nn.Dropout(self.dropout[0]),
            self.activation,
            nn.Conv2d(in_channels = hc(1), out_channels = 1, kernel_size = (3, 3), stride = 1, padding = 1),
        ]
        self.conv_block = nn.Sequential(*conv_block)

        linear_block = [
            nn.Linear(int(self.rank * self.nmode), int(self.rank * self.nmode) // 2),
            nn.Dropout(self.dropout[1]),
            self.activation,
            nn.Linear(int(self.rank * self.nmode) // 2, 1)
        ]
        self.linear_block = nn.Sequential(*linear_block)

    def recon(self, idxs, return_facs = False):

        all_facs = list()
        for embeds in self.embeds_list:
            facs = [embeds[m](idxs[:, m]).unsqueeze(-1) for m in range(self.nmode)]
            factor_list = torch.concat(facs, dim=-1)
            all_facs.append(factor_list)

        all_facs = torch.stack(all_facs, axis = 1)        
        rec = self.conv_block(all_facs)

        rec = rec.reshape(-1, int(self.rank * self.nmode))
        rec = self.linear_block(rec).squeeze()

        # rec = rec.reshape(-1, self.rank, self.nmode)
        if return_facs: return rec, all_facs
        return rec

    def forward(self, idxs, return_facs = False): 
        return self.recon(idxs, return_facs) 

    def get_smooth_loss(self): 
        return 0

    def train_model(self, 
                    X_train,
                    Y_train,
                    validation_portion = 0.1,
                    early_stopping = 0,
                    n_epochs = 10, 
                    lr = 1e-3, 
                    wd = 1e-4, 
                    batch_size = 32,
                    print_rate = 1, 
                    smooth_lambda = None,
                    verbose = False,
                    random_state = 18):

        flag = 0
        ff = True
        f = lambda x, y = 5: " "*(y-len(str(x))) + str(x)
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr = lr, weight_decay = wd)

        self.smooth_lambda = smooth_lambda if smooth_lambda is not None else 0

        if validation_portion > 0:
            x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, 
                                                        test_size = validation_portion, shuffle = True, 
                                                        random_state = random_state)
        else: x_train, y_train = X_train, Y_train

        trainset = TensorDataset(x_train, y_train)
        trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = False)

        rl = 1 - self.smooth_lambda  # regression lambda
        sl = self.smooth_lambda      # smooth lambda
        for epoch in range(1, n_epochs+1):

            self.train()
            for batch in trainloader:
                x, y = batch
                reconstructed, factors = self(x, return_facs = True)
                factors_loss = self.fl * (1/factors.std(axis = 1).mean(axis = (1, 2)).mean())
                reg_loss = rl * self.loss_fn(reconstructed.squeeze(), y.squeeze())
                smooth_loss = sl * self.get_smooth_loss()
                smooth_loss = 0
                loss = reg_loss + smooth_loss + factors_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.losses['train']['Regression'].append(float(reg_loss))
                self.losses['train']['Smooth'].append(float(smooth_loss))
                self.losses['train']['Total'].append(float(loss))

                self.current_training_iteration += 1

            self.eval()
            if validation_portion > 0:
                with torch.no_grad():
                    reconstructed = self(x_val)
                    vreg_loss = rl * self.loss_fn(reconstructed.squeeze(), y_val.squeeze())
                    vsmooth_loss = sl * self.get_smooth_loss()
                    vloss = vreg_loss + vsmooth_loss

                    self.losses['val']['Regression'].append(float(vreg_loss))
                    self.losses['val']['Smooth'].append(float(vsmooth_loss))
                    self.losses['val']['Total'].append(float(vloss))
                    
                if len(self.losses['val']['Total']) > 1 and early_stopping:
                    if self.losses['val']['Total'][-1] > self.losses['val']['Total'][-2]:
                        flag +=1
                        
                    if flag > early_stopping: 
                        if verbose: print(f"\nReached convergence criteria. Stopping training early!\n")
                        break

            self.current_training_epoch += 1

            # ___________ Display progress _______________________________________________________________
            if verbose and (epoch % print_rate == 0 or epoch == n_epochs): 
                if ff:
                    s = f"| Epoch {f(epoch)} | Train Loss = {float(loss):.4f} ({float(reg_loss):.4f} + {float(smooth_loss):.4f}) |"
                    print("_"*len(s))
                    ff = False

                print(f"| Epoch {f(epoch)} | Train Loss = {float(loss):.4f} ({float(reg_loss):.4f} + {float(smooth_loss):.4f}) |")
                s = f" Epoch {f(epoch)} "
                if validation_portion > 0:
                    print(f"|{' '*len(s)}| Valid Loss = {float(vloss):.4f} ({float(vreg_loss):.4f} + {float(vsmooth_loss):.4f}) |")

        s = f"| Epoch {f(epoch)} | Train Loss = {float(loss):.4f} ({float(reg_loss):.4f} + {float(smooth_loss):.4f}) |"
        if verbose: print("_"*len(s))
        del s
        
        self.validation_portion = validation_portion
        

    def plot_losses(self, loss_types = {'train':['Total'], 'val':['Total']}, log_scale = None):
        plt.figure(figsize = (12, 6))

        colors = {
            'train':['black', 'gray', 'brown'],
            'val':['blue', 'red', 'green']
        }

        if 'train' in loss_types:
            for i in range(len(loss_types['train'])):
                plot_loss = loss_types['train'][i]
                x = np.arange(self.current_training_iteration) * (self.current_training_epoch/self.current_training_iteration)
                plt.plot(x, self.losses['train'][plot_loss], 
                         lw = 3, color = colors['train'][i], label = f'Train {plot_loss}')

        if 'val' in loss_types and self.validation_portion > 0:
            for i in range(len(loss_types['val'])):
                plot_loss = loss_types['val'][i]
                x = np.arange(self.current_training_epoch)
                plt.plot(x, self.losses['val'][plot_loss], 
                         lw = 3, color = colors['val'][i], label = f'Valid {plot_loss}')        

        plt.title('Loss Curve', fontsize = 20)
        plt.xlabel('Training Epoch', fontsize = 16)
        plt.ylabel('Loss', fontsize = 16)
        plt.legend(fontsize = 16)
        if log_scale is not None:
            if 'x' in log_scale: plt.xscale('log')
            if 'y' in log_scale : plt.yscale('log')
        plt.show()