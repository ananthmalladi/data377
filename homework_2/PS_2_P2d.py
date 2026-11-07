import os
# pip-installed torch and conda's MKL numpy each ship their own libiomp5md.dll;
# without this, the process aborts with "OMP: Error #15". Must be set before importing torch.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from category_encoders import TargetEncoder

# import sklearn

import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as scs



class MLP_emb(nn.Module):
    '''
    MLP with arbitrary number of hidden layers, and with an embedding layer
    Loss function -log(p_{r,p}(x)), where p_{r,p}(x) is the negative binom. dist.
    p_{r,p}(x) = (x + r - 1 choose x) (1-p)^r p^x = x successes before r failures
    The network will predict both p and r, and will convert p into the mean 
    prediction. 
    '''
    def __init__(self, num_inputs, h_arr, num_outputs, embed_dim, lr=1e-3, batch_size=128, device='cpu', seed=39):
        '''
        num_inputs: dimensionality of the input space
        h_arr: dimensionalities of hidden layers
        num_outputs: dimensionality of the output space, with some caveats:
            When trying to predict a single real number, this is 1. 
            When trying to predict {0,1}, this is also 1. And same for trying to
                predict a single integer. 
            When trying to predict a categorical value {1,...,N}, use N since 
                the PyTorch implementation of this is built for N (rather than
                N-1, which is mathematically what we'd do). 
        '''
        super(MLP_emb, self).__init__() # Initialize nn.module
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        torch.manual_seed(seed)
        self.W_arr = nn.ParameterList()
        self.b_arr = nn.ParameterList()
        h_lst = [embed_dim + 2] + h_arr + [num_outputs]
        # Do an embedding layer:
        self.emb = nn.Embedding(num_inputs, embed_dim)
        # And the rest is as usual
        for i in range(1, len(h_lst)):
            W_mat = torch.randn(h_lst[i], h_lst[i-1], requires_grad=True)
            self.W_arr.append(nn.Parameter(W_mat))
            self.b_arr.append(nn.Parameter(torch.randn(\
                h_lst[i], 1, requires_grad=True)))
        self.h_lst = h_lst
        # Normalize each layer
        self.ln_arr = nn.ModuleList([nn.LayerNorm(h) for h in h_arr])
        self.to(device) # move all parameters and layers to the device at once\

    def forward(self, X):
        '''
        This function turns inputs into outputs via the model we're working with
        '''
        # The d-dimensional embedding layer takes inputs in R^p and turns them 
        # into a matrix in R^(d,p).
        # i.e. (x1, ..., xp) has x1 --> v1, ..., xp --> vp for each vi in R^d. 
        # Then, each unique input (x1, ..., xp) gets mapped to a vector 
        # v = v1 + ... + vp in R^d. The idea is that two vectors that are close 
        # to each-other in R^d should mean something similar, and will be 
        # treated similarly by the network.  
        X_target = X[:,:2]
        X_dense = X[:,2:].long()
        dense_vec = self.emb(X_dense).sum(dim=1) # Embed, then compute v1 + ... + vp
        h = torch.cat([X_target, dense_vec], dim=1).T
        # And the rest goes through 
        for i in range(len(self.W_arr) - 1):
            h = nn.ReLU()((self.ln_arr[i]((self.W_arr[i] @ h + self.b_arr[i]).T)).T)
        out = self.W_arr[-1] @ h + self.b_arr[-1]
        # At this stage, there are two outputs: f(x) and g(x). 
        # These should be converted to things that torch can use in its loss func: 
        # logits = log(p / (1 - p)) = f(x) is an arbitrary real number at this stage
        # r = total_counts = softplus(g(x)) (use instead of exp(g(x)))
        if self.h_lst[-1] == 2:
            log_odds = out[0] 
            r = nn.Softplus()(out[1])
            return log_odds, r
        elif self.h_lst[-1] != 1:
            raise ValueError('How many outputs??? Choose 1 or 2 pls')
        return out[0]

    def fit(self, x_train, y_train, x_val, y_val, num_epochs=100):
        self.train()
        self.trained = False
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        train_loss_arr = []
        val_loss_arr = []
        # best_val_loss = float('inf')
        for epoch in range(num_epochs):
            perm = torch.randperm(len(x_train), device=self.device)
            train_epoch_loss = 0.
            for i in range(0, len(x_train), self.batch_size):
                ind = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                if self.h_lst[-1] == 2:
                    # f(x) = log(p / (1 - p)) => p = sigmoid(f(x)) here
                    log_odds_train, r_train = self(x_train[ind])
                    neg_loss_func = \
                        torch.distributions.negative_binomial.NegativeBinomial(
                        total_count=r_train, logits=log_odds_train).log_prob
                elif self.h_lst[-1] == 1: 
                    # f(x) is a real number, and we use softplus to map to a
                    # positive number for rate 
                    neg_loss_func = \
                        torch.distributions.poisson.Poisson(
                        rate=1e-8 + nn.Softplus()(self(x_train[ind]))).log_prob
                else: 
                    raise ValueError('How many outputs??? Choose 1 or 2 pls')

                train_loss = -neg_loss_func(y_train[ind]).mean()
                train_epoch_loss += train_loss.item() * len(ind)
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0) # clip grad
                optimizer.step()
            train_loss_arr.append(train_epoch_loss / len(x_train))
            with torch.no_grad():
                # Compute validation loss
                if self.h_lst[-1] == 2:
                    log_odds_val, r_val = self(x_val)
                    neg_loss_func = \
                        torch.distributions.negative_binomial.NegativeBinomial(
                        total_count=r_val, logits=log_odds_val).log_prob
                elif self.h_lst[-1] == 1:
                    neg_loss_func = \
                        torch.distributions.poisson.Poisson(
                        rate=1e-8 + nn.Softplus()(self(x_val))).log_prob
                else: 
                    raise ValueError('How many outputs??? Choose 1 or 2 pls')
                val_loss = -neg_loss_func(y_val).mean()
                # if epoch == 1: best_val_loss = val_loss.item()
                val_loss_arr.append(val_loss.item())
                # Do early stopping 
                # if val_loss.item() < best_val_loss:
                #     best_val_loss = val_loss.item()
                #     print('Stopping at epoch {}'.format(epoch + 1))
                #     break

        self.trained = True
        return self, train_loss_arr, val_loss_arr

    def predict(self, x, y=None):
        self.eval()
        with torch.no_grad():
            loss = None 
            if self.h_lst[-1] == 2:
                log_odds, r = self(x)
                # p = nn.Sigmoid()(log_odds), i.e. exp(log_odds) = p / (1 - p)
                # The average, <x>, is r p / (1 - p) for this distribution 
                avg = r * torch.exp(log_odds) # prediction
                if y is not None:
                    neg_loss_func = \
                        torch.distributions.negative_binomial.NegativeBinomial(
                        total_count=r, logits=log_odds).log_prob
                    loss = -neg_loss_func(y).mean()
            elif self.h_lst[-1] == 1:
                if y is not None:
                    neg_loss_func = \
                        torch.distributions.poisson.Poisson(
                        rate=1e-8 + nn.Softplus()(self(x))).log_prob
                    loss = -neg_loss_func(y).mean()
                avg = nn.Softplus()(self(x))
            else:
                raise ValueError('How many outputs??? Choose 1 or 2 pls')
            return avg, loss
        
if __name__ == '__main__':
    try:
        import torch_directml # AMD GPU via DirectML (only in the 'dml' conda env)
        device = torch_directml.device()
    except ImportError:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.distributions.Distribution.set_default_validate_args(False)

    # Get the data, and split into test/validation/training sets as 0.6, 0.2, 0.2.
    # path = 'C:/Users/anant/OneDrive - The University of Chicago/03 UChicago/Courses/DATA 377/Problem Set 2/data/'
    # df = pd.read_csv('retail.csv')
    df = pd.read_csv('retail.csv')
    x = df.drop(columns=['Quantity']).to_numpy()
    y = df['Quantity'].to_numpy() 
    gen = torch.Generator().manual_seed(12)
    train_data, val_data, test_data = torch.utils.data.random_split(range(len(df)), \
        [0.6, 0.2, 0.2], generator=gen)
    x_train, y_train = x[train_data.indices], y[train_data.indices]
    x_val, y_val = x[val_data.indices], y[val_data.indices]
    x_test, y_test = x[test_data.indices], y[test_data.indices]

    # Convert x_train into a dictionary where every value in x_train gets 
    # mapped to an integer. 
    x_to_int = [{} for i in range(len(x[1]))]
    count = 1
    for i in range(len(x_train)):
        for j in range(len(x_train[0])):
            if x_train[i][j] not in x_to_int[j].keys():
                x_to_int[j][x_train[i][j]] = count
                count += 1

    def encode(x): 
        return np.array([[x_to_int[j].get(x[i][j], 0) for j in range(len(x[i]))] \
            for i in range(len(x))])
    

    # For each feature, replace it with the mean of the target for that value.
    # Fit the encoder on training data, then apply that same encoding to val/test
    # (done on the raw numpy arrays, before x and y are converted to tensors)
    enc = TargetEncoder(cols=[0, 1])
    xt_train = enc.fit_transform(x_train, y_train)
    xt_val = enc.transform(x_val)
    xt_test = enc.transform(x_test)

    # Now everything is numerical, so convert to tensors
    xt_train = torch.from_numpy(xt_train.to_numpy().copy()).float().to(device)
    xt_val = torch.from_numpy(xt_val.to_numpy().copy()).float().to(device)
    xt_test = torch.from_numpy(xt_test.to_numpy().copy()).float().to(device)

    # Scale the x_train, x_val, x_test:
    xt_train_means = torch.mean(xt_train, 0)
    xt_train_std = torch.std(xt_train, 0 , correction=0)
    xt_train = (xt_train - xt_train_means) / xt_train_std

    # And scale x_test, x_val data by the train values:
    xt_val = (xt_val - xt_train_means) / xt_train_std
    xt_test = (xt_test - xt_train_means) / xt_train_std

    x_train = torch.from_numpy(encode(x_train).copy()).long().to(device)
    x_val = torch.from_numpy(encode(x_val).copy()).long().to(device)
    x_test = torch.from_numpy(encode(x_test).copy()).long().to(device)
    y_train = torch.from_numpy(y_train.copy()).float().to(device)
    y_val = torch.from_numpy(y_val.copy()).float().to(device)
    y_test = torch.from_numpy(y_test.copy()).float().to(device)

    # Combine into (N, 4): [TE_stock, TE_cust, id_stock, id_cust]. The ids are
    # stored as floats here and cast back to long inside forward().
    x_train = torch.cat([xt_train, x_train.float()], dim=1)
    x_val = torch.cat([xt_val, x_val.float()], dim=1)
    x_test = torch.cat([xt_test, x_test.float()], dim=1)

    # d_in = x_train.size()[1]
    d_in = count
    d_emb = 30
    batch_size = 2048
    lr = 1e-3

    h1_arr, h2_arr = [10], []
    d1_out, d2_out = 2, 1

    # Model 1, 1: use negative binomial loss and 1 hidden layer
    num_epochs_11 = 11 # Stopping around 11 epochs is optimal for early stopping
    net11 = MLP_emb(d_in, h1_arr, d1_out, d_emb, lr, batch_size, device)
    _, train_loss_arr, val_loss_arr = net11.fit(x_train, y_train, x_val, y_val, num_epochs_11)

    plt.figure()
    plt.title('Negative binomial loss vs. time, 1 hidden layer')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(range(2, len(train_loss_arr) + 1), train_loss_arr[1:], label='Training')
    plt.plot(range(2, len(val_loss_arr) + 1), val_loss_arr[1:], label='Validation')
    plt.legend() 

    # Model 1, 2: use poisson loss and 1 hidden layer
    num_epochs_12 = 60 # Stopping around 60 epochs is optimal for early stopping
    net12 = MLP_emb(d_in, h1_arr, d2_out, d_emb, lr, batch_size, device)
    _, train_loss_arr, val_loss_arr = net12.fit(x_train, y_train, x_val, y_val, num_epochs_12)

    plt.figure()
    plt.title('Poisson loss vs. time, 1 hidden layer')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(range(2, len(train_loss_arr) + 1), train_loss_arr[1:], label='Training')
    plt.plot(range(2, len(val_loss_arr) + 1), val_loss_arr[1:], label='Validation')
    plt.legend() 

    # Model 2, 1: use negative binomial loss and 0 hidden layer
    num_epochs_21 = 20 # Stopping around 20 epochs is optimal for early stopping
    net21 = MLP_emb(d_in, h2_arr, d1_out, d_emb, lr, batch_size, device)
    _, train_loss_arr, val_loss_arr = net21.fit(x_train, y_train, x_val, y_val, num_epochs_21)

    plt.figure()
    plt.title('Negative binomial loss vs. time, 0 hidden layers')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(range(2, len(train_loss_arr) + 1), train_loss_arr[1:], label='Training')
    plt.plot(range(2, len(val_loss_arr) + 1), val_loss_arr[1:], label='Validation')
    plt.legend() 

    # Model 2, 2: use poisson loss and 0 hidden layers 
    num_epochs_22 = 100 # Loss is still going down
    net22 = MLP_emb(d_in, h2_arr, d2_out, d_emb, lr, batch_size, device)
    _, train_loss_arr, val_loss_arr = net22.fit(x_train, y_train, x_val, y_val, num_epochs_22)

    plt.figure()
    plt.title('Poisson vs. time, 0 hidden layers')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(range(2, len(train_loss_arr) + 1), train_loss_arr[1:], label='Training')
    plt.plot(range(2, len(val_loss_arr) + 1), val_loss_arr[1:], label='Validation')
    plt.legend() 


    # Compare all four models. Both log_probs are full log-likelihoods, so the
    # NB and Poisson losses are directly comparable.
    models = {
        'NB, 1 hidden layer':       net11,
        'Poisson, 1 hidden layer':  net12,
        'NB, 0 hidden layers':      net21,
        'Poisson, 0 hidden layers': net22,
    }
    print(f'{"Model":26s} {"Train":>10s} {"Val":>10s} {"Test":>10s}')
    for name, net in models.items():
        losses = [net.predict(xx, yy)[1].item() for xx, yy in
                  [(x_train, y_train), (x_val, y_val), (x_test, y_test)]]
        print(f'{name:26s} {losses[0]:10.4f} {losses[1]:10.4f} {losses[2]:10.4f}')

    plt.show()

