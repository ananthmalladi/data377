import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sklearn

import numpy as np
import matplotlib.pyplot as plt

import time
# import scipy.stats as scs

path = 'C:/Users/anant/OneDrive - The University of Chicago/03 UChicago/Courses/DATA 377/Problem Set 2/data/'
batch_size = 1024
try:
    import torch_directml # AMD GPU via DirectML (only in the 'dml' conda env)
    device = torch_directml.device()
except ImportError:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get the data, and split into test/validation/training sets as 0.6, 0.2, 0.2. 
# df = pd.read_parquet(path + 'taxi_trips.parquet')
df = pd.read_parquet('taxi_trips.parquet')
x = torch.from_numpy(df.drop(columns=['trip_duration']).to_numpy().copy()) # 7-dim.
x = x.float().to(device)
y = torch.from_numpy(df['trip_duration'].to_numpy().copy()) # 1-dim.
y = y.float().to(device)
data = TensorDataset(x, y)
gen = torch.Generator().manual_seed(12)
train_data, val_data, test_data = torch.utils.data.random_split(data, \
    [0.6, 0.2, 0.2], generator=gen)
# random_split returns a Subset object, which has .dataset (the whole dataset) and
# .indices (referring to which rows of the whole dataset are in each set)
x_train, y_train = train_data.dataset[train_data.indices]
x_val, y_val = val_data.dataset[val_data.indices]
x_test, y_test = test_data.dataset[test_data.indices]

# Scale the x_train, x_val, x_test:
# print(x_train.size())
x_train_means = torch.mean(x_train, 0)
# print(x_train_means.size())
x_train_std = torch.std(x_train, 0 , correction=0) 
x_train = (x_train - x_train_means) / x_train_std

# And scale x_test, x_val data by the train values:
x_val = (x_val - x_train_means) / x_train_std
x_test = (x_test - x_train_means) / x_train_std

# I'll do the same for the y's. 
y_train_means = torch.mean(y_train)
y_train_std = torch.std(y_train, correction=0)
y_train = (y_train - y_train_means) / y_train_std
y_val = (y_val - y_train_means) / y_train_std
y_test = (y_test - y_train_means) / y_train_std

run_baseline = True
if run_baseline == True:
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error as mse
    # sklearn wants numpy arrays on the CPU
    Xtr, ytr, Xv, yv = (t.cpu().numpy() for t in (x_train, y_train, x_val, y_val))
    # Validation MSE of each (y is standardized, so just predicting the mean gives ~1)
    print('Linear:', mse(yv, LinearRegression().fit(Xtr, ytr).predict(Xv)))
    print('SVR:', mse(yv, SVR().fit(Xtr[:10000], ytr[:10000]).predict(Xv)))
    print('MLP:', mse(yv, MLPRegressor(hidden_layer_sizes=(7, 7, 7), early_stopping=True).fit(Xtr, ytr).predict(Xv)))


class MLP(nn.Module):
    '''
    MLP with arbitrary number of hidden layers
    '''
    def __init__(self, num_inputs, h_arr, num_outputs, seed=39):
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
        torch.manual_seed(seed)
        super(MLP, self).__init__() # Initialize nn.module
        self.W_arr = nn.ParameterList()
        self.b_arr = nn.ParameterList()
        h_lst = [num_inputs] + h_arr + [num_outputs]
        self.h_lst = h_lst
        for i in range(1, len(h_lst)):
            W_mat = torch.randn(h_lst[i], h_lst[i-1], requires_grad=True)
            self.W_arr.append(nn.Parameter(W_mat)).to(device)
            self.b_arr.append(nn.Parameter(torch.randn(\
                h_lst[i], 1, requires_grad=True))).to(device)
        # Normalize each layer
        self.ln_arr = nn.ModuleList([nn.LayerNorm(h) for h in h_arr]).to(device)

    def forward(self, X):
        '''
        This function turns inputs into outputs via the model we're working with
        '''
        # h0 = X, h1 = relu(W0 h0 + b0) = feature map, f = W1 h1 + b1
        self.h_func_arr = [X.T]
        for i in range(len(self.W_arr) - 1):
            linear = torch.matmul(self.W_arr[i], self.h_func_arr[-1]) + \
                self.b_arr[i]
            # Normalize output between layers, before doing nonlinear part
            linear = self.ln_arr[i](linear.T).T
            
            nonlinear = nn.ReLU()(linear)
            self.h_func_arr.append(nonlinear)
        self.f = torch.matmul(self.W_arr[-1], self.h_func_arr[-1]) + self.b_arr[-1]
        # Need to convert the real number f[2] into positive number for st.dev.
        return (self.f[0], torch.exp(self.f[1]))
    

def loss_func(f_mu, f_sigma, y):
    return (y - f_mu)**2 / (2 * f_sigma**2) + torch.log(f_sigma)

def train(model, X_train, Y_train, X_val, Y_val, lr, num_epochs, batch_size, seed):
    torch.manual_seed(seed)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # train_ds = TensorDataset(X_train, Y_train)
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model.train()
    training_loss_arr = []
    val_loss_arr = []
    for epoch in range(num_epochs):
        train_epoch_loss = 0.
        perm = torch.randperm(len(X_train), device=X_train.device)
        for i in range(0, len(X_train), batch_size):
            model.zero_grad()
            ind = perm[i:i + batch_size]
            xi, yi = X_train[ind], Y_train[ind]
            f_mu, f_sigma = model(xi)
            train_loss = loss_func(f_mu, f_sigma, yi).mean()
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) # CLIP GRADIENTS!!
            optimizer.step()
            train_epoch_loss += train_loss.detach() * len(xi)
        training_loss_arr.append(train_epoch_loss)
        # Compute validation loss (for each epoch)
        with torch.no_grad():
            Y_mu_pred, Y_sigma_pred = model(X_val)
            val_loss = loss_func(Y_mu_pred, Y_sigma_pred, Y_val).mean()        
        val_loss_arr.append(val_loss.item())
    return model, np.array(training_loss_arr) / len(X_train), \
        np.array(val_loss_arr)

def compare_models(X_val, Y_val, model_lst):
    '''
    I'll compare validation loss of each of the models here
    '''
    loss_func_arr = []
    for model in model_lst:
        model.eval()
        with torch.no_grad():
            Y_mu_pred, Y_sigma_pred = model(X_val)
            val_loss = loss_func(Y_mu_pred, Y_sigma_pred, Y_val).mean()
        loss_func_arr.append(val_loss.item())
    return loss_func_arr

# Define models for part c. 
batch_size = 2048
seed = 12
n_in = x_train.size()[1]
n_out = 2


h_arr_lst = [[n_in], [n_in - 2], [n_in + 2], [n_in, n_in], [n_in, n_in, n_in]]
lr_lst = [0.5e-1, 0.5e-1, 0.5e-1, 0.5e-1, 0.5e-1]
num_epochs_lst = [50, 50, 50, 50, 100]
# It looks like 100 epochs is quite good for the third model -- going beyond 
# that makes the validation error decrease on average but look noisy.  

model_lst = []
for i in range(len(h_arr_lst)):
    model = MLP(num_inputs=n_in, h_arr=h_arr_lst[i], num_outputs=n_out)
    model, model_training_loss, model_val_loss = train(model, x_train, \
        y_train, x_val, y_val, lr_lst[i], num_epochs_lst[i], batch_size, seed)
    model_lst.append(model)

    plt.figure()
    plt.plot(range(1, len(model_training_loss) + 1), np.log(model_training_loss))
    plt.xlabel('Epochs') # after {}'.format(int(len(model1_training_loss)/2)))
    plt.ylabel('Log(Training loss)')
    plt.title('Log(Training loss) vs. time')

    plt.figure()
    plt.plot(range(1, len(model_val_loss) + 1), np.log(model_val_loss))
    plt.xlabel('Epochs') # after {}'.format(int(len(model1_training_loss)/2)))
    plt.ylabel('Log(Validation loss)')
    plt.title('Log(Validation loss) vs. time')

model_vals = compare_models(x_val, y_val, model_lst)
for i in range(len(model_vals)):
    print('Validation error for model with hidden layers {}:{}'.format(\
        h_arr_lst[i], model_vals[i]))

# It looks like the [7,7,7] model does the best among these, so I'll use that. 

# I'll report the log-loss on the test set, and the classification accuracy
# for P(trip < 45 minutes), thresholded at 0.5
best = model_lst[-1]
best.eval()
with torch.no_grad():
    mu, sigma = best(x_test) # whole test set, standardized units
t = (45 - y_train_means) / y_train_std # 45 min in standardized units
p = torch.distributions.Normal(mu, sigma).cdf(t) # P(Y < 45 | x)

label = (y_test * y_train_std + y_train_means < 45).float()
p_c = p.clamp(1e-7, 1 - 1e-7) 
print('Log-loss:', F.binary_cross_entropy(p_c, label).item())
print('Accuracy:', ((p > 0.5).float() == label).float().mean().item())

# Baseline: predict the training-set fraction of trips under 45 min 
p0 = (y_train * y_train_std + y_train_means < 45).float().mean()
print('Baseline log-loss:', F.binary_cross_entropy(p0.expand_as(label), label).item())
print('Baseline accuracy:', (label == 1).float().mean().item())

plt.show()
