"""
Skeleton code for creating the Random Kitchen Sinks (RKS) and 1-hidden-layer MLP (MLP) models.

Note you are *welcome* to completely ignore this file and use your own code, but it's here if you want to use it as a starting point.
"""
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

import matplotlib.pyplot as plt


from dataclasses import dataclass
from typing import Optional
import numpy as np

from typing import Callable


"""
For Random Kitchen Sinks (RKS)
"""
class RandomKitchenSinks:
    """
    Random Kitchen Sinks featurization with ridge regression.
    """
    def __init__(self, D: int = 1000, sigma: float = 1.0, ridge_lambda: float = 1e-3):
        """
        Potential hyper-parameters you can consider (feel free to add more):
            D: total number of random features (= 2h in the assignment text; if you use both sin and cos features, D should be an even number)
            sigma: scale parameter for the RBF kernel, used for initializing the weight parameters
            ridge_lambda: ridge regularization strength
        """
        # TODO: initialize the model parameters
        # assert D % 2 == 0, 'D must be even'
        self.D = D
        self.sigma = sigma 
        self.ridge_lambda = ridge_lambda

        self.flag = False

    def _features(self, X: np.ndarray) -> np.ndarray:
        """
        Convert the input data X into the random kitchen sinks features. You will likely need to use the parameters initialized in the __init__ method.
        The resulting features should be of shape (n, D or D[+1]), where the plus one is for the intercept term.
        """
        # TODO: implement the feature transformation
        d = (X.shape)[1]
        n = (X.shape)[0]
        if self.flag is False:
            # Choose w and b randomly, just once for each model
            self.w = np.random.normal(scale=self.sigma, size=(d, int(self.D/2)))
            self.b = np.random.random(size=int(self.D/2)) * 2 * np.pi
            self.flag = True 

        # Add a column of 1's at the end to model an intercept
        lin = X @ self.w + self.b
        sin_arr = np.sin(lin)
        cos_arr = np.cos(lin)
        return np.c_[np.concat([sin_arr, cos_arr], axis=1), np.ones(n)]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the ridge regression model to the (1) features computed from the input data X and (2) target values y.
        """
        # TODO: optional code before getting features
        phi = self._features(X)  # shape (n, D or D[+1])
        # TODO: fit the ridge regression model and obtain the fitted ridge coefficients
        n = (phi.shape)[1]
        denom = np.matmul(phi.T, phi) + self.ridge_lambda * np.identity(n)
        num = np.matmul(phi.T, y)
        self.opt_params = np.linalg.solve(denom, num)  # solves denom @ beta = num; 
        # this is faster than inverting the matrix, it seems
        return self.opt_params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the input data X.
        """
        phi = self._features(X)  # shape (n, D or D[+1])
        # self.coef_ = 
        return phi @ self.opt_params  # shape (n,)

def sweep_rks_model(sweep_D: List[int], #sweep_sigma: List[int], sweep_lambda: List[int], \
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple:
    """
    An example sweeping function that sweeps over the number of random features D, it then returns the best model in terms of validation MSE.
    Args:
        sweep_D: list of numbers of random features to sweep over
        X_train: training input data
        y_train: training target values
        X_val: validation input data
        y_val: validation target values
    Returns:
        a tuple (best_model, best_val_mse, best_D, train_mse_arr, val_mse_arr)
    """
    best_model = None
    best_val_mse = float('inf')
    best_D = None
    train_mse_arr = []
    val_mse_arr = []
    for D in sweep_D:
        # for l in sweep_lambda:
        #     for s in sweep_sigma:
        # TODO: fit the model and get the validation MSE
        # model = RandomKitchenSinks(D, s, l).fit(X_train, y_train)
        model = RandomKitchenSinks(D)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        train_mse = np.mean((y_train_pred - y_train)**2)
        val_mse = np.mean((y_val_pred - y_val)**2)
        train_mse_arr.append(train_mse)
        val_mse_arr.append(val_mse)

        if val_mse < best_val_mse:
            best_model = model
            best_val_mse = val_mse
            best_D = D

    return best_model, best_val_mse, best_D, train_mse_arr, val_mse_arr

"""
For 1-hidden-layer MLP (MLP)
"""
class MLP(nn.Module):
    """1-hidden-layer MLP for regression."""
    def __init__(self, d_in: int, d_hidden: int = 128, d_out: int = 1, lr: float = 1e-3, batch_size: int = 128, activation: str = "relu", device: str = "cpu"):
        """
        Potential hyper-parameters you can consider (feel free to add more):
            d_hidden: hidden dimension
            activation: activation function
            lr: learning rate
            batch_size: batch size
        """
        torch.manual_seed(12)
        super().__init__()
        # DO NOT forget this -- GPU will make NN training much faster if you have access to one
        self.device = device
        # TODO: build up the 1-hidden-layer MLP model with modules like `nn.Linear`, `nn.Sequential`, `nn.ReLU`, etc.
        # self.net = None

        self.lr = lr
        self.batch_size = batch_size 

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('I really don\'t want to hard code every function in torch.nn here... use sigmoid, tanh, or relu pls')

        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden).to(device),
            self.activation,
            nn.Linear(d_hidden, d_out).to(device),
        )


    def fit(self, X: np.ndarray, y: np.ndarray, num_epochs=100):
        # turn the numpy data into torch tensors and move to the device
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().view(-1, 1).to(self.device)

        self.net.train()  # make sure to set the model to training mode
        # TODO: fit the MLP model and obtain the fitted MLP coefficients
        # optimizer = xxx
        # loss_fn = yyy
        # for epoch in range(num_epochs):
        #     # set up batch data loader and fit the model within each batch

        
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        for epoch in range(num_epochs):
            perm = torch.randperm(len(X), device=X.device)
            for i in range(0, len(X), self.batch_size):
                self.net.zero_grad()
                ind = perm[i:i + self.batch_size]
                xi, yi = X[ind], y[ind]
                y_pred = self.net(xi)
                train_loss = loss_fn(y_pred, yi).mean()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0) # CLIP GRADIENTS!!
                optimizer.step()
        # return model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            return self.forward(X).detach().cpu().numpy()

def mlp_sweep(sweep_values: List[float], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple:
    """
    See the example `sweep_rks_model` above for inspiration. Good luck!
    """
    batch_size = 128
    d_in = len(X_train[0])
    d_h = 128
    d_out = 1

    best_model = None
    best_val_mse = float('inf')
    best_lr = None
    mse_train_arr = []
    mse_val_arr = []
    for lr in sweep_values:
        # for l in sweep_lambda:
        #     for s in sweep_sigma:
        # TODO: fit the model and get the validation MSE
        # model = RandomKitchenSinks(D, s, l).fit(X_train, y_train)
        model = MLP(d_in, d_h, d_out, lr, batch_size, 'relu', 'cpu')
        model.fit(X_train, y_train)
        model.net.eval()
        y_train_pred = model.predict(X_train).ravel()
        y_val_pred = model.predict(X_val).ravel()
        train_mse = np.mean((y_train_pred - y_train)**2)
        val_mse = np.mean((y_val_pred - y_val)**2)
        mse_train_arr.append(train_mse)
        mse_val_arr.append(val_mse)

        if val_mse < best_val_mse:
            best_model = model
            best_val_mse = val_mse
            best_lr = lr

    return best_model, best_val_mse, best_lr, mse_train_arr, mse_val_arr

# Get the data

@dataclass
class Split:
    X: np.ndarray
    y: Optional[np.ndarray] = None  # None for the unlabeled test split


@dataclass
class HW2Dataset:
    train: Split
    val: Split
    test: Split

    @classmethod
    def from_csv(cls, train_path: str, val_path: str, test_path: str):
        train_data = np.loadtxt(train_path, delimiter=",", skiprows=1)
        val_data = np.loadtxt(val_path, delimiter=",", skiprows=1)
        test_data = np.loadtxt(test_path, delimiter=",", skiprows=1)

        train_split = Split(train_data[:, :2], train_data[:, 2])
        val_split = Split(val_data[:, :2], val_data[:, 2])
        # test.csv contains only x1, x2 -- the labels are held out for grading
        test_split = Split(test_data[:, :2], None)

        return cls(train_split, val_split, test_split)


def plot_surface_2d(model_fn: Callable[[np.ndarray], np.ndarray] | nn.Module,
                    xlim: Tuple[float, float], ylim: Tuple[float, float],
                    title: str = "Fitted surface", filename: str = "surface.png", grid: int = 200):
    """
    An example plotting function that handles both callable functions (e.g. model.predict) for a sk-learn-like model and a PyTorch NN.

    Example usage:
    >>> # some model with a `.predict` method
    >>> rks_model = fit_rks_model(X_train, y_train)
    >>> plot_surface_2d(rks_model.predict, (-3, 3), (-3, 3), title="RKS surface")
    >>> # some PyTorch model
    >>> mlp_model = fit_mlp_model(X_train, y_train)
    >>> # direct feed it to the plotting function, no need to call `.predict`
    >>> plot_surface_2d(mlp_model, (-3, 3), (-3, 3), title="MLP surface")
    """
    xs = np.linspace(xlim[0], xlim[1], grid)
    ys = np.linspace(ylim[0], ylim[1], grid)
    X1, X2 = np.meshgrid(xs, ys)
    Xgrid = np.stack([X1.ravel(), X2.ravel()], axis=1)
    if isinstance(model_fn, nn.Module):
        with torch.no_grad():
            X_grid = torch.from_numpy(Xgrid).float().to(next(model_fn.parameters()).device)
            Z = model_fn(X_grid).cpu().numpy().reshape(grid, grid)
    else:
        Z = model_fn(Xgrid).reshape(grid, grid)

    plt.figure()
    plt.contourf(X1, X2, Z, levels=30)  # default colormap
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar(label="pred")
    plt.tight_layout()
    # or if you want to directly show it
    # plt.show()
    plt.savefig(filename, dpi=300)

# Plot the data and run the sweeps
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, 'train.csv'))
    x1 = df['x1'].to_numpy()
    x2 = df['x2'].to_numpy()
    y = df['y'].to_numpy()

    # data is scattered points (not a grid), so use tricontourf instead of contourf;
    # otherwise laid out like plot_surface_2d so the plots can be compared directly
    plt.figure()
    plt.tricontourf(x1, x2, y, levels=30)  # default colormap
    plt.xlim(-3, 3)  # match the [-3, 3] grid used by plot_surface_2d
    plt.ylim(-3, 3)
    plt.title('Training data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar(label='y')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'train_data_contour.png'), dpi=300)


    np.random.seed(0)
    # resolve CSV paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ds = HW2Dataset.from_csv(os.path.join(script_dir, "train.csv"),
                             os.path.join(script_dir, "val.csv"),
                             os.path.join(script_dir, "test.csv"))
    print(ds.train.X.shape, ds.train.y.shape)
    print(ds.val.X.shape, ds.val.y.shape)
    print(ds.test.X.shape, ds.test.y)

    X_train, X_val, X_test = ds.train.X, ds.val.X, ds.test.X
    y_train, y_val = ds.train.y, ds.val.y

    # Sweep the RKS model on a range of D's
    D_arr = list(range(20, 1000, 10))
    best_rks, best_rks_mse, best_D, train_mse_rks_arr, val_mse_rks_arr = \
        sweep_rks_model(D_arr, X_train, y_train, X_val, y_val)

    print('Best D for RKS: ', best_D)
    print('Best validation MSE of RKS: ', best_rks_mse)

    plt.figure()
    plt.plot(D_arr, train_mse_rks_arr, label='Train')
    plt.plot(D_arr, val_mse_rks_arr, label='Validation')
    plt.title('MSE Error for RKS as a function of number of features')
    plt.xlabel('Number of features')
    plt.ylabel('MSE Error')
    plt.legend()

    # Sweep the MLP on a range of learning rates
    lr_arr = np.logspace(-4, -1, 15)
    best_mlp, best_mlp_mse, best_lr, train_mse_mlp_arr, val_mse_mlp_arr = \
        mlp_sweep(lr_arr, X_train, y_train, X_val, y_val)

    print('Best learning rate for MLP: ', best_lr)
    print('Best validation MSE of MLP: ', best_mlp_mse)

    plt.figure()
    plt.plot(np.log(lr_arr), np.log(train_mse_mlp_arr), label='Train')
    plt.plot(np.log(lr_arr), np.log(val_mse_mlp_arr), label='Validation')
    plt.title('MSE Error for MLP as a function of learning rate')
    plt.xlabel('Log(learning rate)')
    plt.ylabel('Log(MSE Error)')
    plt.legend()

    # Predict values on test array and write to test file for submission
    y_test_rks = best_rks.predict(X_test)
    y_test_mlp = best_mlp.predict(X_test)

    # Test against a couple of benchmarks
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor

    lin = LinearRegression().fit(X_train, y_train)
    print('Validation MSE of linear regression: ', np.mean((lin.predict(X_val) - y_val)**2))

    # Pick max_depth of regression tree by testing on validation data
    # (can't go too high on this -- otherwise you overfit the training data) 
    tree_val_mse = {}
    tree_lst = []
    for depth in range(1, 16):
        tree = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X_train, y_train)
        tree_lst.append(tree)
        tree_val_mse[depth] = np.mean((tree.predict(X_val) - y_val)**2)
    best_depth = min(tree_val_mse, key=tree_val_mse.get)
    best_tree = tree_lst[best_depth-1]
    print('Best max_depth for regression tree: ', best_depth)
    print('Best validation MSE of regression tree: ', tree_val_mse[best_depth])

    # Write test predictions: one per row, same order as test.csv
    # (.ravel() because the MLP returns shape (n, 1))
    pd.DataFrame({'y': y_test_rks.ravel()}).to_csv(os.path.join(script_dir, 'rks_test_predictions.csv'), index=False)
    pd.DataFrame({'y': y_test_mlp.ravel()}).to_csv(os.path.join(script_dir, 'mlp_test_predictions.csv'), index=False)

    # save figures next to this script, regardless of where Python is run from
    plot_surface_2d(best_rks.predict, xlim=[-3,3], ylim=[-3,3], title='Data fitted to RKS', filename=os.path.join(script_dir, 'rks_fitted.png'))
    plot_surface_2d(best_mlp, xlim=[-3,3], ylim=[-3,3], title='Data fitted to MLP', filename=os.path.join(script_dir, 'mlp_fitted.png'))
    plot_surface_2d(lin.predict, xlim=[-3,3], ylim=[-3,3], title='Data fitted to linear regression', filename=os.path.join(script_dir, 'lin_fitted.png'))
    plot_surface_2d(best_tree.predict, xlim=[-3,3], ylim=[-3,3], title='Data fitted to regression tree', filename=os.path.join(script_dir, 'tree_fitted.png'))

    plt.show()