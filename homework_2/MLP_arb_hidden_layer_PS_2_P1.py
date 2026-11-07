import torch 
import torch.nn as nn

class MLP(nn.Module):
    '''
    MLP with arbitrary number of hidden layers
    '''
    def __init__(self, num_inputs, h_arr, num_outputs):
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
        super(MLP, self).__init__() # Initialize nn.module
        self.W_arr = []
        self.b_arr = []
        h_lst = [num_inputs] + h_arr + [num_outputs]
        for i in range(1, len(h_lst)):
            self.W_arr.append(nn.Parameter(torch.randn(h_lst[i], h_lst[i-1], requires_grad=True)))
            self.b_arr.append(nn.Parameter(torch.randn(h_lst[i], 1, requires_grad=True)))

    def forward(self, X):
        '''
        This function turns inputs into outputs via the model we're working with
        '''
        # h0 = X, h1 = relu(W0 h0 + b0) = feature map, f = W1 h1 + b1
        self.h_func_arr = [X]
        for i in range(len(self.W_arr) - 1):
            self.h_func_arr.append(nn.ReLU()(torch.matmul(self.W_arr[i], \
                self.h_func_arr[-1]) + self.b_arr[i]))
        self.f = torch.matmul(self.W_arr[-1], self.h_func_arr[-1]) + self.b_arr[-1]
        return self.f

    