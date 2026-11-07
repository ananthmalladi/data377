import os
import pandas as pd
# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset

# import sklearn

import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as scs

# path = 'C:/Users/anant/OneDrive - The University of Chicago/03 UChicago/Courses/DATA 377/Problem Set 2/data/'
batch_size = 8
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, 'retail.csv'))
x = df.drop(columns=['Quantity']).to_numpy() # 2-dim.
y = df['Quantity'] # 1-dim.

stock_codes = []
customer_id = []
count = 0
for row in df.itertuples(index=False):
    count += 1
    if row.CustomerID not in customer_id:
        customer_id.append(row.CustomerID)
    if row.StockCode not in stock_codes:
        stock_codes.append(row.StockCode)

print(len(stock_codes))
print(len(customer_id))

# One hot encoding, assigning a different unit basis vector of R^n to each 
# input, is not the best option here given that there are 3665 stock codes and 
# 4339 customer id's.  

