import os
import matplotlib.pyplot as plt
import pandas as pd

# load train.csv from this script's folder, regardless of where Python is run from
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, 'train.csv'))
x1 = df['x1'].to_numpy()
x2 = df['x2'].to_numpy()
y = df['y'].to_numpy()

# 2D view laid out like plot_surface_2d (same axes, default viridis colormap)
# so it can be compared side by side with the fitted surfaces in part (d)
plt.figure()
plt.scatter(x1, x2, c=y, s=15)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.title('Training data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(label='y')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'train_data.png'), dpi=300)

plt.show()
