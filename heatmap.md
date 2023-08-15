# Original Fluid Flow

```python
import numpy as np
import seaborn as sns
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
color = scipy.io.loadmat('CCcool.mat')
cc = color['CC']
newcmp = LinearSegmentedColormap.from_list('', cc)

tensor = np.load('tensor.npz')['arr_0']
tensor = tensor[:, :, : 150]
M, N, T = tensor.shape

plt.rcParams['font.size'] = 13
plt.rcParams['mathtext.fontset'] = 'cm'
fig = plt.figure(figsize = (7, 8))
id = np.array([5, 10, 15, 20, 25, 30, 35, 40])
for t in range(8):
    ax = fig.add_subplot(4, 2, t + 1)
    ax = sns.heatmap(tensor[:, :, id[t] - 1], cmap = newcmp, vmin = -5, vmax = 5, cbar = False)
    ax.contour(np.linspace(0, N, N), np.linspace(0, M, M), tensor[:, :, id[t] - 1],
               levels = np.linspace(0.15, 15, 30), colors = 'k', linewidths = 0.7)
    ax.contour(np.linspace(0, N, N), np.linspace(0, M, M), tensor[:, :, id[t] - 1],
               levels = np.linspace(-15, -0.15, 30), colors = 'k', linestyles = 'dashed', linewidths = 0.7)
    plt.xticks([])
    plt.yticks([])
    plt.title(r'$t = {}$'.format(id[t]))
    for _, spine in ax.spines.items():
        spine.set_visible(True)
plt.show()
fig.savefig('fluid_flow_heatmap.png', bbox_inches = 'tight')
```

# Sparse Fluid Flow

```python
import cupy as np
sparse_tensor = np.asnumpy(sparse_tensor)

import numpy as np
import seaborn as sns
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
color = scipy.io.loadmat('CCcool.mat')
cc = color['CC']
newcmp = LinearSegmentedColormap.from_list('', cc)

M, N, T = sparse_tensor.shape

plt.rcParams['font.size'] = 13
plt.rcParams['mathtext.fontset'] = 'cm'
fig = plt.figure(figsize = (7, 8))
id = np.array([5, 10, 15, 20, 25, 30, 35, 40])
for t in range(8):
    ax = fig.add_subplot(4, 2, t + 1)
    ax = sns.heatmap(sparse_tensor[:, :, id[t] - 1], 
                     cmap = newcmp, vmin = -5, vmax = 5, cbar = False)
    plt.xticks([])
    plt.yticks([])
    plt.title(r'$t = {}$'.format(id[t]))
    for _, spine in ax.spines.items():
        spine.set_visible(True)
plt.show()
fig.savefig('fluid_flow_heatmap_at_95_missing_rate.png', bbox_inches = 'tight')
```

# Reconstructed Fluid Flow

```python
import cupy as np
tensor_hat = np.asnumpy(tensor_hat)

import numpy as np
import seaborn as sns
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
color = scipy.io.loadmat('CCcool.mat')
cc = color['CC']
newcmp = LinearSegmentedColormap.from_list('', cc)

M, N, T = tensor_hat.shape

plt.rcParams['font.size'] = 13
plt.rcParams['mathtext.fontset'] = 'cm'
fig = plt.figure(figsize = (7, 8))
id = np.array([5, 10, 15, 20, 25, 30, 35, 40])
for t in range(8):
    ax = fig.add_subplot(4, 2, t + 1)
    ax = sns.heatmap(tensor_hat[:, :, id[t] - 1], cmap = newcmp, vmin = -5, vmax = 5, cbar = False)
    ax.contour(np.linspace(0, N, N), np.linspace(0, M, M), tensor_hat[:, :, id[t] - 1],
               levels = np.linspace(0.15, 15, 30), colors = 'k', linewidths = 0.7)
    ax.contour(np.linspace(0, N, N), np.linspace(0, M, M), tensor_hat[:, :, id[t] - 1],
               levels = np.linspace(-15, -0.15, 30), colors = 'k', linestyles = 'dashed', linewidths = 0.7)
    plt.xticks([])
    plt.yticks([])
    plt.title(r'$t = {}$'.format(id[t]))
    for _, spine in ax.spines.items():
        spine.set_visible(True)
plt.show()
fig.savefig('reconstructed_fluid_flow_heatmap_at_95_missing_rate.png', bbox_inches = 'tight')
```
