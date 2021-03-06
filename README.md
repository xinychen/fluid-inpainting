# Fluid Inpainting
Inpainting Fluid Dynamics with Tensor Decomposition (NumPy)

## Fluid Dynamic Data

```python
import numpy as np
dense_tensor = np.load('tensor.npz')['arr_0']
dense_tensor = dense_tensor[:, :, : 150]

import seaborn as sns
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
color = scipy.io.loadmat('CCcool.mat')
cc = color['CC']
newcmp = LinearSegmentedColormap.from_list('', cc)

fig = plt.figure(figsize = (15, 8))
for t in range(9):
    ax = fig.add_subplot(3, 3, t + 1)
    sns.heatmap(dense_tensor[:, :, t], cmap = newcmp, vmin = -5, vmax = 5)
    plt.title('t = {}'.format(t + 1))
    plt.xticks([])
    plt.yticks([])
```

## CP Tensor Decomposition

```python
import numpy as np

def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(int(tensor_size.shape[0])):
        if i != mode:
            index.append(int(i))
    size = []
    for i in index:
        size.append(int(tensor_size[i]))
    return np.moveaxis(np.reshape(mat, size, order = 'F'), 0, mode)
```

```python
def kr_prod(a, b):
    return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0] * b.shape[0], -1)

def update_cg(var, r, q, Aq, rold):
    alpha = rold / np.inner(q, Aq)
    var = var + alpha * q
    r = r - alpha * Aq
    rnew = np.inner(r, r)
    q = r + (rnew / rold) * q
    return var, r, q, rnew

def ell(ind_mode, f_mat, mat):
    return ((f_mat @ mat.T) * ind_mode) @ mat

def conj_grad(sparse_tensor, ind, fact_mat, mode, maxiter = 5):
    dim, rank = fact_mat[mode].shape
    ind_mode = ten2mat(ind, mode)
    f = np.reshape(fact_mat[mode], -1, order = 'F')
    temp = []
    for k in range(3):
        if k != mode:
            temp.append(fact_mat[k])
    mat = kr_prod(temp[-1], temp[0])
    r = np.reshape(ten2mat(sparse_tensor, mode) @ mat
                   - ell(ind_mode, fact_mat[mode], mat), -1, order = 'F')
    q = r.copy()
    rold = np.inner(r, r)
    for it in range(maxiter):
        Q = np.reshape(q, (dim, rank), order = 'F')
        Aq = np.reshape(ell(ind_mode, Q, mat), -1, order = 'F')
        alpha = rold / np.inner(q, Aq)
        f, r, q, rold = update_cg(f, r, q, Aq, rold)
    return np.reshape(f, (dim, rank), order = 'F')

def cp_decompose(dense_tensor, sparse_tensor, rank, maxiter = 50):
    dim = sparse_tensor.shape
    fact_mat = []
    for k in range(3):
        fact_mat.append(0.01 * np.random.randn(dim[k], rank))
    ind = sparse_tensor != 0
    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    show_iter = 1
    for it in range(maxiter):
        for k in range(3):
            fact_mat[k] = conj_grad(sparse_tensor, ind, fact_mat, k)
        tensor_hat = np.einsum('ur, vr, xr -> uvx', 
                               fact_mat[0], fact_mat[1], fact_mat[2])
        if (it + 1) % show_iter == 0:
            print('Iter: {}'.format(it + 1))
            rse = (np.linalg.norm(tensor_hat[pos_test] - dense_tensor[pos_test], 2) 
                  / np.linalg.norm(dense_tensor[pos_test], 2))
            print(rse)
            print()
    return tensor_hat, fact_mat
```

## Evaluate CP Decomposition

```python
import numpy as np
dense_tensor = np.load('tensor.npz')['arr_0']
np.random.seed(1)

dense_tensor = dense_tensor[:, :, : 150]

M, N, T = dense_tensor.shape
random_tensor = np.random.rand(M, N, T)
p = 0.9
sparse_tensor = dense_tensor * np.round(random_tensor + 0.5 - p)

import time
start = time.time()
rank = 100
tensor_hat, fact_mat = cp_decompose(dense_tensor, sparse_tensor, rank)
end = time.time()
print('Running time: %d seconds'%(end - start))
```

## Cylinder Wake Data with 90% Missing Values

```python
import seaborn as sns
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
color = scipy.io.loadmat('CCcool.mat')
cc = color['CC']
newcmp = LinearSegmentedColormap.from_list('', cc)

fig = plt.figure(figsize = (15, 8))
for t in range(9):
    ax = fig.add_subplot(3, 3, t + 1)
    sns.heatmap(sparse_tensor[:, :, t], cmap = newcmp, vmin = -5, vmax = 5)
    plt.title('t = {}'.format(t + 1))
    plt.xticks([])
    plt.yticks([])
```

## Reconstructed Fluid Dynamics

```python
import seaborn as sns
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
color = scipy.io.loadmat('CCcool.mat')
cc = color['CC']
newcmp = LinearSegmentedColormap.from_list('', cc)

fig = plt.figure(figsize = (15, 8))
i = 1
for t in [0, 10, 20]:
    ax = fig.add_subplot(3, 3, i)
    sns.heatmap(dense_tensor[:, :, t], cmap = newcmp, vmin = -5, vmax = 5)
    plt.title('Ground truth data (t = {})'.format(t + 1))
    plt.xticks([])
    plt.yticks([])
    i += 1
for t in [0, 10, 20]:
    ax = fig.add_subplot(3, 3, i)
    sns.heatmap(sparse_tensor[:, :, t], cmap = newcmp, vmin = -5, vmax = 5)
    plt.title('Sparse data (t = {})'.format(t + 1))
    plt.xticks([])
    plt.yticks([])
    i += 1
for t in [0, 10, 20]:
    ax = fig.add_subplot(3, 3, i)
    sns.heatmap(tensor_hat[:, :, t], cmap = newcmp, vmin = -5, vmax = 5)
    plt.title('Reconstructed data (t = {})'.format(t + 1))
    plt.xticks([])
    plt.yticks([])
    i += 1
```
