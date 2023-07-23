# diy-dnn
Simple NumPy feed-forward neural network library from scratch. Applied to MNIST dataset classification task.

## 1. Summary

### 1.1.1 Layers

| Layer | Implemented | Forward *(element-wise)* | Forward *(matrix form)* |
| :---: | :---: | :---: | :---: |
| Dense | ✓ | $z_i^l = \sum_j{w_{ij}^l a_j^{l-1}} + b_i^l$ | $\mathbf{Z}^l = \mathbf{W}^l \cdot \mathbf{A}\^{l-1} + \mathbf{b}^l$ |
| ReLU | ✓ | $a_i = Relu(z_i)$ | $\mathbf{A} = Relu(\mathbf{Z})$ |
| Softmax | ✓ | $$a_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$ | $$\mathbf{A} = \frac{\exp(\mathbf{Z})}{\mathbf{1}^T \cdot \exp(\mathbf{Z})}$$ |
| BatchNorm | ✓ | $$\hat{x}_i = \frac{x_i - \mu(x_j)}{\sqrt{\sigma(x_j)^2 + \epsilon}}$$ <br> $$y_i = \gamma \hat{x}_i + \beta$$ | $$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu}{\sqrt{\mathbf{\sigma}^2 + \epsilon}}$$ <br> $$\mathbf{y} = \gamma \hat{\mathbf{x}} + \beta$$ |
| (Cost) Categorical cross-entropy | ✓ | xxx | xxx |
| Attention | | xxx | xxx |

| Layer | Forward | Backward |
| :---: | :---: | :---: |
| Dense | <img src="media/dense_forward.png" alt="Image" width="600"/> | <img src="media/dense_backward.png" alt="Image" width="600"/> |
| | Notes | Notes |
| Relu | <img src="media/relu_forward.png" alt="Image" width="100"/> | <img src="media/relu_backward.png" alt="Image" width="100"/> |
| | Notes | Notes |
| Softmax | <img src="media/softmax_forward.png" alt="Image" width="300"/> | <img src="media/softmax_backward.png" alt="Image" width="300"/> |
| | Notes | Notes |
| BatchNorm | <img src="media/batchnorm_forward.png" alt="Image" width="300"/> | <img src="media/batchnorm_backward.png" alt="Image" width="300"/> |
| | Notes | Notes |
| (Cost) Categorical cross-entropy | | |
| | Notes | Notes |
| Attention | | |
| | Notes | Notes |

### 1.1.2 Optimisers

| Optimiser | Implemented | Equation | Explanation |
| :---: | :---: | :---: | :---: |
| (Mini-batch) SGD | ✓ |  |  |
| Adam | |  |  |
| AdamW | |  |  |

### 1.1.3 Weight initialisers

| Optimiser | Implemented | Equation | Explanation |
| :---: | :---: | :---: | :---: |
| Zero | ✓ |  |  |
| He | ✓ |  |  |

TEMP: check dense backprop (1/m) term

## 2. Usage
1. Define architectures and train on MNIST dataset in `.../experiments/mnist_classification_train_and_validate.py`
  1. `DATA_CACHE_DIR`, `PLOTS_DIR` and `MODEL_CHECKPOINTS_DIR` define the directories for saving training/validation loop artefacts
  2. `RUN_SETTINGS` is a list of dictionaries. Each element of the list is a neural net architecture (allowing training and validation of several candidate architectures). Each dictionary defines both the neural net `"architecture"`, and training loop settings.
2. Post-process runs in `.../experiments/mnist_classification_evaluate.py`
  1. (Script has similar global parameters to set as `...train_and_validate`)
  2. Generates a summary log of metrics for all runs included
  3. Generates a visualisation of sample MNIST inferences for the trained model (on the validation set)

## 3. Notation / conventions
- Temp: e.g. dz = del(L)/del(z)

## 3. Matrix Calculus Cheat-sheet

### 3.1. Numerator-layout convention
* Given the following input objects...
    * $x, y$: Scalars
    * $\mathbf{x}, \mathbf{y}$: Column vectors, sizes respectively (n, 1), (m, 1)
    * $\mathbf{X}, \mathbf{Y}$: Matrices, sizes respectively (n, m), (p, q)
* ...the following derivatives (with dimensions) are possible
    * $\frac{\partial y}{\partial x}$ // scalar-by-scalar // Scalar output (size 1)
    * $\frac{\partial \mathbf{y}}{\partial x}$ // vector-by-scalar // Column vector output (same shape as $\mathbf{y}$)
    * $\frac{\partial y}{\partial \mathbf{x}}$ // scalar-by-vector // Row vector output (same shape as $\mathbf{x}^T$)
    * $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ // vector-by-vector (Jacobian) // Matrix output (shape (n, m) i.e. num rows of $\mathbf{y}$ and num columns of $\mathbf{x}^T$)
    * $\frac{\partial \mathbf{Y}}{\partial x}$ // matrix-by-scalar // Matrix output, same shape as input matrix \mathbf{Y} (n, m)
    * $\frac{\partial y}{\partial \mathbf{X}}$ // scalar-by-matrix // Matrix output, same shape as transpose of denominator matrix (same shape as $\mathbf{X}^T$, i.e. (q, p))
 
### 3.2. Cost derivative abbreviation
Following the standard machine-learning abbreviation of the cost. 
* Given:
    * Scalar cost, $J$
    * Any neural network layer object, generally a vector e.g. $\mathbf{z}$ or matrix e.g. $\mathbf{W}$
* A partial differential of the cost w.r.t the network object, e.g. $\frac{\partial J}{\partial \mathbf{z}}$, can be abbreviated $d\mathbf{z}$
* Following the numerator-layout convention above, the non-scalar properties in the denominator should be transposed 

#### 3.2.1. Work-through with a specific example

<p align="center">
  <img src="media/dense_forward.png" alt="Image" width="800"/>
</p>

```math
\mathbf{z} = \begin{pmatrix}
z_1 \\
z_2
\end{pmatrix}
```

<p align="center">
  <img src="media/dense_backward.png" alt="Image" width="800"/>
</p>

