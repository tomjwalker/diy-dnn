# diy-dnn
Simple NumPy feed-forward neural network library from scratch. Applied to MNIST dataset classification task.

## 1. Summary

### 1.1.1 Layers - forward propagation

| Implemented | Layer | Element-wise | Matrix form | Diagram | 
| --- | --- | --- | --- | --- |
| Dense | ✓ | $\mathbf{Z}^l = \mathbf{W}^l \cdot \mathbf{A}\^{l-1} + \mathbf{b}^l$ | $\mathbf{Z}^l = \mathbf{W}^l \cdot \mathbf{A}\^{l-1} + \mathbf{b}^l$ | <img src="./media/dense_forward.png" alt="Image" style="max-width: 50px;"/> |
| ReLU | ✓ | $\mathbf{A}^l = Relu(\mathbf{Z}^l)$ | $\mathbf{A}^l = Relu(\mathbf{Z}^l)$ | --- |
| Softmax | ✓ | --- | --- | --- |
| BatchNorm | ✓ | --- | --- | --- |

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

| Transformation | Element-wise | Matrix form | Notes |
| --- | --- | --- | --- |
| --- | --- | --- | --- |
| --- | --- | --- | --- |
| --- | --- | --- | --- |
