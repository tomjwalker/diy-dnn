# diy-dnn
Simple NumPy feed-forward neural network library from scratch. Applied to MNIST dataset classification task.

## 1. Summary

### 1.1. Layers

| Implemented | Layer | Forward equation | Backward equation(s) | TEMP | 
| --- | --- | --- | --- | --- |
| Dense | ✓ | $\mathbf{Z}_l = \mathbf{W}_l \cdot \mathbf{A}\_{l-1} + \mathbf{b}_l$ | --- | NO: check (1/m) term in backprop |
| ReLU | ✓ | --- | --- | --- |
| Softmax | ✓ | --- | --- | --- |
| BatchNorm | ✓ | --- | --- | --- |

## 2. Usage
1. Define architectures and train on MNIST dataset in `.../experiments/mnist_classification_train_and_validate.py`
  1. `DATA_CACHE_DIR`, `PLOTS_DIR` and `MODEL_CHECKPOINTS_DIR` define the directories for saving training/validation loop artefacts
  2. `RUN_SETTINGS` is a list of dictionaries. Each element of the list is a neural net architecture (allowing training and validation of several candidate architectures). Each dictionary defines both the neural net `"architecture"`, and training loop settings.
2. Post-process runs in `.../experiments/mnist_classification_evaluate.py`
  1. (Script has similar global parameters to set as `...train_and_validate`)
  2. Generates a summary log of metrics for all runs included
  3. Generates a visualisation of sample MNIST inferences for the trained model (on the validation set)
