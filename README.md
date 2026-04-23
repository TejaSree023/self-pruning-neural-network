# Self-Pruning Neural Network

## Overview

This project implements a neural network that learns to **prune its own weights during training** using gate-based sparsity.

## Key Idea

Each weight is multiplied by a learnable gate:

effective_weight = weight × sigmoid(gate_score)

If the gate approaches zero, the weight is effectively removed.

## Loss Function

Total Loss = CrossEntropyLoss + λ × SparsityLoss

* SparsityLoss = L1-style sum of gate values
* λ controls pruning strength

## Results

| Lambda | Test Accuracy | Sparsity (%) |
| ------ | ------------- | ------------ |
| 1e-4   | 0.49          | 9.95         |
| 5e-4   | 0.48          | 9.96         |
| 1e-3   | 0.46          | 10.00        |

## Insights

* Higher λ → more pruning, slightly lower accuracy
* Strict threshold (1e-2) leads to gradual sparsity changes

## Tech Stack

* Python
* PyTorch

## How to Run

```bash
python main.py
```

## Output

* Accuracy vs sparsity comparison
* Gate distribution plot

## Author

Teja Sree
