# TTT: T-Test Transformer Package

A PyTorch-based Python package that provides a pretrained neural network for computing t-statistics from pairs of small sets. The T-Test Transformer uses attention mechanisms to process variable-length sets and outputs statistical comparisons.

## Installation

This package can be installed from source by cloning the repository:

```bash
git clone https://github.com/vals/ttt.git
cd ttt
pip install -e .
```

**Dependencies:** PyTorch (>=1.10), numpy, scipy

## Quick Start

```python
from ttt import ttt

# Compute t-statistic for two sets
set_x = [1.2, 3.4, 2.1, 5.6]
set_y = [2.3, 1.9, 4.2]

# Get predicted t-statistic
predicted_t = ttt.t_statistic(set_x, set_y)
print(f"Predicted t-statistic: {predicted_t:.4f}")

# Compare with scipy
from scipy import stats
actual_t, _ = stats.ttest_ind(set_x, set_y, equal_var=False)
print(f"Actual t-statistic: {actual_t:.4f}")
```

## Usage

The package provides a pretrained model that can be used immediately:

```python
from ttt import ttt

# Works with Python lists
result = ttt.t_statistic([1, 2, 3, 4], [2, 3, 4, 5])

# Works with different set sizes
result = ttt.t_statistic([1.5, 2.1], [3.2, 4.1, 5.3, 2.8])

# Handles edge cases (small sets)
result = ttt.t_statistic([1.0], [2.0, 3.0])
```

## What is a t-statistic?

The t-statistic is a measure of how different two groups are, taking into account both the difference in means and the variability within each group. It's commonly used in statistical hypothesis testing.

The TTT (T-Test Transformer) package provides a neural network that has learned to compute t-statistics from pairs of small sets, offering a fast alternative to traditional statistical computation.

## Features

- **Zero Setup**: Pretrained model ready to use immediately
- **Variable Set Sizes**: Handles sets of different lengths (2-10 elements)
- **Fast Computation**: Neural network inference faster than traditional methods
- **Robust**: Handles edge cases and small sample sizes
- **Simple API**: Single method `t_statistic(set_x, set_y)`

## Performance

The model is trained on synthetic data with uniform distributions and learns to approximate Welch's t-test (unequal variance assumption). It performs well on:

- Small sets (2-10 elements)
- Various distributions
- Edge cases (single elements, identical sets)

## License

MIT License