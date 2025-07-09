import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import Tuple, Callable, Optional


def normalize_data(data: torch.Tensor, mean: Optional[torch.Tensor] = None, 
                   std: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize data to zero mean and unit variance.
    
    Args:
        data: Input tensor to normalize
        mean: Optional precomputed mean (if None, computed from data)
        std: Optional precomputed std (if None, computed from data)
    
    Returns:
        Tuple of (normalized_data, mean, std)
    """
    if mean is None:
        mean = data.mean()
    if std is None:
        std = data.std()
    
    # Avoid division by zero
    std = torch.clamp(std, min=1e-8)
    
    normalized = (data - mean) / std
    return normalized, mean, std


def denormalize_data(normalized_data: torch.Tensor, mean: torch.Tensor, 
                     std: torch.Tensor) -> torch.Tensor:
    """
    Denormalize data using provided mean and std.
    
    Args:
        normalized_data: Normalized tensor
        mean: Mean used for normalization
        std: Standard deviation used for normalization
    
    Returns:
        Denormalized tensor
    """
    return normalized_data * std + mean


def mean_pooling(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Apply mean pooling along specified dimension.
    
    Args:
        x: Input tensor
        dim: Dimension to pool over
    
    Returns:
        Mean-pooled tensor
    """
    return x.mean(dim=dim)


def masked_mean_pooling(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Apply mean pooling along specified dimension, excluding masked (padded) positions.
    
    Args:
        x: Input tensor (B, seq_len, dim)
        mask: Boolean mask tensor (B, seq_len) where True indicates real data
        dim: Dimension to pool over (default: 1, sequence dimension)
    
    Returns:
        Mean-pooled tensor excluding masked positions
    """
    if mask.dim() == 2 and x.dim() == 3:
        # Expand mask to match x dimensions: (B, seq_len) -> (B, seq_len, 1)
        mask = mask.unsqueeze(-1)
    
    # Set masked positions to 0 for summation
    masked_x = x * mask.float()
    
    # Sum over the specified dimension
    sum_x = masked_x.sum(dim=dim)
    
    # Count non-masked positions
    count = mask.float().sum(dim=dim)
    
    # Avoid division by zero
    count = torch.clamp(count, min=1e-8)
    
    # Compute mean
    return sum_x / count


def synthetic_set_pair_generator(batch_size: int = 32, n1_range: Tuple[int, int] = (2, 10), 
                                 n2_range: Tuple[int, int] = (2, 10), 
                                 dim_input: int = 1, 
                                 task: str = "t_statistic") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic paired sets for training with t-statistic targets.
    
    Args:
        batch_size: Number of samples in batch
        n1_range: Range for size of first set (default: (2, 10))
        n2_range: Range for size of second set (default: (2, 10))
        dim_input: Input dimension
        task: Task type - only "t_statistic" supported (computes Welch's t-test statistic)
    
    Returns:
        Tuple of (x_batch, y_batch, targets) where targets are t-statistics
    """
    x_batch = []
    y_batch = []
    targets = []
    
    for _ in range(batch_size):
        # Random set sizes
        n1 = np.random.randint(n1_range[0], n1_range[1] + 1)
        n2 = np.random.randint(n2_range[0], n2_range[1] + 1)
        
        # Generate random sets from uniform distribution [-10, 10]
        x = torch.rand(n1, dim_input) * 20.0 - 10.0  # Scale [0,1] to [-10,10]
        y = torch.rand(n2, dim_input) * 20.0 - 10.0  # Scale [0,1] to [-10,10]
        
        # Compute target based on task
        if task == "t_statistic":
            # Convert to numpy for scipy.stats
            x_np = x.squeeze().numpy() if dim_input == 1 else x.numpy()
            y_np = y.squeeze().numpy() if dim_input == 1 else y.numpy()
            
            # Handle edge cases
            if n1 == 1 and n2 == 1:
                # For single samples, use simple difference
                target = float(x_np - y_np) if dim_input == 1 else float((x_np - y_np).mean())
            elif n1 == 1 or n2 == 1:
                # For single sample vs multiple, use z-score like approach
                if n1 == 1:
                    target = float((x_np - y_np.mean()) / (y_np.std() + 1e-8))
                else:
                    target = float((x_np.mean() - y_np) / (x_np.std() + 1e-8))
            else:
                # Normal case: compute t-statistic using Welch's t-test
                try:
                    if dim_input == 1:
                        t_stat, _ = stats.ttest_ind(x_np, y_np, equal_var=False, nan_policy='propagate')
                    else:
                        # For multi-dimensional, use mean along feature dimension
                        x_means = x_np.mean(axis=1)
                        y_means = y_np.mean(axis=1)
                        t_stat, _ = stats.ttest_ind(x_means, y_means, equal_var=False, nan_policy='propagate')
                    
                    target = float(t_stat) if not np.isnan(t_stat) else 0.0
                except:
                    # Fallback to simple difference if t-test fails
                    target = float((x_np.mean() - y_np.mean()))
        else:
            raise ValueError(f"Unknown task: {task}. Only 't_statistic' is supported.")
        
        x_batch.append(x)
        y_batch.append(y)
        targets.append(target)
    
    return x_batch, y_batch, torch.tensor(targets, dtype=torch.float32)


def pad_sequences(sequences: list, max_length: Optional[int] = None, 
                   padding_value: float = -1e9) -> torch.Tensor:
    """
    Pad sequences to the same length with a configurable padding value.
    
    Args:
        sequences: List of tensors with different lengths
        max_length: Maximum length to pad to (if None, use longest sequence)
        padding_value: Value to use for padding (default: -1e9, avoids conflict with meaningful zeros)
    
    Returns:
        Padded tensor of shape (batch_size, max_length, dim)
    """
    if max_length is None:
        max_length = max(seq.size(0) for seq in sequences)
    
    batch_size = len(sequences)
    dim = sequences[0].size(-1)
    
    padded = torch.full((batch_size, max_length, dim), padding_value, dtype=sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        length = min(seq.size(0), max_length)
        padded[i, :length] = seq[:length]
    
    return padded


def create_padding_mask(sequences: list, max_length: Optional[int] = None) -> torch.Tensor:
    """
    Create padding mask for sequences.
    
    Args:
        sequences: List of tensors with different lengths
        max_length: Maximum length (if None, use longest sequence)
    
    Returns:
        Boolean mask tensor where True indicates real data, False indicates padding
    """
    if max_length is None:
        max_length = max(seq.size(0) for seq in sequences)
    
    batch_size = len(sequences)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    
    for i, seq in enumerate(sequences):
        length = min(seq.size(0), max_length)
        mask[i, :length] = True
    
    return mask


def batch_generator(x_data: list, y_data: list, targets: torch.Tensor, 
                    batch_size: int, shuffle: bool = True, padding_value: float = -1e9):
    """
    Generate batches from data with padding masks.
    
    Args:
        x_data: List of x tensors
        y_data: List of y tensors  
        targets: Target values
        batch_size: Size of each batch
        shuffle: Whether to shuffle data
        padding_value: Value to use for padding (default: -1e9)
    
    Yields:
        Batches of (x_batch, y_batch, x_mask, y_mask, target_batch)
    """
    n_samples = len(x_data)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        
        x_batch = [x_data[idx] for idx in batch_indices]
        y_batch = [y_data[idx] for idx in batch_indices]
        target_batch = targets[batch_indices]
        
        # Pad sequences
        x_padded = pad_sequences(x_batch, padding_value=padding_value)
        y_padded = pad_sequences(y_batch, padding_value=padding_value)
        
        # Create padding masks
        x_mask = create_padding_mask(x_batch)
        y_mask = create_padding_mask(y_batch)
        
        yield x_padded, y_padded, x_mask, y_mask, target_batch


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Root Mean Square Error.
    
    Args:
        predictions: Predicted values
        targets: True target values
    
    Returns:
        RMSE value
    """
    mse = torch.mean((predictions - targets) ** 2)
    return torch.sqrt(mse).item()


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        predictions: Predicted values
        targets: True target values
    
    Returns:
        MAE value
    """
    mae = torch.mean(torch.abs(predictions - targets))
    return mae.item()


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in monitored quantity to qualify as improvement
            restore_best_weights: Whether to restore model weights from the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights for
        
        Returns:
            True if training should be stopped, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False