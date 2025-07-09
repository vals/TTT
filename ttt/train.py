import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from typing import Callable, Optional, Dict, List, Tuple
import time
import os
from datetime import datetime
from .utils import (
    synthetic_set_pair_generator, batch_generator, compute_rmse, compute_mae,
    EarlyStopping, normalize_data, denormalize_data
)


def create_prediction_scatter_plot(predictions, targets, title="Predictions vs Targets"):
    """Create a scatter plot of predictions vs targets for TensorBoard."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(targets, predictions, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
    
    ax.set_xlabel('True T-Statistics')
    ax.set_ylabel('Predicted T-Statistics')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    
    return np.array(image).transpose(2, 0, 1)  # Convert to CHW format for TensorBoard


def train_model(model: nn.Module, 
                data_generator: Callable = synthetic_set_pair_generator,
                val_data: Optional[Tuple] = None,
                lr: float = 3e-4,
                min_lr: float = 1e-5,
                batch_size: int = 64,
                epochs: int = 15,
                max_steps: Optional[int] = None,
                weight_decay: float = 1e-2,
                gradient_clip: float = 1.0,
                val_batch_size: int = 256,
                val_frequency: int = 500,
                early_stopping_patience: int = 5,
                device: str = "cpu",
                verbose: bool = True,
                padding_value: float = -1e9,
                tensorboard_log_dir: Optional[str] = None,
                experiment_name: str = "ttt_experiment",
                **data_gen_kwargs) -> Dict:
    """
    Train a PairSetTransformer model with CPU optimization.
    
    Args:
        model: PairSetTransformer model to train
        data_generator: Function to generate training data
        val_data: Optional validation data tuple (x_val, y_val, targets_val)
        lr: Initial learning rate
        min_lr: Minimum learning rate for cosine annealing
        batch_size: Training batch size
        epochs: Number of training epochs
        max_steps: Maximum number of training steps (overrides epochs if provided)
        weight_decay: Weight decay for AdamW optimizer
        gradient_clip: Gradient clipping value
        val_batch_size: Validation batch size
        val_frequency: Frequency of validation (in steps)
        early_stopping_patience: Early stopping patience
        device: Device to train on
        verbose: Whether to print progress
        padding_value: Value to use for padding sequences (default: -1e9)
        tensorboard_log_dir: Directory for TensorBoard logs (if None, no logging)
        experiment_name: Name for this experiment run
        **data_gen_kwargs: Additional kwargs for data generator
    
    Returns:
        Training history dictionary
    """
    # Set up CPU optimization
    if device == "cpu":
        torch.set_num_threads(torch.get_num_threads())
        
    model = model.to(device)
    
    # Set up TensorBoard logging
    writer = None
    if tensorboard_log_dir is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(tensorboard_log_dir, f"{experiment_name}_{timestamp}")
        writer = SummaryWriter(log_dir)
        if verbose:
            print(f"TensorBoard logging to: {log_dir}")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Generate validation data if not provided
    if val_data is None:
        if verbose:
            print("Generating validation data...")
        val_x, val_y, val_targets = data_generator(
            batch_size=5000, **data_gen_kwargs
        )
        val_data = (val_x, val_y, val_targets)
    
    val_x, val_y, val_targets = val_data
    
    # Normalize validation targets
    val_targets_norm, val_mean, val_std = normalize_data(val_targets)
    
    # Calculate total steps
    if max_steps is None:
        steps_per_epoch = 1000  # Default steps per epoch
        total_steps = epochs * steps_per_epoch
    else:
        total_steps = max_steps
        epochs = max_steps // 1000 + 1
    
    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'learning_rates': [],
        'epoch': [],
        'step': []
    }
    
    # Log hyperparameters to TensorBoard
    if writer is not None:
        hparams = {
            'lr': lr,
            'min_lr': min_lr,
            'batch_size': batch_size,
            'epochs': epochs,
            'weight_decay': weight_decay,
            'gradient_clip': gradient_clip,
            'd_model': model.d_model,
            'n_heads': model.n_heads,
            'num_self_layers': model.num_self_layers,
            'num_cross_layers': model.num_cross_layers,
        }
        writer.add_hparams(hparams, {})
        
        # Log model architecture as text
        model_summary = str(model)
        writer.add_text("Model Architecture", model_summary, 0)
    
    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    
    if verbose:
        print(f"Starting training for {epochs} epochs ({total_steps} steps)")
        print(f"Batch size: {batch_size}, Learning rate: {lr}")
        print(f"Device: {device}")
        if writer is not None:
            print(f"TensorBoard logging enabled")
        print("-" * 50)
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Generate training data for this epoch
        train_x, train_y, train_targets = data_generator(
            batch_size=batch_size * 20, **data_gen_kwargs  # Generate more data per epoch
        )
        
        # Normalize training targets
        train_targets_norm, train_mean, train_std = normalize_data(train_targets)
        
        # Create batches
        for batch_x, batch_y, x_mask, y_mask, batch_targets in batch_generator(
            train_x, train_y, train_targets_norm, batch_size, shuffle=True, padding_value=padding_value
        ):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            x_mask = x_mask.to(device)
            y_mask = y_mask.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_x, batch_y, x_mask, y_mask)
            loss = nn.MSELoss()(predictions, batch_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            step += 1
            
            # Log training metrics to TensorBoard
            if writer is not None:
                writer.add_scalar('Loss/Train_Step', loss.item(), step)
                writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], step)
                
                # Log gradient norms periodically
                if step % 100 == 0:
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    writer.add_scalar('Gradients/Total_Norm', total_norm, step)
            
            # Validation
            if step % val_frequency == 0:
                val_loss, val_rmse, val_mae = validate_model(
                    model, val_x, val_y, val_targets_norm, val_mean, val_std,
                    val_batch_size, device, padding_value
                )
                
                # Store history
                train_loss_avg = np.mean(epoch_losses[-50:])  # Moving average
                history['train_loss'].append(train_loss_avg)
                history['val_loss'].append(val_loss)
                history['val_rmse'].append(val_rmse)
                history['val_mae'].append(val_mae)
                history['learning_rates'].append(scheduler.get_last_lr()[0])
                history['epoch'].append(epoch)
                history['step'].append(step)
                
                # Log validation metrics to TensorBoard
                if writer is not None:
                    writer.add_scalar('Loss/Train_Epoch', train_loss_avg, step)
                    writer.add_scalar('Loss/Validation', val_loss, step)
                    writer.add_scalar('Metrics/Val_RMSE', val_rmse, step)
                    writer.add_scalar('Metrics/Val_MAE', val_mae, step)
                    
                    # Log model parameters histograms every few validations
                    if step % (val_frequency * 5) == 0:
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                writer.add_histogram(f'Parameters/{name}', param.data, step)
                                if param.grad is not None:
                                    writer.add_histogram(f'Gradients/{name}', param.grad.data, step)
                
                if verbose:
                    elapsed_time = time.time() - start_time
                    print(f"Epoch {epoch+1:3d} | Step {step:6d} | "
                          f"Train Loss: {history['train_loss'][-1]:.6f} | "
                          f"Val Loss: {val_loss:.6f} | "
                          f"Val RMSE: {val_rmse:.6f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                          f"Time: {elapsed_time:.1f}s")
                
                # Early stopping check
                if early_stopping(val_loss, model):
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}, step {step}")
                    break
                
                model.train()  # Back to training mode
            
            if step >= total_steps:
                break
        
        if step >= total_steps:
            break
    
    # Final validation with prediction visualization
    final_val_loss, final_val_rmse, final_val_mae, final_predictions, final_targets = validate_model(
        model, val_x, val_y, val_targets_norm, val_mean, val_std,
        val_batch_size, device, padding_value, return_predictions=True
    )
    
    # Log final metrics to TensorBoard
    if writer is not None:
        writer.add_scalar('Final/Validation_Loss', final_val_loss, step)
        writer.add_scalar('Final/Validation_RMSE', final_val_rmse, step)
        writer.add_scalar('Final/Validation_MAE', final_val_mae, step)
        
        # Add final prediction scatter plot
        if final_predictions is not None and final_targets is not None:
            plot_image = create_prediction_scatter_plot(
                final_predictions, final_targets, 
                "Final Validation: Predictions vs Targets"
            )
            writer.add_image('Predictions/Final_Validation', plot_image, step)
        
        writer.close()
    
    if verbose:
        total_time = time.time() - start_time
        print("-" * 50)
        print(f"Training completed in {total_time:.1f}s")
        print(f"Final validation - Loss: {final_val_loss:.6f}, "
              f"RMSE: {final_val_rmse:.6f}, MAE: {final_val_mae:.6f}")
        if writer is not None:
            print(f"TensorBoard logs saved to: {log_dir}")
    
    return history


def validate_model(model: nn.Module, 
                   val_x: List, 
                   val_y: List, 
                   val_targets: torch.Tensor,
                   target_mean: torch.Tensor,
                   target_std: torch.Tensor,
                   batch_size: int,
                   device: str,
                   padding_value: float = -1e9,
                   return_predictions: bool = False) -> Tuple[float, float, float, ...]:
    """
    Validate the model on validation data.
    
    Args:
        model: Model to validate
        val_x: Validation x data
        val_y: Validation y data
        val_targets: Validation targets (normalized)
        target_mean: Target normalization mean
        target_std: Target normalization std
        batch_size: Validation batch size
        device: Device to run validation on
    
    Returns:
        Tuple of (val_loss, val_rmse, val_mae) or (val_loss, val_rmse, val_mae, predictions, targets) if return_predictions=True
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y, x_mask, y_mask, batch_targets in batch_generator(
            val_x, val_y, val_targets, batch_size, shuffle=False, padding_value=padding_value
        ):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            x_mask = x_mask.to(device)
            y_mask = y_mask.to(device)
            batch_targets = batch_targets.to(device)
            
            predictions = model(batch_x, batch_y, x_mask, y_mask)
            loss = nn.MSELoss()(predictions, batch_targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_targets.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Denormalize for metric calculation
    all_predictions_denorm = denormalize_data(all_predictions, target_mean, target_std)
    all_targets_denorm = denormalize_data(all_targets, target_mean, target_std)
    
    val_loss = total_loss / num_batches
    val_rmse = compute_rmse(all_predictions_denorm, all_targets_denorm)
    val_mae = compute_mae(all_predictions_denorm, all_targets_denorm)
    
    if return_predictions:
        return val_loss, val_rmse, val_mae, all_predictions_denorm, all_targets_denorm
    else:
        return val_loss, val_rmse, val_mae


def evaluate_model(model: nn.Module,
                   test_x: List,
                   test_y: List,
                   test_targets: torch.Tensor,
                   batch_size: int = 256,
                   device: str = "cpu",
                   padding_value: float = -1e9) -> Dict:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        test_x: Test x data
        test_y: Test y data
        test_targets: Test targets
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y, x_mask, y_mask, batch_targets in batch_generator(
            test_x, test_y, test_targets, batch_size, shuffle=False, padding_value=padding_value
        ):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            x_mask = x_mask.to(device)
            y_mask = y_mask.to(device)
            
            predictions = model(batch_x, batch_y, x_mask, y_mask)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_targets.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    rmse = compute_rmse(all_predictions, all_targets)
    mae = compute_mae(all_predictions, all_targets)
    mse = torch.mean((all_predictions - all_targets) ** 2).item()
    
    # Compute R²
    ss_res = torch.sum((all_targets - all_predictions) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'r2': r2.item(),
        'predictions': all_predictions,
        'targets': all_targets
    }