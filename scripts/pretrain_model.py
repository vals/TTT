#!/usr/bin/env python3
"""
TTT (T-Test Transformer) Model Pretraining Script

This script trains a T-Test Transformer model on synthetic data to learn
t-statistic computation from pairs of small sets. The trained model is saved
as a pretrained model that can be used directly via `from ttt import ttt`.

The script:
1. Creates a PairSetTransformer model with default configuration
2. Trains it on synthetic set pairs with t-statistic targets
3. Evaluates the trained model on test data
4. Saves the model for package distribution (ttt/pretrained_ttt_model.pth)
"""

import torch
import numpy as np
import argparse
from ttt.model import PairSetTransformer
from ttt.train import train_model, evaluate_model
from ttt.utils import synthetic_set_pair_generator


def main():
    """
    Train a T-Test Transformer model and save it as the pretrained model.
    
    This function creates, trains, and evaluates a T-Test Transformer model
    on synthetic data. The trained model is saved both as a standalone file
    and as the package's pretrained model for immediate use.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train T-Test Transformer model')
    parser.add_argument('--epochs', type=int, default=10_000,
                        help='Number of epochs to train (default: 10,000)')
    args = parser.parse_args()
    
    print("TTT (T-Test Transformer) - Model Pretraining")
    print("=" * 40)
    print(f"Training for {args.epochs} epochs")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Model configuration
    dim_input = 1
    d_model = 128
    n_heads = 8
    num_self_layers = 3
    num_cross_layers = 3
    dropout = 0.1
    
    # Create model
    print("Creating PairSetTransformer model...")
    model = PairSetTransformer(
        dim_input=dim_input,
        d_model=d_model,
        n_heads=n_heads,
        num_self_layers=num_self_layers,
        num_cross_layers=num_cross_layers,
        dropout=dropout
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training configuration
    training_config = {
        'lr': 3e-4,
        'batch_size': 64,
        'epochs': args.epochs,  # Use command line argument
        'weight_decay': 1e-2,
        'gradient_clip': 1.0,
        'val_frequency': 100,
        'early_stopping_patience': 3,
        'device': 'cpu',
        'padding_value': -1e9,  # Use large negative value for padding (avoids conflict with meaningful zeros)
        'tensorboard_log_dir': './runs',  # Enable TensorBoard logging
        'experiment_name': 'basic_ttt_demo'
    }
    
    # Data generation configuration
    data_config = {
        'n1_range': (2, 10),
        'n2_range': (2, 10),
        'dim_input': dim_input,
        'task': 't_statistic'
    }
    
    print("\nTraining configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    
    print("\nData configuration:")
    for key, value in data_config.items():
        print(f"  {key}: {value}")
    
    # Train the model
    print("\nStarting training...")
    history = train_model(
        model=model,
        data_generator=synthetic_set_pair_generator,
        **training_config,
        **data_config
    )
    
    print(f"Training completed! Final validation RMSE: {history['val_rmse'][-1]:.6f}")
    
    # Save the trained model
    model_path = 'trained_ttt_model.pth'
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    # Also save as pretrained model for the package
    import os
    # Get the directory containing this script, then go up one level to find ttt/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(script_dir)  # Go up from scripts/ to parent directory
    package_model_path = os.path.join(package_dir, 'ttt', 'pretrained_ttt_model.pth')
    
    model.save_model(package_model_path)
    print(f"Model also saved as pretrained model to: {package_model_path}")
    print("This model will be automatically loaded when importing 'ttt'")
    
    # Generate test data
    print("\nGenerating test data...")
    test_x, test_y, test_targets = synthetic_set_pair_generator(
        batch_size=1000, **data_config
    )
    
    # Evaluate the model
    print("Evaluating model on test data...")
    test_results = evaluate_model(
        model=model,
        test_x=test_x,
        test_y=test_y,
        test_targets=test_targets,
        batch_size=256,
        device='cpu'
    )
    
    print("\nTest Results:")
    print(f"  RMSE: {test_results['rmse']:.6f}")
    print(f"  MAE:  {test_results['mae']:.6f}")
    print(f"  R²:   {test_results['r2']:.6f}")
    
    # Show a few predictions vs targets
    print("\nSample predictions vs targets:")
    predictions = test_results['predictions'][:10]
    targets = test_results['targets'][:10]
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        print(f"  Sample {i+1}: Pred={pred:.4f}, Target={target:.4f}, "
              f"Error={abs(pred-target):.4f}")
    
    print("\nExample completed successfully!")
    print("\nTo view TensorBoard logs, run:")
    print("  tensorboard --logdir=./runs")
    print("Then open http://localhost:6006 in your browser")


if __name__ == "__main__":
    main()