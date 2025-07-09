# Pretrained Model Information

## Location
The pretrained TTT model should be placed at:
```
ttt/pretrained_ttt_model.pth
```

## How to Generate a Pretrained Model

1. **Run the model pretraining script:**
   ```bash
   python scripts/pretrain_model.py
   ```
   This will automatically save a trained model to `ttt/pretrained_ttt_model.pth`

2. **Train your own model:**
   ```python
   from ttt import PairSetTransformer, train_model
   
   # Create and train model
   model = PairSetTransformer(dim_input=1, d_model=128, n_heads=8)
   train_model(model, ...)
   
   # Save as pretrained model
   model.save_model('ttt/pretrained_ttt_model.pth')
   ```

## Usage
Once the pretrained model is in place, users can import and use it directly:

```python
from ttt import ttt

# Use the pretrained model
if ttt is not None:
    result = ttt.t_statistic([1.2, 3.4, 2.1], [2.3, 1.9, 4.2])
    print(f"Predicted t-statistic: {result}")
else:
    print("No pretrained model available")
```

## File Size
The pretrained model file should be approximately 4-5 MB in size.

## Distribution
The model file will be automatically included in pip distributions via the MANIFEST.in file.