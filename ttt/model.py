import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import masked_mean_pooling


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            # Expand mask for multi-head attention: (B, seq_len) -> (B, 1, 1, seq_len)
            # This broadcasts to (B, n_heads, seq_len, seq_len) for attention scores
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attended)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, mask=None):
        # Cross-attention with residual connection
        attn_output = self.cross_attention(query, key_value, key_value, mask)
        x = self.norm1(query + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class PairSetTransformer(nn.Module):
    def __init__(self, dim_input, d_model=128, n_heads=8, num_self_layers=3, 
                 num_cross_layers=3, dropout=0.1):
        super().__init__()
        
        self.dim_input = dim_input
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_self_layers = num_self_layers
        self.num_cross_layers = num_cross_layers
        
        # Embedding layers
        self.embed_x = nn.Linear(dim_input, d_model)
        self.embed_y = nn.Linear(dim_input, d_model)
        
        # Intra-set self-attention layers
        self.self_layers_x = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) 
            for _ in range(num_self_layers)
        ])
        self.self_layers_y = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) 
            for _ in range(num_self_layers)
        ])
        
        # Cross-set attention layers
        self.cross_layers_x = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout) 
            for _ in range(num_cross_layers)
        ])
        self.cross_layers_y = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout) 
            for _ in range(num_cross_layers)
        ])
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y, x_mask=None, y_mask=None):
        # x: (B, n1, dim_input)
        # y: (B, n2, dim_input)
        # x_mask: (B, n1) boolean mask for x (True = real data, False = padding)
        # y_mask: (B, n2) boolean mask for y (True = real data, False = padding)
        
        # Embedding
        x_emb = self.dropout(self.embed_x(x))  # (B, n1, d_model)
        y_emb = self.dropout(self.embed_y(y))  # (B, n2, d_model)
        
        # Create attention masks (invert for attention - True = attend, False = ignore)
        x_attn_mask = x_mask if x_mask is not None else None
        y_attn_mask = y_mask if y_mask is not None else None
        
        # Intra-set self-attention
        for layer in self.self_layers_x:
            x_emb = layer(x_emb, x_attn_mask)
            
        for layer in self.self_layers_y:
            y_emb = layer(y_emb, y_attn_mask)
            
        # Cross-set attention
        for cross_x, cross_y in zip(self.cross_layers_x, self.cross_layers_y):
            x_cross = cross_x(x_emb, y_emb, y_attn_mask)  # X attending to Y
            y_cross = cross_y(y_emb, x_emb, x_attn_mask)  # Y attending to X
            x_emb = x_cross
            y_emb = y_cross
            
        # Masked mean pooling over sets
        if x_mask is not None:
            phi_x = masked_mean_pooling(x_emb, x_mask, dim=1)  # (B, d_model)
        else:
            phi_x = x_emb.mean(dim=1)  # (B, d_model)
            
        if y_mask is not None:
            phi_y = masked_mean_pooling(y_emb, y_mask, dim=1)  # (B, d_model)
        else:
            phi_y = y_emb.mean(dim=1)  # (B, d_model)
        
        # Combine features: [φ(X), φ(Y), φ(X)−φ(Y), φ(X)⊙φ(Y)]
        diff = phi_x - phi_y
        prod = phi_x * phi_y
        combined = torch.cat([phi_x, phi_y, diff, prod], dim=1)  # (B, 4*d_model)
        
        # Final regression output
        output = self.head(combined)  # (B, 1)
        
        return output.squeeze(-1)  # (B,)
    
    def predict(self, set_x, set_y, padding_value=-1e9):
        """
        Simple prediction interface for two sets (e.g., Python lists).
        
        Args:
            set_x: First set as Python list or 1D array-like
            set_y: Second set as Python list or 1D array-like
            padding_value: Value to use for padding (default: -1e9)
        
        Returns:
            Predicted t-statistic as a float
        """
        from .utils import pad_sequences, create_padding_mask
        
        # Optimize for CPU inference
        if not torch.cuda.is_available():
            torch.set_num_threads(torch.get_num_threads())
        
        # Convert inputs to tensors if needed
        if not isinstance(set_x, torch.Tensor):
            set_x = torch.tensor(set_x, dtype=torch.float32)
        if not isinstance(set_y, torch.Tensor):
            set_y = torch.tensor(set_y, dtype=torch.float32)
        
        # Ensure proper shape: (n,) -> (n, 1)
        if set_x.dim() == 1:
            set_x = set_x.unsqueeze(-1)
        if set_y.dim() == 1:
            set_y = set_y.unsqueeze(-1)
        
        # Create batch of size 1
        x_batch = [set_x]
        y_batch = [set_y]
        
        # Pad sequences and create masks
        x_padded = pad_sequences(x_batch, padding_value=padding_value)
        y_padded = pad_sequences(y_batch, padding_value=padding_value)
        x_mask = create_padding_mask(x_batch)
        y_mask = create_padding_mask(y_batch)
        
        # Set model to evaluation mode
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            prediction = self.forward(x_padded, y_padded, x_mask, y_mask)
        
        # Return single float value
        return prediction.item()
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'dim_input': self.dim_input,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'num_self_layers': self.num_self_layers,
                'num_cross_layers': self.num_cross_layers
            }
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded PairSetTransformer model
        """
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Create model with saved configuration
        model = cls(**checkpoint['model_config'])
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def t_statistic(self, set_x, set_y, padding_value=-1e9):
        """
        Alias for predict() method - compute predicted t-statistic for two sets.
        
        Args:
            set_x: First set as Python list or 1D array-like
            set_y: Second set as Python list or 1D array-like
            padding_value: Value to use for padding (default: -1e9)
        
        Returns:
            Predicted t-statistic as a float
        """
        return self.predict(set_x, set_y, padding_value)