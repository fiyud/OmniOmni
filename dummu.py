import torch
import torch.nn as nn
from yolo_distiller.ultralytics.nn.modules.conv import DWConv

class LayerNorm_s(nn.Module):
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class InvertedBottleneck(nn.Module):
    """Inverted Bottleneck with expanded dimension for better feature extraction"""
    def __init__(self, dim, expansion_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        return self.net(x)

class SpatialMixer(nn.Module):
    """Spatial mixing block with depthwise convolution and normalization"""
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.spatial_mix = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim, bias=False),
            nn.InstanceNorm2d(dim, affine=True)  # Thay vì LayerNorm_s
        )

    
    def forward(self, x):
        return self.spatial_mix(x)

class AdaptiveInput(nn.Module):
    """Adaptive input projection with dimension validation"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj = nn.Conv2d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x):
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} channels, got {x.size(1)}")
        return self.proj(x)

class ResidualScale(nn.Module):
    """Residual connection with learnable scaling"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x, residual):
        if self.scale is not None:
            residual = residual * self.scale
        return x + self.drop_path(residual)

class ConvNextBlockV2(nn.Module):
    """Improved ConvNext block with modular architecture"""
    def __init__(self, 
                 input_dim, 
                 dim, 
                 kernel_size=7,
                 expansion_ratio=4,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 dropout=0.1):
        super().__init__()
        
        # Modular components
        self.input_adapter = AdaptiveInput(input_dim, dim)
        self.spatial_mixer = SpatialMixer(dim, kernel_size)
        self.channel_mixer = InvertedBottleneck(dim, expansion_ratio, dropout)
        self.residual_scale = ResidualScale(dim, drop_path, layer_scale_init_value)
        
        # Optional: Add attention mechanism
        self.attention = None  # Can be extended with self-attention if needed
        
    def forward(self, x):
        identity = self.input_adapter(x)
        
        # Spatial mixing
        x = self.spatial_mixer(identity)
        
        # Channel mixing
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.channel_mixer(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        # Optional attention
        if self.attention is not None:
            x = self.attention(x)
        
        # Residual connection
        out = self.residual_scale(identity, x)
        
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_convnext_block():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model instance
    input_dim = 64  # Input channels
    dim = 128       # Output channels
    block = ConvNextBlockV2(
        input_dim=input_dim,
        dim=dim,
        drop_path=0.1,
        layer_scale_init_value=1e-6,
        kernel_size=7
    )
    
    # Create dummy input: [batch_size, channels, height, width]
    x = torch.randn(2, input_dim, 32, 32)
    
    # Print input shape and statistics
    print(f"\nInput Tensor:")
    print(f"Shape: {x.shape}")
    print(f"Mean: {x.mean():.4f}")
    print(f"Std: {x.std():.4f}")
    
    # Forward pass
    try:
        # Set to eval mode for consistent results
        block.eval()
        with torch.no_grad():
            output = block(x)
            
            # Print output shape and statistics
            print(f"\nOutput Tensor:")
            print(f"Shape: {output.shape}")
            print(f"Mean: {output.mean():.4f}")
            print(f"Std: {output.std():.4f}")
            
            # Print intermediate tensor shapes
            print("\nIntermediate tensor shapes:")
            x_temp = block.dwconv(x)
            print(f"After dwconv: {x_temp.shape}")
            
            x_temp = x_temp.permute(0, 2, 3, 1)
            print(f"After first permute: {x_temp.shape}")
            
            x_temp = block.norm(x_temp)
            print(f"After norm: {x_temp.shape}")
            
            x_temp = block.pwconv1(x_temp)
            print(f"After pwconv1: {x_temp.shape}")
            
            return output
            
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        raise

# Function to test different input sizes
def test_different_sizes():
    test_sizes = [
        (1, 64, 16, 16),   # Small
        (2, 64, 32, 32),   # Medium
        (4, 64, 64, 64),   # Large
    ]
    
    for batch, channels, height, width in test_sizes:
        print(f"\nTesting input size: [{batch}, {channels}, {height}, {width}]")
        x = torch.randn(batch, channels, height, width)
        block = ConvNextBlockV2(input_dim=channels, dim=128)
        
        with torch.no_grad():
            output = block(x)
            print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    print("Testing single forward pass...")
    output = test_convnext_block()
    
    print("\nTesting different input sizes...")
    test_different_sizes()