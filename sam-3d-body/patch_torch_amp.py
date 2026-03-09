"""
Patch torch.amp to add custom_fwd and custom_bwd for PyTorch < 2.2.0 compatibility.
This should be imported before loading DINOv3 models.
"""
import torch

# Check if custom_fwd and custom_bwd are missing
if not hasattr(torch.amp, 'custom_fwd'):
    # Add custom_fwd and custom_bwd as no-op decorators for compatibility
    def custom_fwd(fn):
        """No-op decorator for PyTorch < 2.2.0 compatibility"""
        return fn
    
    def custom_bwd(fn):
        """No-op decorator for PyTorch < 2.2.0 compatibility"""
        return fn
    
    # Monkey-patch torch.amp
    torch.amp.custom_fwd = custom_fwd
    torch.amp.custom_bwd = custom_bwd
    
    print("Patched torch.amp with custom_fwd and custom_bwd for compatibility")

