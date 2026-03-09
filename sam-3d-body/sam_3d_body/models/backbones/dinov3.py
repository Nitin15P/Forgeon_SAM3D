# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch import nn

# Patch torch.amp for PyTorch < 2.2.0 compatibility
if not hasattr(torch.amp, 'custom_fwd'):
    def custom_fwd(fn=None, **kwargs):
        """No-op decorator for PyTorch < 2.2.0 compatibility"""
        if fn is None:
            # Called with kwargs: @custom_fwd(device_type='cpu')
            def decorator(func):
                return func
            return decorator
        # Called as @custom_fwd
        return fn
    
    def custom_bwd(fn=None, **kwargs):
        """No-op decorator for PyTorch < 2.2.0 compatibility"""
        if fn is None:
            # Called with kwargs: @custom_bwd(device_type='cpu')
            def decorator(func):
                return func
            return decorator
        # Called as @custom_bwd
        return fn
    
    torch.amp.custom_fwd = custom_fwd
    torch.amp.custom_bwd = custom_bwd

# Patch torch._dynamo.config for PyTorch < 2.2.0 compatibility
# DINOv3 code tries to set accumulated_cache_size_limit which doesn't exist in PyTorch 2.1.2
if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
    if not hasattr(torch._dynamo.config, 'accumulated_cache_size_limit'):
        # Monkey-patch __setattr__ to allow setting this attribute
        _original_setattr = torch._dynamo.config.__class__.__setattr__
        def _patched_setattr(self, name, value):
            if name == 'accumulated_cache_size_limit':
                # Allow setting this specific attribute
                object.__setattr__(self, name, value)
            else:
                # Use original behavior for other attributes
                _original_setattr(self, name, value)
        torch._dynamo.config.__class__.__setattr__ = _patched_setattr


class Dinov3Backbone(nn.Module):
    def __init__(
        self, name="dinov2_vitb14", pretrained_weight=None, cfg=None, *args, **kwargs
    ):
        super().__init__()
        self.name = name
        self.cfg = cfg

        self.encoder = torch.hub.load(
            "facebookresearch/dinov3",
            self.name,
            source="github",
            pretrained=False,
            drop_path=self.cfg.MODEL.BACKBONE.DROP_PATH_RATE,
        )
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.embed_dims = self.encoder.embed_dim

    def forward(self, x, extra_embed=None):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert extra_embed is None, "Not Implemented Yet"

        y = self.encoder.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]

        return y

    def get_layer_depth(self, param_name: str, prefix: str = "encoder."):
        """Get the layer-wise depth of a parameter.
        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.
        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.encoder.n_blocks + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix) :]

        if param_name in ("cls_token", "pos_embed", "storage_tokens"):
            layer_depth = 0
        elif param_name.startswith("patch_embed"):
            layer_depth = 0
        elif param_name.startswith("blocks"):
            layer_id = int(param_name.split(".")[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers
