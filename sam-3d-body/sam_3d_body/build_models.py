# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch

# Patch torch.amp for PyTorch < 2.2.0 compatibility (needed for DINOv3)
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

from .models.meta_arch import SAM3DBody
from .utils.config import get_config
from .utils.checkpoint import load_state_dict


def load_sam_3d_body(checkpoint_path: str = "", device: str = "cuda", mhr_path: str = ""):
    print("Loading SAM 3D Body model...")
    
    # Check the current directory, and if not present check the parent dir.
    model_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if not os.path.exists(model_cfg):
        # Looks at parent dir
        model_cfg = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
        )

    model_cfg = get_config(model_cfg)

    # Disable face for inference
    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    model_cfg.freeze()

    # Initialze the model
    model = SAM3DBody(model_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    load_state_dict(model, state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model, model_cfg


def _hf_download(repo_id):
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=repo_id)
    return os.path.join(local_dir, "model.ckpt"), os.path.join(local_dir, "assets", "mhr_model.pt")


def load_sam_3d_body_hf(repo_id, **kwargs):
    ckpt_path, mhr_path = _hf_download(repo_id)
    return load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path, **kwargs)
