# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

# Captum Imports
from captum.attr import IntegratedGradients

def compute_integrated_gradients(model, query_tensor, neighbor_tensor, baseline=None, steps=50):
    """
    Compute IG attributions using L2 distance (consistent with retrieval).
    """
    if baseline is None:
        baseline = 0 * neighbor_tensor  # Black image baseline
    
    neighbor_tensor.requires_grad_()
    
    def similarity_fn(neighbor_input):
        with torch.no_grad():
            query_embed = model(query_tensor)
        neighbor_embed = model(neighbor_input)
        return -torch.norm(query_embed - neighbor_embed, p=2).unsqueeze(0)  # Negative for maximization
    
    ig = IntegratedGradients(similarity_fn)
    
    # Stable attribution computation
    attributions = ig.attribute(
        inputs=neighbor_tensor,
        baselines=baseline,
        n_steps=steps,
        internal_batch_size=1,
        return_convergence_delta=False  
    )
    
    return attributions

def process_ig_to_heatmap(attributions):
    """
    Converts IG attributions to a normalized heatmap for visualization.
    """
    attr_np = attributions.detach().cpu().squeeze().numpy()
    if attr_np.ndim == 3:
        attr_np = np.mean(attr_np, axis=0)
    attr_np = (attr_np - attr_np.mean()) / (attr_np.std() + 1e-8)
    attr_np = np.clip(attr_np, -3, 3)
    return (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
