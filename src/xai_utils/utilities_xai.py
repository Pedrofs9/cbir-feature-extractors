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
        return_convergence_delta=False  # Avoid tensor issues
    )
    
    # Verification
    with torch.no_grad():
        original_dist = -similarity_fn(neighbor_tensor).item()  # Get actual L2 distance
        print(f"\n[DEBUG] Original L2 distance: {original_dist:.4f}")
        
        mask = (attributions.abs() > attributions.abs().mean()).float()
        masked_dist = -similarity_fn(neighbor_tensor * mask).item()
        print(f"[DEBUG] Masked L2 distance: {masked_dist:.4f}")
        print(f"[DEBUG] Distance increase: {masked_dist - original_dist:.4f}")
    
    return attributions

def compute_sbsm(query_tensor, neighbor_tensor, model, block_size=24, stride=12):
    """
    SBSM (Similarity-Based Saliency Map) for any CNN that outputs flat feature vectors.
    """
    device = next(model.parameters()).device
    query_tensor = query_tensor.to(device)
    neighbor_tensor = neighbor_tensor.to(device)

    print("[DEBUG] Running SBSM with block_size =", block_size, ", stride =", stride)

    # Extract base embeddings
    with torch.no_grad():
        query_feat = model(query_tensor)
        base_feat = model(neighbor_tensor)

        print("[DEBUG] Output shapes - query:", query_feat.shape, ", neighbor:", base_feat.shape)

        query_feat = query_feat.flatten(1)
        base_feat = base_feat.flatten(1)
        base_dist = F.pairwise_distance(query_feat, base_feat, p=2).item()
        print("[DEBUG] L2 distance between query and neighbor:", base_dist)

    # Prepare saliency map
    _, _, H, W = neighbor_tensor.shape
    saliency = torch.zeros(H, W).to(device)
    count = torch.zeros(H, W).to(device)

    # Generate occlusion masks
    mask_batch = []
    positions = []
    for y in range(0, H - block_size + 1, stride):
        for x in range(0, W - block_size + 1, stride):
            mask = torch.ones_like(neighbor_tensor)
            mask[:, :, y:y + block_size, x:x + block_size] = 0
            mask_batch.append(mask)
            positions.append((y, x))

    if not mask_batch:
        raise RuntimeError("No masks generated. Check block_size and stride vs. input image dimensions.")

    print(f"[DEBUG] Total masked patches to evaluate: {len(mask_batch)}")

    # Batch processing
    mask_batch = torch.cat(mask_batch, dim=0).to(device)
    repeated_neighbor = neighbor_tensor.repeat(mask_batch.shape[0], 1, 1, 1)
    masked_imgs = repeated_neighbor * mask_batch

    with torch.no_grad():
        masked_feats = model(masked_imgs).flatten(1)
        dists = F.pairwise_distance(query_feat.expand_as(masked_feats), masked_feats, p=2)

    for idx, (y, x) in enumerate(positions):
        dist_drop = max(dists[idx].item() - base_dist, 0)
        importance_mask = torch.zeros(H, W).to(device)
        importance_mask[y:y + block_size, x:x + block_size] = 1
        saliency += dist_drop * importance_mask
        count += importance_mask

    # Normalize and smooth
    saliency = saliency / (count + 1e-8)
    saliency = saliency.cpu().numpy()
    saliency = np.maximum(saliency, 0)
    if saliency.max() > 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = gaussian_filter(saliency, sigma=min(block_size // 6, 3))

    print("[DEBUG] Final saliency map shape:", saliency.shape)
    return saliency

def process_ig_to_heatmap(attributions):
    attr_np = attributions.detach().cpu().squeeze().numpy()
    if attr_np.ndim == 3:
        attr_np = np.mean(attr_np, axis=0)
    attr_np = (attr_np - attr_np.mean()) / (attr_np.std() + 1e-8)
    attr_np = np.clip(attr_np, -3, 3)
    return (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
