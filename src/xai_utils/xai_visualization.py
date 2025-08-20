import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import os
import traceback
import torch

# Local imports
from xai_utils.utilities_xai import compute_integrated_gradients, process_ig_to_heatmap
from utilities_visualization import (_concat_images_horizontally, _create_row_with_labels, 
                                              _setup_triplet_canvas, _load_and_resize_triplet_images, 
                                              _get_triplet_fonts, _place_triplet_images, 
                                              _create_gt_rank_mapping, _setup_ranking_canvas, 
                                              _add_query_to_canvas)

# ========== REUSABLE XAI FUNCTIONS ==========

def create_heatmap_overlay(pil_img, attribution):
    """Standardized heatmap generation with dimension handling"""
    # Convert attribution to proper numpy array
    attr_np = attribution.detach().cpu().numpy()
    
    # Handle different attribution shapes
    if attr_np.ndim == 4:  # Batch dimension present
        attr_np = attr_np[0]  # Remove batch dim
    if attr_np.ndim == 3:  # Channel dimension present
        attr_np = attr_np.mean(0)  # Average across channels
    attr_np = np.squeeze(attr_np)  # Remove any remaining single-dim entries
    
    # Normalize and threshold
    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
    threshold = np.percentile(attr_np, 85)
    
    # Create overlay
    green = np.zeros((*attr_np.shape, 4))  # RGBA array
    green[..., :3] = [0.043, 0.859, 0.082]  # RGB for #0bdb15
    green[..., 3] = np.where(attr_np >= threshold, 0.7, 0)  # Alpha channel
    
    # Convert to PIL Image
    overlay_img = Image.fromarray((green * 255).astype(np.uint8), 'RGBA')
    base_img = pil_img.convert('RGBA')
    return Image.alpha_composite(base_img, overlay_img).convert('RGB')

def _process_image_with_xai(image_path, model, device, transform, reference_tensor=None):
    """Process single image with XAI"""
    image_tensor = transform(image_path).unsqueeze(0).to(device)
    
    if reference_tensor is None:
        reference_tensor = image_tensor
    
    attr = compute_integrated_gradients(model, reference_tensor, image_tensor)
    img = create_heatmap_overlay(Image.open(image_path).convert('RGB'), attr)
    
    return img

def apply_xai_to_triplet(query_path, pos_path, neg_path, model, device, transform):
    """Process triplet images with Integrated Gradients"""
    # Process query
    query_img = _process_image_with_xai(query_path, model, device, transform)
    
    # Process positive and negative using query as reference
    query_tensor = transform(query_path).unsqueeze(0).to(device)
    pos_img = _process_image_with_xai(pos_path, model, device, transform, query_tensor)
    neg_img = _process_image_with_xai(neg_path, model, device, transform, query_tensor)
    
    return query_img, pos_img, neg_img

def apply_xai_to_ranking(query_path, neighbor_paths, model, device, transform):
    """Process query and neighbors for ranking visualization with Integrated Gradients"""
    query_img = _process_image_with_xai(query_path, model, device, transform)
    
    # Process neighbors using query as reference
    query_tensor = transform(query_path).unsqueeze(0).to(device)
    neighbor_imgs = []
    for path in neighbor_paths:
        neighbor_img = _process_image_with_xai(path, model, device, transform, query_tensor)
        neighbor_imgs.append(neighbor_img)
    
    return query_img, neighbor_imgs

def _create_xai_ranking_visualization(query_img, neighbor_imgs, model_ordering, gt_ordering, save_path=None):
    """Create ranking visualization from XAI-processed images"""
    # Resize all images uniformly
    img_height = 150
    resized_query = ImageOps.contain(query_img, (img_height, img_height))
    resized_neighbors = [ImageOps.contain(img, (img_height, img_height)) for img in neighbor_imgs]

    # Create mapping from image index to ground truth rank
    gt_rank_mapping = _create_gt_rank_mapping(gt_ordering)

    # Split into two rows
    num_images = len(model_ordering)
    split_point = (num_images + 1) // 2
    top_row_indices = model_ordering[:split_point]
    bottom_row_indices = model_ordering[split_point:]

    # Create rows using helper function
    top_row_images = _create_row_with_labels(resized_neighbors, top_row_indices, gt_rank_mapping, 1, True)
    bottom_row_images = _create_row_with_labels(resized_neighbors, bottom_row_indices, gt_rank_mapping, len(top_row_images)+1, True)
   
    # Concatenate images using helper
    top_row = _concat_images_horizontally(top_row_images)
    bottom_row = _concat_images_horizontally(bottom_row_images)

    # Create canvas
    spacing = 20
    label_height = 30
    canvas, draw, canvas_width, canvas_height = _setup_ranking_canvas(resized_query, top_row, bottom_row, spacing, label_height)

    # Add query image (heatmapped)
    query_x = 0
    query_y = label_height + (canvas_height - label_height - resized_query.height) // 2
    canvas.paste(resized_query, (query_x, query_y))
    
    # Add query label
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    label = "QUERY"
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Center label horizontally above the query image
    label_x = query_x + (resized_query.width - text_width) // 2
    label_y = query_y - text_height - 5  # 5 pixels above the image

    draw.text((label_x, label_y), label, font=font, fill='black')

    # Add results
    results_x = query_x + resized_query.width + spacing
    results_y = label_height
    canvas.paste(top_row, (results_x, results_y))
    canvas.paste(bottom_row, (results_x, results_y + top_row.height + 10))

    # Save or display
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        canvas.save(save_path)
        print(f"Saved XAI visualization to: {save_path}")
    else:
        canvas.show()

# ========== MAIN XAI VISUALIZATION FUNCTIONS ==========

def visualize_rankings_with_xai(query_path, neighbor_paths, model_ordering, gt_ordering, model, device, transform, save_path=None):
    """
    Generate XAI visualization for rankings using helper functions
    """
    try:
        # Get XAI-processed images
        query_img, neighbor_imgs = apply_xai_to_ranking(
            query_path, neighbor_paths,
            model, device, transform
        )
        
        # Create visualization
        _create_xai_ranking_visualization(
            query_img, neighbor_imgs, model_ordering, gt_ordering, save_path
        )

    except Exception as e:
        print(f"Error generating XAI visualization: {str(e)}")
        traceback.print_exc()

def visualize_triplet_with_xai(query_path, pos_path, neg_path, pos_distance, neg_distance, loss_value, correct, model, device, transform, save_path=None):
    """Triplet visualization with XAI heatmaps using helper functions"""
    try:
        # Get XAI-processed images
        query_img, pos_img, neg_img = apply_xai_to_triplet(
            query_path, pos_path, neg_path, model, device, transform
        )

        # Resize images
        img_height = 200
        query_img = ImageOps.contain(query_img, (img_height, img_height))
        pos_img = ImageOps.contain(pos_img, (img_height, img_height))
        neg_img = ImageOps.contain(neg_img, (img_height, img_height))
        
        # Setup canvas and fonts 
        canvas, draw, spacing, label_height = _setup_triplet_canvas(query_img, pos_img, neg_img)
        font, bold_font = _get_triplet_fonts()
        
        # Place images with labels 
        canvas = _place_triplet_images(canvas, draw, query_img, pos_img, neg_img, spacing, label_height, font)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            canvas.save(save_path)

    except Exception as e:
        print(f"Error generating XAI triplet visualization: {str(e)}")