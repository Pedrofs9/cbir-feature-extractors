# Imports

# PyTorch Imports
import torch

# MONAI Imports
from monai.visualize import CAM, GradCAM, GradCAMpp, GuidedBackpropGrad, GuidedBackpropSmoothGrad, SmoothGrad, VanillaGrad



# Function: Compute CAM by MONAI
def compute_cam_monai(in_tensor, class_idx, nn_module, target_layers, fc_layers):

    # Enable gradients on input tensor
    in_tensor.requires_grad_()

    # Create CAM object
    CAM_ = CAM(nn_module=nn_module, target_layers=target_layers, fc_layers=fc_layers)

    # Compute results
    result = CAM_(x=in_tensor, class_idx=class_idx)

    return result



# Function: Compute GradCAM by MONAI
def compute_gradcam_monai(in_tensor, class_idx, nn_module, target_layers, fc_layers):

    # Enable gradients on input tensor
    in_tensor.requires_grad_()

    # Create GradCAM object
    GCAM = GradCAM(nn_module=nn_module, target_layers=target_layers, fc_layers=fc_layers)

    # Compute results
    result = GCAM(x=in_tensor, class_idx=class_idx)

    return result


# Function: Compute GradCAMpp by MONAI
def compute_gradcampp_monai(in_tensor, class_idx, nn_module, target_layers, fc_layers):

    # Enable gradients on input tensor
    in_tensor.requires_grad_()

    # Create GradCAMpp object
    GCAMpp = GradCAMpp(nn_module=nn_module, target_layers=target_layers, fc_layers=fc_layers)

    # Compute results
    result = GCAMpp(x=in_tensor, class_idx=class_idx)

    return result



# Function: Compute GuidedBackpropGrad by MONAI
def compute_guided_backprop_grad_monai(in_tensor, class_idx, model):

    # Enable gradients on input tensor
    in_tensor.requires_grad_()

    # Create GuidedBackpropGrad object
    GBpG = GuidedBackpropGrad(model)

    # Compute results
    result = GBpG(x=in_tensor, index=class_idx)

    return result



# Function: Compute GuidedBackpropSmoothGrad by MONAI
def compute_guided_backpropsm_grad_monai(in_tensor, class_idx, model):

    # Enable gradients on input tensor
    in_tensor.requires_grad_()

    # Create GuidedBackpropSmoothGrad object
    GBpSG = GuidedBackpropSmoothGrad(model)

    # Compute results
    result = GBpSG(x=in_tensor, index=class_idx)

    return result



# Function: Compute SmoothGrad by MONAI
def compute_guided_smooth_grad_monai(in_tensor, class_idx, model):

    # Enable gradients on input tensor
    in_tensor.requires_grad_()

    # Create SmoothGrad object
    SG = SmoothGrad(model)

    # Compute results
    result = SG(x=in_tensor, index=class_idx)

    return result



# Function: Compute VanillaGrad by MONAI
def compute_guided_vanilla_grad_monai(in_tensor, class_idx, model):

    # Enable gradients on input tensor
    in_tensor.requires_grad_()

    # Create VanillaGrad object
    VG = VanillaGrad(model)

    # Compute results
    result = VG(x=in_tensor, index=class_idx)

    return result



# Function: Compute MONAI Results
def compute_monai_results(in_tensor, class_idx, method, **kwargs):

    assert method in ('CAM', 'GradCAM', 'GradCAMpp', 'GuidedBackpropGrad', 'GuidedBackpropSmoothGrad', 'SmoothGrad', 'VanillaGrad'), 'Please provide a valid method.'

    # Select attribution method
    if method == 'CAM':
        result = compute_cam_monai(in_tensor, class_idx, kwargs['nn_module'], kwargs['target_layers'], kwargs['fc_layers'])
    elif method == 'GradCAM':
        result = compute_gradcam_monai(in_tensor, class_idx,  kwargs['nn_module'], kwargs['target_layers'], kwargs['fc_layers'])
    elif method == 'GradCAMpp':
        result = compute_gradcampp_monai(in_tensor, class_idx,  kwargs['nn_module'], kwargs['target_layers'], kwargs['fc_layers'])
    elif method == 'GuidedBackpropGrad':
        result = compute_guided_backprop_grad_monai(in_tensor, class_idx, kwargs['model'])
    elif method == 'GuidedBackpropSmoothGrad':
        result = compute_guided_backpropsm_grad_monai(in_tensor, class_idx, kwargs['model'])
    elif method == 'SmoothGrad':
        result = compute_guided_smooth_grad_monai(in_tensor, class_idx, kwargs['model'])
    else:
        result = compute_guided_vanilla_grad_monai(in_tensor, class_idx, kwargs['model'])

    return result
