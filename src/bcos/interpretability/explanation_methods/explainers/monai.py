# MONAI Imports
from monai.visualize import (
    CAM,
    GradCAM,
    GradCAMpp,
    GuidedBackpropGrad,
    GuidedBackpropSmoothGrad,
    SmoothGrad,
    VanillaGrad
)

# PyTorch Imports
import torch

# Project Imports
from interpretability.explanation_methods.utils import ExplainerBase



# Class: GradCampp_MONAI
class GradCampp_MONAI(ExplainerBase):

    # Method: __init__
    def __init__(self, model, target_layers, fc_layers, model_fwd_mean_sub=False):

        # Get model and put it into eval mode
        self.model = model
        self.model.eval()

        # Get the parameters for the MONAI's GradCam++
        self.target_layers = target_layers
        self.fc_layers = fc_layers
        self.model_fwd_mean_sub = model_fwd_mean_sub
        
        # Initialise GradCAMpp from MONAI
        self.explainer = GradCAMpp(
            nn_module=self.model_forward_with_mean_subtracted if self.model_fwd_mean_sub else self.model,
            target_layers=self.target_layers,
            fc_layers=self.fc_layers
        )

        return
    

    # Method: model_forward_with_mean_subtracted
    def model_forward_with_mean_subtracted(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        assert (
            out.dim() == 2
        ), f"model output must be 2D (batch_size, num_classes) but is {out.ndim}D"
        
        return out - out.mean(dim=1, keepdim=True)


    # Method: attribute
    def attribute(self, img, target):

        # Enable gradient computation
        img.requires_grad_()

        # Compute MONAI's GradCam++ attribute
        attribute_ = self.explainer(
            x=img,
            class_idx=target
        )

        return attribute_
    

    # Method: attribute_selection
    def attribute_selection(self, image, targets):
        """
        From https://github.com/moboehle/B-cos/blob/4cd3b8ffc24d64c8b5b3262479/training/trainer_base.py#L467-L481
        Runs trainer.attribute for the list of targets.


        Args:
            image: Input image.
            targets: Target classes to check contributions for.

        Returns: Contributions for desired level.

        """
        
        attribute_selection_ = torch.cat([self.attribute(image, t) for t in targets], dim=0)

        return attribute_selection_



# Class: CAM_MONAI
class CAM_MONAI(ExplainerBase):

    # Method: __init__
    def __init__(self, model, target_layers, fc_layers, model_fwd_mean_sub=False):

        # Get model and put it into eval mode
        self.model = model
        self.model.eval()

        # Get the parameters for the MONAI's GradCam++
        self.target_layers = target_layers
        self.fc_layers = fc_layers
        self.model_fwd_mean_sub = model_fwd_mean_sub
        
        # Initialise CAM from MONAI
        self.explainer = CAM(
            nn_module=self.model_forward_with_mean_subtracted if self.model_fwd_mean_sub else self.model,
            target_layers=self.target_layers,
            fc_layers=self.fc_layers
        )

        return
    

    # Method: model_forward_with_mean_subtracted
    def model_forward_with_mean_subtracted(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        assert (
            out.dim() == 2
        ), f"model output must be 2D (batch_size, num_classes) but is {out.ndim}D"
        
        return out - out.mean(dim=1, keepdim=True)
    

    # Method: attribute
    def attribute(self, img, target):

        # Enable gradient computation
        img.requires_grad_()

        # Compute MONAI's CAM attribute
        attribute_ = self.explainer(
            x=img,
            class_idx=target
        )

        return attribute_
    

    # Method: attribute_selection
    def attribute_selection(self, image, targets):
        """
        From https://github.com/moboehle/B-cos/blob/4cd3b8ffc24d64c8b5b3262479/training/trainer_base.py#L467-L481
        Runs trainer.attribute for the list of targets.


        Args:
            image: Input image.
            targets: Target classes to check contributions for.

        Returns: Contributions for desired level.

        """
        
        attribute_selection_ = torch.cat([self.attribute(image, t) for t in targets], dim=0)

        return attribute_selection_



# Class: GradCAM_MONAI
class GradCAM_MONAI(ExplainerBase):

    # Method: __init__
    def __init__(self, model, target_layers, fc_layers, model_fwd_mean_sub=False):

        # Get model and put it into eval mode
        self.model = model
        self.model.eval()

        # Get the parameters for the MONAI's GradCam++
        self.target_layers = target_layers
        self.fc_layers = fc_layers
        self.model_fwd_mean_sub = model_fwd_mean_sub
        
        # Initialise GradCAM from MONAI
        self.explainer = GradCAM(
            nn_module=self.model_forward_with_mean_subtracted if self.model_fwd_mean_sub else self.model,
            target_layers=self.target_layers,
            fc_layers=self.fc_layers
        )

        return
    

    # Method: model_forward_with_mean_subtracted
    def model_forward_with_mean_subtracted(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        assert (
            out.dim() == 2
        ), f"model output must be 2D (batch_size, num_classes) but is {out.ndim}D"
        
        return out - out.mean(dim=1, keepdim=True)
    

    # Method: attribute
    def attribute(self, img, target):

        # Enable gradient computation
        img.requires_grad_()

        # Compute MONAI's GradCAM attribute
        attribute_ = self.explainer(
            x=img,
            class_idx=target
        )

        return attribute_
    

    # Method: attribute_selection
    def attribute_selection(self, image, targets):
        """
        From https://github.com/moboehle/B-cos/blob/4cd3b8ffc24d64c8b5b3262479/training/trainer_base.py#L467-L481
        Runs trainer.attribute for the list of targets.


        Args:
            image: Input image.
            targets: Target classes to check contributions for.

        Returns: Contributions for desired level.

        """
        
        attribute_selection_ = torch.cat([self.attribute(image, t) for t in targets], dim=0)

        return attribute_selection_



# Class: GuidedBackpropGrad_MONAI
class GuidedBackpropGrad_MONAI(ExplainerBase):

    # Method: __init__
    def __init__(self, model, model_fwd_mean_sub=False):

        # Get model and put it into eval mode
        self.model = model
        self.model.eval()

        # Initialise GuidedBackpropGrad from MONAI
        self.model_fwd_mean_sub = model_fwd_mean_sub
        self.explainer = GuidedBackpropGrad(
            self.model_forward_with_mean_subtracted if self.model_fwd_mean_sub else self.model
        )

        return
    

    # Method: model_forward_with_mean_subtracted
    def model_forward_with_mean_subtracted(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        assert (
            out.dim() == 2
        ), f"model output must be 2D (batch_size, num_classes) but is {out.ndim}D"
        
        return out - out.mean(dim=1, keepdim=True)


    # Method: attribute
    def attribute(self, img, target):

        # Enable gradient computation
        img.requires_grad_()

        # Compute MONAI's GuidedBackpropGrad attribute
        attribute_ = self.explainer(
            x=img,
            class_idx=target
        )

        return attribute_
    

    # Method: attribute_selection
    def attribute_selection(self, image, targets):
        """
        From https://github.com/moboehle/B-cos/blob/4cd3b8ffc24d64c8b5b3262479/training/trainer_base.py#L467-L481
        Runs trainer.attribute for the list of targets.


        Args:
            image: Input image.
            targets: Target classes to check contributions for.

        Returns: Contributions for desired level.

        """
        
        attribute_selection_ = torch.cat([self.attribute(image, t) for t in targets], dim=0)

        return attribute_selection_



# Class: GuidedBackpropSmoothGrad_MONAI
class GuidedBackpropSmoothGrad_MONAI(ExplainerBase):

    # Method: __init__
    def __init__(self, model, model_fwd_mean_sub=False):

        # Get model and put it into eval mode
        self.model = model
        self.model.eval()
        
        # Initialise GuidedBackpropSmoothGrad from MONAI
        self.model_fwd_mean_sub = model_fwd_mean_sub
        self.explainer = GuidedBackpropSmoothGrad(
            self.model_forward_with_mean_subtracted if self.model_fwd_mean_sub else self.model
        )

        return
    

    # Method: model_forward_with_mean_subtracted
    def model_forward_with_mean_subtracted(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        assert (
            out.dim() == 2
        ), f"model output must be 2D (batch_size, num_classes) but is {out.ndim}D"
        
        return out - out.mean(dim=1, keepdim=True)


    # Method: attribute
    def attribute(self, img, target):

        # Enable gradient computation
        img.requires_grad_()

        # Compute MONAI's GuidedBackpropSmoothGrad attribute
        attribute_ = self.explainer(
            x=img,
            class_idx=target
        )

        return attribute_
    

    # Method: attribute_selection
    def attribute_selection(self, image, targets):
        """
        From https://github.com/moboehle/B-cos/blob/4cd3b8ffc24d64c8b5b3262479/training/trainer_base.py#L467-L481
        Runs trainer.attribute for the list of targets.


        Args:
            image: Input image.
            targets: Target classes to check contributions for.

        Returns: Contributions for desired level.

        """
        
        attribute_selection_ = torch.cat([self.attribute(image, t) for t in targets], dim=0)

        return attribute_selection_



# Class: SmoothGrad_MONAI
class SmoothGrad_MONAI(ExplainerBase):

    # Method: __init__
    def __init__(self, model, model_fwd_mean_sub=False):

        # Get model and put it into eval mode
        self.model = model
        self.model.eval()
        
        # Initialise SmoothGrad from MONAI
        self.model_fwd_mean_sub = model_fwd_mean_sub
        self.explainer = SmoothGrad(
            self.model_forward_with_mean_subtracted if self.model_fwd_mean_sub else self.model
        )

        return
    

    # Method: model_forward_with_mean_subtracted
    def model_forward_with_mean_subtracted(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        assert (
            out.dim() == 2
        ), f"model output must be 2D (batch_size, num_classes) but is {out.ndim}D"
        
        return out - out.mean(dim=1, keepdim=True)


    # Method: attribute
    def attribute(self, img, target):

        # Enable gradient computation
        img.requires_grad_()

        # Compute MONAI's SmoothGrad attribute
        attribute_ = self.explainer(
            x=img,
            class_idx=target
        )

        return attribute_
    

    # Method: attribute_selection
    def attribute_selection(self, image, targets):
        """
        From https://github.com/moboehle/B-cos/blob/4cd3b8ffc24d64c8b5b3262479/training/trainer_base.py#L467-L481
        Runs trainer.attribute for the list of targets.


        Args:
            image: Input image.
            targets: Target classes to check contributions for.

        Returns: Contributions for desired level.

        """
        
        attribute_selection_ = torch.cat([self.attribute(image, t) for t in targets], dim=0)

        return attribute_selection_



# Class: VanillaGrad_MONAI
class VanillaGrad_MONAI(ExplainerBase):

    # Method: __init__
    def __init__(self, model, model_fwd_mean_sub=False):

        # Get model and put it into eval mode
        self.model = model
        self.model.eval()
        
        # Initialise VanillaGrad from MONAI
        self.model_fwd_mean_sub = model_fwd_mean_sub
        self.explainer = VanillaGrad(
            self.model_forward_with_mean_subtracted if self.model_fwd_mean_sub else self.model
        )

        return
    

    # Method: model_forward_with_mean_subtracted
    def model_forward_with_mean_subtracted(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        assert (
            out.dim() == 2
        ), f"model output must be 2D (batch_size, num_classes) but is {out.ndim}D"
        
        return out - out.mean(dim=1, keepdim=True)


    # Method: attribute
    def attribute(self, img, target):

        # Enable gradient computation
        img.requires_grad_()

        # Compute MONAI's VanillaGrad attribute
        attribute_ = self.explainer(
            x=img,
            class_idx=target
        )

        return attribute_
    

    # Method: attribute_selection
    def attribute_selection(self, image, targets):
        """
        From https://github.com/moboehle/B-cos/blob/4cd3b8ffc24d64c8b5b3262479/training/trainer_base.py#L467-L481
        Runs trainer.attribute for the list of targets.


        Args:
            image: Input image.
            targets: Target classes to check contributions for.

        Returns: Contributions for desired level.

        """
        
        attribute_selection_ = torch.cat([self.attribute(image, t) for t in targets], dim=0)

        return attribute_selection_
