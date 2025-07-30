# Imports
# from PIL import Image
# import timm

# PyTorch Imports
import torch
# from torch.nn import functional as F
# import torchvision.transforms as transforms
# from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights

# Bcos Imports
# from bcos.models.convnext import (
#     bcosconvnext_atto,
#     bcosconvnext_tiny,
#     bcosconvnext_small,
#     bcosconvnext_base,
#     bcosconvnext_large
# )
from bcos.models.densenet import (
    baseline_densenet121,
    baseline_densenet161,
    baseline_densenet169,
    baseline_densenet201,
    bcosdensenet121,
    bcosdensenet121_pretr,
    bcosdensenet121_pretr_l,
    bcosdensenet161,
    bcosdensenet161_pretr,
    bcosdensenet169,
    bcosdensenet169_pretr,
    bcosdensenet201,
    bcosdensenet201_pretr
)
# from bcos.models.pretrained import (
#     bcos_pretr_convnext_base,
#     bcos_pretr_vgg11_bnu,
#     bcos_pretr_resnext50_32x4d,
#     bcos_pretr_convnext_base_bnu,
#     bcos_pretr_convnext_tiny,
#     bcos_pretr_convnext_tiny_bnu,
#     bcos_pretr_densenet121,
#     bcos_pretr_densenet121_long,
#     bcos_pretr_densenet161,
#     bcos_pretr_densenet169,
#     bcos_pretr_densenet201,
#     bcos_pretr_resnet101,
#     bcos_pretr_resnet152,
#     bcos_pretr_resnet152_long,
#     bcos_pretr_resnet18,
#     bcos_pretr_resnet34,
#     bcos_pretr_resnet50
# )
# from bcos.models.resnet import (
#     bcosresnet18,
#     bcosresnet34,
#     bcosresnet50,
#     bcosresnet101,
#     bcosresnet152,
#     bcosresnext50_32x4d,
#     bcosresnext101_32x8d,
#     bcoswide_resnet50_2,
#     bcoswide_resnet101_2
# )
# from bcos.models.vgg import (
#     bcosvgg11,
#     bcosvgg11_bnu,
#     bcosvgg13,
#     bcosvgg13_bnu,
#     bcosvgg16,
#     bcosvgg16_bnu,
#     bcosvgg19,
#     bcosvgg19_bnu
# )



# Dictionary: Models dictionary
MODELS_DICT = {
    "baseline_densenet121":baseline_densenet121(num_classes=0, pretrained=False),
    "baseline_densenet121_pretr":baseline_densenet121(num_classes=0, pretrained=True),
    "baseline_densenet161":baseline_densenet161(num_classes=0, pretrained=False),
    "baseline_densenet161_pretr":baseline_densenet161(num_classes=0, pretrained=True),
    "baseline_densenet169":baseline_densenet169(num_classes=0, pretrained=False),
    "baseline_densenet169_pretr":baseline_densenet169(num_classes=0, pretrained=True),
    "baseline_densenet201":baseline_densenet201(num_classes=0, pretrained=False),
    "baseline_densenet201_pretr":baseline_densenet201(num_classes=0, pretrained=True),
    "bcosdensenet121":bcosdensenet121(num_classes=0),
    "bcosdensenet121_pretr":bcosdensenet121_pretr(num_classes=0),
    "bcosdensenet121_pretr_l":bcosdensenet121_pretr_l(num_classes=0),
    "bcosdensenet161":bcosdensenet161(num_classes=0),
    "bcosdensenet161_pretr":bcosdensenet161_pretr(num_classes=0),
    "bcosdensenet169":bcosdensenet169(num_classes=0),
    "bcosdensenet169_pretr":bcosdensenet169_pretr(num_classes=0),
    "bcosdensenet201":bcosdensenet201(num_classes=0),
    "bcosdensenet201_pretr":bcosdensenet201_pretr(num_classes=0)
}



# if __name__ == "__main__":

#     for k, v in MODELS_DICT.items():
#         model = v
#         if "bcos" in k:
#             rand_tensor = torch.rand(1, 6, 224, 224)
#         else:
#             rand_tensor = torch.rand(1, 3, 224, 224)
#         out = model(rand_tensor)
#         print(k, rand_tensor.shape, out.shape)