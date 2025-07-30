# Albumentations Imports
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# PyTorch Imports
import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

# Project Imports
import bcos.data.transforms as custom_transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        is_bcos=False,
    ):
        self.args = get_args_dict(ignore=["mean", "std"])
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(
                    autoaugment.RandAugment(
                        interpolation=interpolation, magnitude=ra_magnitude
                    )
                )
            elif auto_augment_policy == "ta_wide":
                trans.append(
                    autoaugment.TrivialAugmentWide(interpolation=interpolation)
                )
            elif auto_augment_policy == "augmix":
                trans.append(
                    autoaugment.AugMix(
                        interpolation=interpolation, severity=augmix_severity
                    )
                )
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(
                    autoaugment.AutoAugment(
                        policy=aa_policy, interpolation=interpolation
                    )
                )
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )
        if not is_bcos:
            trans.append(transforms.Normalize(mean=mean, std=std))
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        if is_bcos:
            trans.append(custom_transforms.AddInverse())

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms.transforms)})"

    def __rich_repr__(self):
        # https://rich.readthedocs.io/en/stable/pretty.html
        yield "transforms", self.transforms.transforms

    def __to_config__(self):
        """See bcos.experiments.utils.sanitize_config for details."""
        result = dict(
            transform=repr(self),
            **self.args,
        )
        result["interpolation"] = str(result["interpolation"])
        return result


class ImageNetClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        interpolation=InterpolationMode.BILINEAR,
        is_bcos=False,
    ):
        self.args = get_args_dict(ignore=["mean", "std"])
        trans = [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            custom_transforms.AddInverse()
            if is_bcos
            else transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    @property
    def resize(self):
        return self.transforms.transforms[0]

    @property
    def center_crop(self):
        return self.transforms.transforms[1]

    def no_scale(self, img):
        x = img
        for t in self.transforms.transforms[2:]:
            x = t(x)
        return x

    # this is intended for when using the pretrained models
    def transform_with_options(self, img, center_crop=True, resize=True):
        x = img
        if resize:
            x = self.resize(x)
        if center_crop:
            x = self.center_crop(x)
        x = self.no_scale(x)
        return x

    def with_args(self, **kwargs):
        args = self.args.copy()
        args.update(kwargs)
        return self.__class__(**args)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms

    def __to_config__(self):
        """See bcos.experiments.utils.sanitize_config for details."""
        result = dict(
            transform=repr(self),
            **self.args,
        )
        result["interpolation"] = str(result["interpolation"])
        return result


CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)


class CIFAR10ClassificationPresetTrain:
    def __init__(
        self,
        *,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        is_bcos=False,
    ):
        trans = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            custom_transforms.AddInverse()
            if is_bcos
            else transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms


class CIFAR10ClassificationPresetTest:
    def __init__(
        self,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        is_bcos=False,
    ):
        trans = [
            transforms.ToTensor(),
            custom_transforms.AddInverse()
            if is_bcos
            else transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms


def get_args_dict(ignore: "tuple | list" = tuple()):
    """Helper for saving args easily."""
    import inspect

    frame = inspect.currentframe().f_back
    av = inspect.getargvalues(frame)
    ignore = tuple(ignore) + ("self", "cls")
    return {arg: av.locals[arg] for arg in av.args if arg not in ignore}



# Class: CXR8ClassificationPresetTrain
class CXR8ClassificationPresetTrain:
    def __init__(self, *, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.transforms = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.RandomAffine(degrees=45, translate=(0.15, 0.15), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms



# Class: CXR8ClassificationPresetTest
class CXR8ClassificationPresetTest:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.transforms = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms



# Class: VinDrCXRClassificationPresetTrain
class VinDrCXRClassificationPresetTrain:
    def __init__(self, *, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.inverse = None
        if is_bcos:
            self.transforms = A.Compose(
                [
                    A.Resize(height=448, width=448, always_apply=True, p=1),
                    A.Affine(rotate=(-45, 45), translate_percent=(0.0, 0.15), scale=(0.9, 1.1), p=1),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=[])
            )
            self.inverse = custom_transforms.AddInverse()
        else:
            self.transforms = A.Compose(
                [
                    A.Resize(height=448, width=448, always_apply=True, p=1),
                    A.Affine(rotate=(-45, 45), translate_percent=(0.0, 0.15), scale=(0.9, 1.1), p=1),
                    A.Normalize(mean=mean, std=std, always_apply=True, p=1),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=[])
            )

    def __call__(self, image, bboxes=None):

        # Get transformed image
        new_bboxes = None
        if bboxes:
            res = self.transforms(image=image, bboxes=bboxes)
            new_bboxes = res["bboxes"]
        else:
            res = self.transforms(image=image, bboxes=[[0, 0, 1.0, 1.0]])

        if self.inverse:
            return self.inverse(res["image"]), new_bboxes
        return res["image"], new_bboxes

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms



# Class: VinDrCXRClassificationPresetTest
class VinDrCXRClassificationPresetTest:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.inverse = None
        if is_bcos:
            self.transforms = A.Compose(
                [
                    A.Resize(height=448, width=448, always_apply=True, p=1),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=[])
            )
            self.inverse = custom_transforms.AddInverse()
        else:
            self.transforms = A.Compose(
                [
                    A.Resize(height=448, width=448, always_apply=True, p=1),
                    A.Normalize(mean=mean, std=std, always_apply=True, p=1),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=[])
            )

    def __call__(self, image, bboxes=None):

        # Get transformed image
        new_bboxes = None
        if bboxes:
            res = self.transforms(image=image, bboxes=bboxes)
            new_bboxes = res["bboxes"]
        else:
            res = self.transforms(image=image, bboxes=[[0, 0, 1.0, 1.0]])

        if self.inverse:
            return self.inverse(res["image"]), new_bboxes
        return res["image"], new_bboxes


    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms



# Class: ISIC2018ClassificationPresetTrain
class ISIC2018ClassificationPresetTrain:
    def __init__(self, *, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.transforms = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.9, 1.3), shear=10),
                transforms.ToTensor(),
                custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms



# Class: ISIC2018ClassificationPresetTest
class ISIC2018ClassificationPresetTest:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.transforms = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms



# Class: ISIC2018CINetClassificationPresetTrain
class ISIC2018CINetClassificationPresetTrain:
    def __init__(self, *, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.transforms = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.9, 1.3), shear=10),
                transforms.ToTensor(),
                custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms



# Class: ISIC2018CINetClassificationPresetTest
class ISIC2018CINetClassificationPresetTest:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.transforms = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms



# Class: IDRIDClassificationPresetTrain
class IDRIDClassificationPresetTrain:
    def __init__(self, *, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms



# Class: IDRIDClassificationPresetTest
class IDRIDClassificationPresetTest:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms



# Class: EyePACSClassificationPresetTrain
class EyePACSClassificationPresetTrain:
    def __init__(self, *, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms



# Class: EyePACSClassificationPresetTest
class EyePACSClassificationPresetTest:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms


# Class: APTOSClassificationPresetTrain
class APTOSClassificationPresetTrain:
    def __init__(self, *, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.transforms = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.RandomAffine(degrees=180, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms



# Class: APTOSClassificationPresetTest
class APTOSClassificationPresetTest:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.transforms = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms


# Class: HerlevClassificationPresetTrain
class HerlevClassificationPresetTrain:
    def __init__(self, *, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomAffine(180, (0, 0.1), (0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(saturation=(0.5, 2.0)),
                transforms.ToTensor(),  # vgg normalization
                custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms


# Class: HerlevClassificationPreset
class HerlevClassificationPresetTest:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_bcos=False):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms