# Imports
import math  # noqa

# Project Imports
from bcos.data.presets import CXR8ClassificationPresetTrain, CXR8ClassificationPresetTest
from bcos.experiments.utils import configs_cli, create_configs_with_different_seeds, update_config
from bcos.modules import norms
from bcos.modules.losses import BinaryCrossEntropyLoss, UniformOffLabelsBCEWithLogitsLoss
from bcos.optim import LRSchedulerFactory, OptimizerFactory

__all__ = ["CONFIGS"]

NUM_CLASSES = 15
TR_PERC = 0.5

# These are mainly based on the recipes from
# https://github.com/pytorch/vision/blob/93723b481d1f6e/references/classification/README.md
DEFAULT_BATCH_SIZE = 64  # per GPU! * 4 = 256 effective
DEFAULT_NUM_EPOCHS = 100
DEFAULT_LR = 1e-3
NUM_WORKERS = 4
IS_BCOS = True
DEFAULT_NORM_LAYER = norms.NoBias(norms.BatchNormUncentered2d)  # bnu-linear

DEFAULT_OPTIMIZER = OptimizerFactory(name="Adam", lr=DEFAULT_LR)
DEFAULT_LR_SCHEDULE = LRSchedulerFactory(
    name="cosineannealinglr",
    epochs=DEFAULT_NUM_EPOCHS,
    warmup_method="linear",
    warmup_epochs=5,
    warmup_decay=0.01,
)

DEFAULTS = dict(
    data=dict(
        train_transform=CXR8ClassificationPresetTrain(is_bcos=IS_BCOS),
        test_transform=CXR8ClassificationPresetTest(is_bcos=IS_BCOS),
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        num_classes=NUM_CLASSES,
        tr_perc=TR_PERC
    ),
    model=dict(
        is_bcos=IS_BCOS,
        # "name": None,
        args=dict(
            num_classes=NUM_CLASSES,
            norm_layer=DEFAULT_NORM_LAYER,
            logit_bias=-math.log(NUM_CLASSES - 1),
        ),
        bcos_args=dict(
            b=2,
            max_out=1,
        ),
    ),
    criterion=UniformOffLabelsBCEWithLogitsLoss(),
    test_criterion=BinaryCrossEntropyLoss(),
    optimizer=DEFAULT_OPTIMIZER,
    lr_scheduler=DEFAULT_LR_SCHEDULE,
    trainer=dict(max_epochs=DEFAULT_NUM_EPOCHS,),
    use_agc=True,
)


# Function: update_default
def update_default(new_config):
    return update_config(DEFAULTS, new_config)


RESNET_DEPTHS = [18, 34, 50, 101, 152]
resnets = {
    f"resnet{depth}": update_default(
        dict(
            model=dict(
                name=f"resnet{depth}",
            ),
        )
    )
    for depth in RESNET_DEPTHS
}

RESNEXT_DEPTHS_AND_WIDTHS = ["50_32x4d"]
resnexts = {
    f"resnext{depth_and_width}": update_default(
        dict(
            model=dict(
                name=f"resnext{depth_and_width}",
            ),
            lr_scheduler=DEFAULT_LR_SCHEDULE.with_epochs(100),
            trainer=dict(
                max_epochs=100,
            ),
        )
    )
    for depth_and_width in RESNEXT_DEPTHS_AND_WIDTHS
}

DENSENET_DEPTHS = [121, 161, 201, 169]
densenets = {
    f"densenet{depth}": update_default(
        dict(
            model=dict(
                name=f"densenet{depth}",
            ),
        )
    )
    for depth in DENSENET_DEPTHS
}

VGG_DEPTHS = [11]
vggs_bnu = {
    f"vgg{depth}_bnu": update_default(
        dict(
            model=dict(
                name=f"vgg{depth}_bnu",
            )
        )
    )
    for depth in VGG_DEPTHS
}

# -------------------------------------------------------------------------

CONFIGS = dict()
CONFIGS.update(resnets)
CONFIGS.update(resnexts)
CONFIGS.update(densenets)
CONFIGS.update(vggs_bnu)

# FIXME: Remove uppon testing
# CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[420, 1337]))
CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[42]))

if __name__ == "__main__":
    configs_cli(CONFIGS)
