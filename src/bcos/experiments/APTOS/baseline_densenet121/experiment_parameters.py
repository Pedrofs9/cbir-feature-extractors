# Imports
import math  # noqa

import torch.nn as nn

# Project Imports
from bcos.data.presets import APTOSClassificationPresetTrain, APTOSClassificationPresetTest
from bcos.experiments.utils import configs_cli, create_configs_with_different_seeds, update_config
from bcos.optim import LRSchedulerFactory, OptimizerFactory

__all__ = ["CONFIGS"]

NUM_CLASSES = 5
PRETRAINED = True
TR_PERC = 1.0

# These are mainly based on the recipes from
# https://github.com/pytorch/vision/blob/93723b481d1f6e/references/classification/README.md
DEFAULT_BATCH_SIZE = 8  # per GPU! * 4 = 256 effective
DEFAULT_NUM_EPOCHS = 100
DEFAULT_LR = 1e-4
NUM_WORKERS = 4
IS_BCOS = False

# no norm layer specification because the models set BN as default anyways

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
        train_transform=APTOSClassificationPresetTrain(is_bcos=IS_BCOS),
        test_transform=APTOSClassificationPresetTest(is_bcos=IS_BCOS),
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        num_classes=NUM_CLASSES,
        tr_perc=TR_PERC
    ),
    model=dict(
        is_bcos=IS_BCOS,
        args=dict(
            num_classes=NUM_CLASSES,
            pretrained=PRETRAINED,
        ),
    ),
    criterion=nn.CrossEntropyLoss(),
    test_criterion=nn.CrossEntropyLoss(),
    optimizer=DEFAULT_OPTIMIZER,
    lr_scheduler=DEFAULT_LR_SCHEDULE,
    trainer=dict(max_epochs=DEFAULT_NUM_EPOCHS,),
)


# Function: update_default
def update_default(new_config):
    return update_config(DEFAULTS, new_config)



# Dcitionary w/ DenseNet parameters
DENSENET_DEPTHS = [121, 161, 201, 169]
densenets = {
    f"baseline_densenet{depth}": update_default(
        dict(
            model=dict(
                name=f"baseline_densenet{depth}",
            ),
        )
    )
    for depth in DENSENET_DEPTHS
}

# -------------------------------------------------------------------------

CONFIGS = dict()
CONFIGS.update(densenets)
CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[42]))

if __name__ == "__main__":
    configs_cli(CONFIGS)
