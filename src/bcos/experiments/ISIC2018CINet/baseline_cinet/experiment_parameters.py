# Imports
import math  # noqa

# Project Imports
from bcos.data.presets import ISIC2018CINetClassificationPresetTrain, ISIC2018CINetClassificationPresetTest
from bcos.experiments.utils import configs_cli, create_configs_with_different_seeds, update_config
from bcos.modules import norms
from bcos.optim import LRSchedulerFactory, OptimizerFactory
from torch import nn

__all__ = ["CONFIGS"]

NUM_CLASSES = 7
PRETRAINED = False
TR_PERC = 1.0

# These are mainly based on the recipes from
# https://github.com/pytorch/vision/blob/93723b481d1f6e/references/classification/README.md
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_EPOCHS = 100
DEFAULT_LR = 2e-4
NUM_WORKERS = 4
IS_BCOS = False
DEFAULT_OPTIMIZER = OptimizerFactory(name="AdamW", lr=DEFAULT_LR, weight_decay=1e-4)
DEFAULT_LR_SCHEDULE = LRSchedulerFactory(
    name="steplr",
    step_size=40,
    gamma=0.1,
)

DEFAULTS = dict(
    data=dict(
        train_transform=ISIC2018CINetClassificationPresetTrain(is_bcos=IS_BCOS),
        test_transform=ISIC2018CINetClassificationPresetTest(is_bcos=IS_BCOS),
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        num_classes=NUM_CLASSES,
        tr_perc=TR_PERC
    ),
    model=dict(
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
    use_agc=True,
)


# Function: update_default
def update_default(new_config):
    return update_config(DEFAULTS, new_config)





baseline_cinet = {f"baseline_cinet": update_default(dict(model=dict(name=f"baseline_cinet")))}

CONFIGS = dict()
CONFIGS.update(baseline_cinet)
CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[42]))

if __name__ == "__main__":
    configs_cli(CONFIGS)
