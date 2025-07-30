# Imports
import math  # noqa

# Project Imports
from bcos.data.presets import ISIC2018CINetClassificationPresetTrain, ISIC2018CINetClassificationPresetTest
from bcos.experiments.utils import configs_cli, create_configs_with_different_seeds, update_config
from bcos.modules import norms
from bcos.modules.losses import BinaryCrossEntropyLoss, UniformOffLabelsBCEWithLogitsLoss
from bcos.optim import LRSchedulerFactory, OptimizerFactory

__all__ = ["CONFIGS"]

NUM_CLASSES = 7
TR_PERC = 1.0

# These are mainly based on the recipes from
# https://github.com/pytorch/vision/blob/93723b481d1f6e/references/classification/README.md
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_EPOCHS = 100
DEFAULT_LR = 2e-4
NUM_WORKERS = 4
DEFAULT_NORM_LAYER = norms.NoBias(norms.BatchNormUncentered2d)  # bnu-linear
IS_BCOS = True
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





bcos_cinet = {f"bcos_cinet": update_default(dict(model=dict(name=f"bcos_cinet")))}

CONFIGS = dict()
CONFIGS.update(bcos_cinet)
CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[42]))

if __name__ == "__main__":
    configs_cli(CONFIGS)
