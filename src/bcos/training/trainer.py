import os
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.plugins.environments as pl_env_plugins
import torch
import torchmetrics
from pytorch_lightning.utilities import rank_zero_info
import torchvision

try:
    import rich  # noqa: F401

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

import bcos.training.callbacks as custom_callbacks
from bcos.experiments.utils import CHECKPOINT_LAST_FILENAME, Experiment, sanitize_config
from bcos.training.agc import adaptive_clip_grad_
from bcos.training.ema import ExponentialMovingAverage


class ClassificationLitModel(pl.LightningModule):
    def __init__(self, dataset, base_network, experiment_name):
        super().__init__()
        self.experiment = Experiment(dataset, base_network, experiment_name)
        config = self.experiment.config
        model = self.experiment.get_model()

        self.save_hyperparameters()  # passed arguments
        self.save_hyperparameters(sanitize_config(config))  # the config as well

        self.dataset = dataset
        self.base_network = base_network
        self.experiment_name = experiment_name
        self.config = config
        self.model = model
        self.is_bcos = self.config["model"].get("is_bcos", False)
        self.criterion = self.config["criterion"]
        self.test_criterion = self.config["test_criterion"]
        num_classes = config["data"]["num_classes"]


        # Metrics (for each dataset)
        if self.dataset == "CXR8" or self.dataset == "VinDrCXR":
            num_labels = 15 if self.dataset == "CXR8" else 22

            # Train
            self.train_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, compute_on_cpu=True, average="macro")
            self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, compute_on_cpu=True, average="macro")
            self.train_auc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, compute_on_cpu=True, average="macro")
            self.train_exact = torchmetrics.ExactMatch(task="multilabel", num_labels=num_labels, compute_on_cpu=True)

            # Evaluation (Validation or Test)
            self.eval_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, compute_on_cpu=True, average="macro")
            self.eval_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, compute_on_cpu=True, average="macro")
            self.eval_auc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, compute_on_cpu=True, average="macro")
            self.eval_exact = torchmetrics.ExactMatch(task="multilabel", num_labels=num_labels, compute_on_cpu=True)


        else:

            # Train
            self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, compute_on_cpu=True, average="macro")
            self.train_bacc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, compute_on_cpu=True, average="weighted")
            self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, compute_on_cpu=True, average="macro")
            self.train_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes, compute_on_cpu=True, average="macro")

            # Evaluation (Validation or Test)
            self.eval_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, compute_on_cpu=True, average="macro")
            self.eval_bacc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, compute_on_cpu=True, average="weighted")
            self.eval_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, compute_on_cpu=True, average="macro")
            self.eval_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes, compute_on_cpu=True, average="macro")


        self.ema = None  # will be set during setup(stage="fit")
        self.ema_steps = None

        has_ema = self.config.get("ema", None) is not None
        self.eval_acc_ema = self.eval_acc.clone() if has_ema else None

        self.lr_warmup_epochs = self.config["lr_scheduler"].warmup_epochs

        self.use_agc = self.config.get("use_agc", False)
        if self.use_agc:
            rank_zero_info("Adaptive Gradient Clipping is enabled!")

    def setup(self, stage: str) -> None:
        if stage != "fit":
            return

        ema_config = self.config.get("ema", None)
        if ema_config is None:
            return

        decay = ema_config["decay"]
        self.ema_steps = ema_config.get("steps", 32)
        rank_zero_info(f"Using EMA with decay={decay} and steps={self.ema_steps}")

        # see https://github.com/pytorch/vision/blob/657c0767c5ca5564c8b437ac442/references/classification/train.py#L317
        adjust = (
            self.trainer.world_size
            * self.trainer.accumulate_grad_batches
            * self.config["data"]["batch_size"]
            * self.ema_steps
            / self.trainer.max_epochs
        )
        alpha = 1.0 - decay
        alpha = min(1.0, alpha * adjust)
        self.ema = ExponentialMovingAverage(self.model, decay=1.0 - alpha)
        self.ema.requires_grad_(False)

    def configure_optimizers(self):
        optimizer = self.config["optimizer"].create(self.model)
        scheduler = self.config["lr_scheduler"].create(
            optimizer,
            # this is total as in "whole" training
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def forward(self, in_tensor, in_tensor2=None):
        if "cinet" in self.base_network:
            return self.model(in_tensor, in_tensor2)
        return self.model(in_tensor)

    def training_step(self, batch, batch_idx):

        if "cinet" in self.base_network:
            img0, img1, classlabel0, classlabel1, synlabel, _, _, _, _ = batch
            result0, result1, synscore = self(img0, img1)
            loss0 = self.criterion(result0, classlabel0)
            loss1 = self.criterion(result1, classlabel1)
            synloss = self.criterion(synscore, synlabel)
            loss =  loss0 + loss1 + 0.1 * synloss
            labels = classlabel0
            outputs = result0
        else:
            images, labels = batch
            outputs = self(images)
            if self.dataset == "CXR8" or self.dataset == "VinDrCXR":
                loss = self.criterion(outputs, labels.float())
            else:
                loss = self.criterion(outputs, labels)


        with torch.no_grad():
            
            # Log the training loss
            self.log("train_loss", loss)

            if "cinet" in self.base_network:
                self.log("train_loss0", loss0)
                self.log("train_loss1", loss1)
                self.log("train_synloss", synloss)

            self.train_acc(outputs, labels)            
            self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
            if hasattr(self, "train_bacc"):
                self.train_bacc(outputs, labels)            
                self.log("train_bacc", self.train_bacc, on_step=True, on_epoch=True, prog_bar=True)

            if hasattr(self, "train_f1"):
                self.train_f1(outputs, labels)            
                self.log("train_f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True)

            if hasattr(self, "train_auc"):     
                self.train_auc(outputs, labels)            
                self.log("train_auc", self.train_auc, on_step=True, on_epoch=True, prog_bar=True)

            if hasattr(self, "train_exact"):
                self.train_exact(outputs, labels)
                self.log("train_exact", self.train_exact, on_step=True, on_epoch=True, prog_bar=True)

            if self.ema is not None and batch_idx % self.ema_steps == 0:
                ema = self.ema
                ema.update_parameters(self.model)
                if self.trainer.current_epoch < self.lr_warmup_epochs:
                    ema.n_averaged.fill_(0)

        return loss

    def eval_step(self, batch, _batch_idx, val_or_test):

        if "cinet" in self.base_network:
            img0, img1, classlabel0, classlabel1, synlabel, _, _, _, _ = batch
            result0, result1, synscore = self(img0, img1)
            loss0 = self.criterion(result0, classlabel0)
            loss1 = self.criterion(result1, classlabel1)
            synloss = self.criterion(synscore, synlabel)
            loss =  loss0 + loss1 + 0.1 * synloss            
            labels = classlabel0
            outputs = result0
        else:
            images, labels = batch
            outputs = self(images)
            if self.dataset == "CXR8" or self.dataset == "VinDrCXR":
                loss = self.test_criterion(outputs, labels.float())
            else:
                loss = self.test_criterion(outputs, labels)
        
        # Compute metrics
        self.log(f"{val_or_test}_loss", loss)

        if "cinet" in self.base_network:
            self.log("eval_loss0", loss0)
            self.log("eval_loss1", loss1)
            self.log("eval_synloss", synloss)

        self.eval_acc(outputs, labels)            
        self.log(f"{val_or_test}_acc", self.eval_acc, on_epoch=True, prog_bar=True)

        if hasattr(self, "eval_bacc"):
            self.eval_bacc(outputs, labels)            
            self.log(f"{val_or_test}_bacc", self.eval_bacc, on_epoch=True, prog_bar=True)

        if hasattr(self, "eval_f1"):
            self.eval_f1(outputs, labels)            
            self.log(f"{val_or_test}_f1", self.eval_f1, on_epoch=True, prog_bar=True)

        if hasattr(self, "eval_auc"):     
            self.eval_auc(outputs, labels)            
            self.log(f"{val_or_test}_auc", self.eval_auc, on_epoch=True, prog_bar=True)

        if hasattr(self, "eval_exact"):
            self.eval_exact(outputs, labels)
            self.log(f"{val_or_test}_exact", self.eval_exact, on_epoch=True, prog_bar=True)

        if self.ema is not None:
            ema = self.ema
            outputs = ema.module(images)
            ema_loss = self.test_criterion(outputs, labels)
            self.eval_acc_ema(outputs, labels)
            self.log(f"{val_or_test}_loss_ema", loss)
            self.log(f"{val_or_test}_acc_ema", self.eval_acc_ema, on_epoch=True)
            return {"loss": loss, "loss_ema": ema_loss}
        else:
            return loss

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_gradient_clipping(
        self,
        optimizer,
        optimizer_idx,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ) -> None:
        # Note: this is called even if gradient_clip_val etc. is None
        if not self.use_agc:
            self.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)
        else:
            adaptive_clip_grad_(self.parameters())

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        # we only care about total grad norm
        norm_type = float(self.trainer.track_grad_norm)
        total_norm = grad_norm_dict[f"grad_{norm_type}_norm_total"]
        del grad_norm_dict
        self.log(
            "gradients/total_norm",
            total_norm,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if self.trainer.gradient_clip_val is not None:
            clipped_total_norm = min(
                float(self.trainer.gradient_clip_val), float(total_norm)
            )
            self.log(
                "gradients/clipped_total_norm",
                clipped_total_norm,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )


def put_trainer_args_into_trainer_config(args, trainer_config):
    if args.distributed:
        # https://github.com/Lightning-AI/lightning/discussions/6761#discussioncomment-2614296
        trainer_config["strategy"] = "ddp_find_unused_parameters_false"

    if args.fast_dev_run:
        trainer_config["fast_dev_run"] = True

    if hasattr(args, "nodes"):  # on slurm
        trainer_config["num_nodes"] = args.nodes

    if args.track_grad_norm:
        trainer_config["track_grad_norm"] = 2.0

    if hasattr(args, "amp") and args.amp:
        trainer_config["precision"] = 16

    if args.debug:
        trainer_config["deterministic"] = True


def run_training(args):
    """
    Instantiates everything and runs the training.
    """
    base_directory = args.base_directory
    dataset = args.dataset
    base_network = args.base_network
    experiment_name = args.experiment_name
    save_dir = Path(base_directory, dataset, base_network, experiment_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    # set up loggers early so that WB starts capturing output asap
    loggers = setup_loggers(args)

    # get config
    exp = Experiment(dataset, base_network, experiment_name)
    config = exp.config.copy()

    # get and set seed
    seed = exp.config.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    # init model
    model = ClassificationLitModel(
        dataset,
        base_network,
        experiment_name,
    )
    rank_zero_info(f"Model: {repr(model.model)}")

    # jit the internal model if specified
    if args.jit:
        model.model = torch.jit.script(model.model)
        rank_zero_info("Jitted the model!")

    # load pretrained weights from baseline to bcos
    if args.pretrained_weights:
        print(f"Loading pretrained weights from {args.pretrained_weights}")
        if args.pretrained_weights == "densenet121_imagenet":
            pretrained_model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT).features
            pretrained_state_dict = pretrained_model.state_dict()
            args.features_only = True
        else:
            ckpt = torch.load(args.pretrained_weights)
            if 'state_dict' in ckpt:
                pretrained_state_dict = ckpt["state_dict"]
            else:
                pretrained_state_dict = ckpt
        
        state_dict = {}
        for k,v in pretrained_state_dict.items():
            if "model" in k:
                state_dict[k] = v
            else:
                state_dict["model." + k] = v
        
        if "model.features.conv0.linear.weight" in state_dict:
            nchannels = state_dict["model.features.conv0.linear.weight"].shape[1]
        elif "model.features.conv0.weight" in state_dict:
            nchannels = state_dict["model.features.conv0.weight"].shape[1]
        is_bcos = nchannels == 6

        if is_bcos:
            # ignore classification layer in case pretrained bcos network had a different number of classes
            if(state_dict["model.classifier.linear.weight"].shape[0] != model.model.classifier.linear.weight.shape[0]):
                del state_dict["model.classifier.linear.weight"]
                args.features_only = True
            missing, extra = model.load_state_dict(state_dict, strict=False)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                new_v = v
                if "conv" in k:
                    nr = k.split("conv")[-1].split(".")[0]
                    new_k = k.replace(f"conv{nr}.", f"conv{nr}.linear.")
                elif "classifier.weight" in k and not args.features_only:
                    new_k = k.replace("classifier.weight", "classifier.linear.weight")
                    new_v = v[:, :, None, None]
                
                if "conv0.linear" in new_k:
                    new_v = v.repeat(1, 2, 1, 1)

                new_state_dict[new_k] = new_v
            missing, _ = model.load_state_dict(new_state_dict, strict=False)
        assert len(missing) == 0 if not args.features_only else len(missing) == 1

    # init datamodule
    datamodule = model.experiment.get_datamodule(
        cache_dataset=getattr(args, "cache_dataset", None),
    )

    # callbacks
    callbacks = setup_callbacks(args, config)

    # init trainer
    trainer_config = config["trainer"]
    put_trainer_args_into_trainer_config(args, trainer_config)

    # plugin for slurm
    if "SLURM_JOB_ID" in os.environ:  # we're on slurm
        # let submitit handle requeuing
        trainer_config["plugins"] = [
            pl_env_plugins.SLURMEnvironment(auto_requeue=False)
        ]

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
        **trainer_config,
    )

    # decide whether to resume
    ckpt_path = None
    if args.resume:
        ckpt_path = save_dir / CHECKPOINT_LAST_FILENAME
        ckpt_path = ckpt_path if ckpt_path.exists() else None

    # start training
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def setup_loggers(args):
    loggers = []
    save_dir = Path(
        args.base_directory, args.dataset, args.base_network, args.experiment_name
    )

    if args.wandb_logger:
        wandb_logger = pl_loggers.WandbLogger(
            name=args.wandb_name or args.experiment_name,
            save_dir=str(save_dir),
            project=args.wandb_project,
            id=args.wandb_id,
        )
        loggers.append(wandb_logger)

    if args.csv_logger:
        csv_logger = pl_loggers.CSVLogger(
            save_dir=str(save_dir / "csv_logs"),
            name="",
            flush_logs_every_n_steps=1000,
        )
        loggers.append(csv_logger)

    if args.tensorboard_logger:
        tensorboard_logger = pl_loggers.TensorBoardLogger(
            save_dir=Path(
                "tb_logs",
                args.base_directory,
                args.dataset,
                args.base_network,
                args.experiment_name,
            ),
            name=args.experiment_name,
        )
        loggers.append(tensorboard_logger)

    return loggers


def setup_callbacks(args, config):
    callbacks = []
    save_dir = Path(
        args.base_directory, args.dataset, args.base_network, args.experiment_name
    )

    # the most important one
    save_callback = pl_callbacks.ModelCheckpoint(
        dirpath=save_dir,
        monitor="val_f1",
        mode="max",
        filename="best",
        save_last=True,
        save_top_k=1,
        verbose=True,
    )
    callbacks.append(save_callback)

    use_ema = config.get("ema", None) is not None
    if use_ema:
        save_callback = pl_callbacks.ModelCheckpoint(
            dirpath=save_dir,
            monitor="val_acc_ema",
            mode="max",
            filename="best_ema",
            save_top_k=1,
            verbose=True,
        )
        callbacks.append(save_callback)

    # lr monitor
    has_logger = args.wandb_logger or args.tensorboard_logger or args.csv_logger
    if has_logger:  # ow it's useless
        callbacks.append(pl_callbacks.LearningRateMonitor())
    slurm_or_submitit = hasattr(args, "nodes") or "SLURM_JOB_ID" in os.environ
    refresh_rate = args.refresh_rate or (20 if slurm_or_submitit else 5)
    if HAS_RICH and not slurm_or_submitit:
        callbacks.append(pl_callbacks.RichProgressBar(refresh_rate=refresh_rate))
    else:
        callbacks.append(pl_callbacks.TQDMProgressBar(refresh_rate=refresh_rate))

    # save metrics to checkpoint
    callbacks.append(custom_callbacks.MetricsTracker())

    # do explanation logging
    if args.explanation_logging:
        log_every = args.explanation_logging_every_n_epochs
        rank_zero_info(f"Will log explanations every {log_every} epoch(s)!")
        callbacks.append(
            custom_callbacks.ExplanationsLogger(log_every_n_epochs=log_every)
        )
    else:
        rank_zero_info("Explanation logging is disabled!")

    # for debugging purposes
    if args.debug:
        callbacks.append(custom_callbacks.ModelUpdateHasher())

    return callbacks
