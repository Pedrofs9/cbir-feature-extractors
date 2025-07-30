# Imports
import argparse

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x: x  # noqa: E731

# PyTorch Imports
import torch
import torchmetrics

import numpy as np
import os
import json

# Project Imports
from bcos.data.datamodules import ClassificationDataModule
from bcos.experiments.utils import Experiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model.", add_help=add_help
    )

    # specify save dir and experiment config
    parser.add_argument(
        "--hubconf",
        help="Test model from local hubconf file.",
    )

    parser.add_argument(
        "--base_directory",
        default="./experiments",
        help="The base directory.",
    )
    parser.add_argument(
        "--dataset",
        choices=["ImageNet", "CIFAR10", "ISIC2018", "CXR8", "IDRID", "EyePACS", "APTOS", "ISIC2018CINet", "VinDrCXR", "Herlev"],
        default="ImageNet",
        help="The dataset.",
    )
    parser.add_argument(
        "--base_network", help="The model config or base network to use."
    )
    parser.add_argument("--experiment_name", help="The name of the experiment to run.")

    # FIXME:
    # group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--reload", help="What ckpt to load. ['last', 'best', 'epoch_<N>', 'best_any']"
    )
    parser.add_argument("--weights", type=str, help="Specific path to the weight state dict to load.")

    parser.add_argument(
        "--ema",
        default=False,
        action="store_true",
        help="Load the EMA stored version if it exists. Not applicable for reload='best_any'.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size to use. Default is 1"
    )
    parser.add_argument(
        "--no-cuda",
        default=False,
        action="store_true",
        help="Force into not using cuda.",
    )

    return parser



# Function: Run evaluation on the test set
def run_evaluation(args):
    global device
    if args.no_cuda:
        device = torch.device("cpu")

    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)

    # get model, config, and data
    model, config = load_model_and_config(args)
    test_loader = get_test_loader(args.dataset, config)
    dataset = args.dataset
    save_path = os.path.dirname(args.weights)

    # do evaluation
    evaluate(model, test_loader, dataset, save_path)



# Function: Evaluate model
def evaluate(model, data_loader, dataset, save_path):
    
    # Put model in evaluation mode
    model.eval()

    # Initialise empty variables
    y_true = y_pred = None
    correct_imgs = []

    if dataset == "ImageNet":
        img_count = 0
    with torch.inference_mode():
        for batch in tqdm(data_loader):
            if dataset == "ISIC2018CINet":
                image, image1, label, _, _, _, _, _, _,  imgids = batch
                image1 = image1.to(device, non_blocking=True)
            elif dataset == "ImageNet":
                image, label = batch
            else:
                image, label, imgids = batch
            
            # Get batch of data
            image = image.to(device, non_blocking=True)
            # Get model outputs
            if dataset == "ISIC2018CINet":
                output, _, _ = model(image, image1)
                output = torch.sigmoid(output)
            else:
                output = model(image)

            for i in range(label.shape[0]):
                if dataset == "CXR8" or dataset == "VinDrCXR":
                    equal = np.array_equal(torch.where(output[i].cpu() > 0.5, 1, 0), label[i])     
                else:
                    preds = torch.argmax(output[i].cpu(), dim=-1)
                    equal = preds == label[i]

                if(equal):
                    if dataset == "ImageNet": 
                        correct_imgs.append(data_loader.dataset.samples[img_count][0].split("/")[-1])
                    else:
                        correct_imgs.append(imgids[i])
                
                if dataset == "ImageNet": img_count += 1

            # Concatenate y_pred
            if y_pred is None:
                y_pred = torch.clone(output.cpu())
            else:
                y_pred = torch.cat((y_pred, output.cpu()), dim=0)            
                
            # Concatenate y_true
            if y_true is None:
                y_true = torch.clone(label)
            else:
                y_true = torch.cat((y_true, label), dim=0)
                
    # Compute metrics
    metrics = {}
    if dataset == "CXR8" or dataset == "VinDrCXR":
        num_labels = 15 if dataset == "CXR8" else 22
        eval_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, compute_on_cpu=True, average="macro")
        metrics['acc'] = eval_acc(y_pred.cpu(), y_true.cpu())
        eval_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, compute_on_cpu=True, average="macro")
        metrics['f1'] = eval_f1(y_pred.cpu(), y_true.cpu())
        eval_auc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels, compute_on_cpu=True, average="macro")
        metrics['auc'] = eval_auc(y_pred.cpu(), y_true.cpu())
        eval_exact = torchmetrics.ExactMatch(task="multilabel", num_labels=num_labels, compute_on_cpu=True)
        metrics['exact'] = eval_exact(y_pred.cpu(), y_true.cpu())

    else:
        if dataset == "ISIC2018" or dataset == "ISIC2018CINet" or dataset == "Herlev":
            num_classes = 7 
        elif dataset == "ImageNet":
            num_classes = 1000
        else:
            num_classes = 5
        eval_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, compute_on_cpu=True, average="macro")
        metrics['acc'] = eval_acc(y_pred.cpu(), y_true.cpu())
        eval_bacc = torchmetrics.Accuracy(task="multiclass",num_classes=num_classes, compute_on_cpu=True, average="weighted")
        metrics['bacc'] = eval_bacc(y_pred.cpu(), y_true.cpu())
        eval_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, compute_on_cpu=True, average="macro")
        metrics['f1'] = eval_f1(y_pred.cpu(), y_true.cpu())
        eval_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes, compute_on_cpu=True, average="macro")
        metrics['auc'] = eval_auc(y_pred.cpu(), y_true.cpu())

    with open(os.path.join(save_path, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            print(f"{k}: {v.item()}", file=f)
            print(f"{k}: {v.item()}")
    
    with open(os.path.join(save_path, "correct_imgs.json"), "w") as f:
        json.dump(correct_imgs, f, indent=4)


def check_correct(output, target, topk=(1,)):
    with torch.inference_mode():
        maxk = max(topk)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum()
            res.append(correct_k)
        return res


def load_model_and_config(args):
    # a bit messy because of trying to directly use hubconf
    if args.hubconf is not None:
        import hubconf

        model = getattr(hubconf, args.hubconf)(pretrained=True)
        config = model.config


    else:
        experiment = Experiment(
            base_directory=args.base_directory,
            path_or_dataset=args.dataset,
            base_network=args.base_network,
            experiment_name=args.experiment_name,
        )
        config = experiment.config

        
        model = experiment.load_trained_model(
            reload=args.reload,
            verbose=True,
            ema=args.ema,
            weights=args.weights
        )


    # Pass model into GPU or CPU
    model = model.to(device)


    return model, config


def get_test_loader(dataset, config):
    registry = ClassificationDataModule.registry()
    # print(f"Registry: {registry}")
    if dataset in registry:
        datamodule = registry[dataset](config)
    else:
        available_datasets = list(registry.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset}'. Available datasets are: {available_datasets}"
        )

    # get data and set batchsize
    datamodule.batch_size = args.batch_size
    if dataset != "ImageNet":
        datamodule.setup("test", True)
    else:
        datamodule.setup("val")
    test_loader = datamodule.test_dataloader()

    return test_loader


if __name__ == "__main__":
    args = get_parser().parse_args()
    run_evaluation(args)
