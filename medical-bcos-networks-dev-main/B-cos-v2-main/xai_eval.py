# Imports
import argparse
import os
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import pandas as pd
import random
random.seed(42)

# PyTorch Imports
import torch
torch.manual_seed(0)
from torch.utils.data import Subset, DataLoader
from torchvision.transforms.functional import resize

# Project Imports
from bcos.experiments.utils import Experiment
from interpretability.analyses.utils import Analyser, get_explainer_factory
from bcos.data.datamodules import CXR8DataModule, VinDrCXRDataModule

import matplotlib.pyplot as plt

import quantus
import json

from interpretability.metrics.faithfulness_correlation import FaithfulnessCorrelation
from interpretability.metrics.selectivity import Selectivity

from sklearn.model_selection import train_test_split

METRICS = {
    # Faithfulness
    "Selectivity": Selectivity(perturb_baseline="mean", patch_size=32, disable_warnings=True, return_aggregate=False),
    "Correlation": FaithfulnessCorrelation(perturb_baseline="mean", subset_size=224, nr_runs=100, disable_warnings=True, return_aggregate=False),
    # Robustness
    "MaxSensitivity": quantus.MaxSensitivity(nr_samples=5, return_nan_when_prediction_changes=False, disable_warnings=True, return_aggregate=False),
    "RRS": quantus.RelativeRepresentationStability(nr_samples=5, return_nan_when_prediction_changes=False, disable_warnings=True, return_aggregate=False),
    # Complexity
    "Sparseness": quantus.Sparseness(disable_warnings=True, return_aggregate=False),
    "Complexity": quantus.Complexity(disable_warnings=True, return_aggregate=False),
    # Localisation
    "Focus": quantus.Focus(disable_warnings=True, return_aggregate=False),
    "AttributionLocalisation": quantus.AttributionLocalisation(disable_warnings=True, return_aggregate=False)
}

class MosaicCollator:
    def __init__(self):
        pass
        
    def __call__(self, batch):
        # assuming square images
        _, w, _ = batch[0][0].shape
        bs = len(batch)
        stride = int(w / (bs / 2))

        mosaic_img = torch.zeros_like(batch[0][0])
        mosaic_img[:, :stride, :stride] = resize(batch[0][0], [stride, stride])
        mosaic_img[:, :stride, stride:] = resize(batch[1][0], [stride, stride])
        mosaic_img[:, stride:, :stride] = resize(batch[2][0], [stride, stride])
        mosaic_img[:, stride:, stride:] = resize(batch[3][0], [stride, stride])      

        # plot image for debugging purposes
        # plot_img = mosaic_img[:3, :, :]
        # plot_img = plot_img.cpu().numpy()
        # plot_img = np.transpose(plot_img, (1, 2, 0))
        # MEAN = [0.485, 0.456, 0.406]
        # STD = [0.229, 0.224, 0.225]

        # for c in range(plot_img.shape[2]):
        #     plot_img[:,:, c] = (plot_img[:, :, c] * STD[c]) + MEAN[c]

        # plot_img = ((plot_img - plot_img.min()) * (1/(plot_img.max() - plot_img.min()) * 255)).astype('uint8')
        # plt.imsave("mosaic.png", plot_img)

        if len(batch[0]) > 3: # CI-Net
            labels = [i[2].item() for i in batch][0]
        else:
            labels = [i[1] for i in batch][0]
                
        if len(batch[0]) > 2:
            ids = [i[-1] for i in batch][0]
            return mosaic_img.clone().detach().unsqueeze(0), labels.clone().detach().unsqueeze(0), [ids]
        else:
            return mosaic_img.clone().detach().unsqueeze(0), labels.clone().detach().unsqueeze(0)


class XAIMetricsAnalyser(Analyser):

    # Class attribute: Default configuration for xAI post-hoc method
    default_config = {
        "explainer_name": "Ours",
        "explainer_config": None
    }


    # Method: __init__
    def __init__(self, imgs_file: str, experiment: Experiment, verbose: bool = True, **config):
        
        """
        This analyser evaluates the localisation metric (see CoDA-Net paper).
        Args:
            experiment: Experiment object.
            **config:
                explainer_config: Config key for the explanation configurations.
                explainer_name: Which explanation method to load. Default is Ours.
                verbose: Warn when overwriting passed parameters with the analysis config parameters.
        """



        super().__init__(experiment=experiment, **config)

        with open(imgs_file, "r") as f:
            self.valid_images = json.load(f)

        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get explainer
        self.get_explainer = get_explainer_factory(supress_import_warnings=True)


        # Load model
        model_data = experiment.load_trained_model(
            reload="weights",
            ema=self.config["ema"],
            weights=self.config["weights"],
            verbose=verbose,
            return_training_ckpt_if_possible=True,
        )
        model = model_data["model"]

        # Set in evaluation mode...
        model = model.eval()
        self.model = model.to(self.device, non_blocking=True)

        # Get epoch
        try:
            if "ckpt" in model_data and model_data["ckpt"] is not None:
                self._epoch = model_data["ckpt"]["epoch"] + 1
            else:
                self._epoch = None
        except:
            self._epoch = None


        # Load explainer (xAI post-hoc method)
        self.explainer = self.get_explainer(
            model,
            self.config["explainer_name"],
            self.config["explainer_config"],
            batch_size=self.config["batch_size"],
        )
        
        # Get base save folder to save results
        self._base_save_folder = experiment.save_dir


    # Method: Get loaded epoch
    def get_loaded_epoch(self) -> Optional[int]:
        return self._epoch

    # Method: Get data
    def get_data(self, mosaic=False):
        # Get datamodules
        dataset_name = self.experiment.dataset
        datamodule = self.experiment.get_datamodule()
        if dataset_name == "ImageNet":
            datamodule.setup("val") 
            dataset = datamodule.eval_dataset
        else: 
            datamodule.setup("test", True)
            dataset = datamodule.test_dataset


        valid_idxs = []
        valid_labels = []
        gidx = 0
        for batch in tqdm(dataset, desc="Getting correctly classified instances"):
            if dataset_name == "ISIC2018CINet":
                _, _, label, _, _, _, _, _, _, imgid = batch
            elif dataset_name == "ImageNet":
                _, label = batch
                imgid = dataset.samples[gidx][0].split("/")[-1]
            else:
                _, label, imgid = batch
                if isinstance(datamodule, CXR8DataModule) or isinstance(datamodule, VinDrCXRDataModule):
                    if(np.sum(np.array(label), axis=0) == 0): continue
                    label = torch.argmax(label, dim=-1)
            if imgid in self.valid_images:
                valid_idxs.append(gidx)
                valid_labels.append(label)
            gidx += 1

        if dataset_name == "ImageNet":
            train_size = 2000
        else:
            train_size = 100

        if(len(valid_idxs) > train_size):
            valid_idxs, _, valid_labels, _ = train_test_split(valid_idxs, valid_labels, train_size=train_size, random_state=42, shuffle=True, stratify=valid_labels)

        df = pd.DataFrame({'imgs': valid_idxs, 'labels': valid_labels})

        if mosaic:
            all_idxs = [] 
            # for each image let's ensure we have 3 other images of different labels and our target image is in the top-left corner, i.e. comes first
            for img_id, label in zip(valid_idxs, valid_labels):
                all_idxs.append(img_id)
                sample = df[df['labels'] != label]
                other_imgs = random.sample(list(sample["imgs"].values), 3)
                all_idxs.extend(other_imgs)
            
            valid_idxs = all_idxs

        return Subset(dataset, valid_idxs)


    # Method: generate explanations
    def generate_explanation(self, inputs, targets, **kwargs):
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs).to(self.device)
            if inputs.ndim < 4: inputs = inputs.unsqueeze(0)
        attribution = self.explainer.attribute(inputs, targets)
        attribution = attribution.sum(1, keepdim=True)
        attribution = attribution.detach().cpu().numpy()

        return attribution

    # Method: compute the XAI metric
    def compute_metric(self, mname, dataloader, bbox=False):
        metric = METRICS[mname]
        datamodule = self.experiment.get_datamodule()
        dataset_name = self.experiment.dataset
        if bbox and hasattr(dataloader.dataset.dataset, "return_bbox"): dataloader.dataset.dataset.return_bbox = True

        results = []
        all_ids = []
        if dataset_name == "ImageNet": img_count = 0
        for batch in tqdm(dataloader, f"Computing metric {mname}"):
            if dataset_name == "ISIC2018CINet" and not isinstance(dataloader.collate_fn, MosaicCollator):
                x, _, y, _, _, _, _, _, _, imgid = batch
            elif dataset_name == "ImageNet":
                x, y = batch
                imgid = [dataloader.dataset.dataset.samples[i][0].split("/")[-1] for i in range(img_count, img_count+x.shape[0])]
                img_count += x.shape[0]
            else:
                bboxes = None
                if dataset_name == "VinDrCXR" and bbox:
                    x, y, bboxes, cnames, imgid = batch
                else:
                    x, y, imgid = batch

            if isinstance(datamodule, CXR8DataModule) or isinstance(datamodule, VinDrCXRDataModule):
                y = torch.argmax(y, dim=-1)

            x, y = x.to(self.device), y.to(self.device)

            expl = self.generate_explanation(inputs=x, targets=y)
            if (np.sum(expl) == 0): 
                print(imgid)
                continue

            if mname == "Focus":
                eval_res = metric(
                    model=self.model,
                    x_batch=x.detach().cpu().numpy(),
                    y_batch=y.detach().cpu().numpy(),
                    a_batch=expl,
                    custom_batch=[(1, 0, 0, 0)],
                    device=self.device,
                    explain_func=self.generate_explanation,
                    batch_size=1,
                )
            elif mname == "AttributionLocalisation":
                if bboxes is None: continue
                if len(bboxes) == 0: continue
                masks = np.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=np.uint8)
                cnames = list(np.array(cnames).flatten())
                if dataloader.dataset.dataset.use_global_labels:
                    assert hasattr(dataloader.dataset.dataset, "global_labels_to_names_dict"), "Dataset needs to have a dict global_labels_to_names_dict for AttributionLocalisation metric"
                    target_box = tuple(bboxes[cnames.index(dataloader.dataset.dataset.global_labels_to_names_dict[y[0].item()])][0].tolist())
                else:
                    assert hasattr(dataloader.dataset.dataset, "local_labels_to_names_dict"), "Dataset needs to have a dict local_labels_to_names_dict for AttributionLocalisation metric"
                    target_box = tuple(bboxes[cnames.index(dataloader.dataset.dataset.local_labels_to_names_dict[y[0].item()])][0].tolist())
                masks[:, :, int(target_box[1]):int(target_box[3]), int(target_box[0]):int(target_box[2])] = 1
                eval_res = metric(
                    model=self.model,
                    x_batch=x.detach().cpu().numpy(),
                    y_batch=y.detach().cpu().numpy(),
                    a_batch=expl,
                    s_batch=masks,
                    device=self.device,
                    explain_func=self.generate_explanation,
                    batch_size=self.config["batch_size"],
                )
            else:
                eval_res = metric(
                    model=self.model,
                    x_batch=x.detach().cpu().numpy(),
                    y_batch=y.detach().cpu().numpy(),
                    a_batch=expl,
                    device=self.device,
                    explain_func=self.generate_explanation,
                    batch_size=self.config["batch_size"],
                )
            if mname == "Selectivity":
                eval_res = metric.get_auc_score
                results.extend(eval_res)
            else:
                results.extend(eval_res)
            all_ids.extend(imgid)
        
        results_dict = dict(zip(all_ids, results))
        self.save_results(results_dict, fname=f"{mname}_individual")
        
        if mname == "RRS" or mname == "MaxSensitivity":
            results = np.max(results)
        else:
            results = np.mean(results)
        print(f"{mname}: {results}")
        return results
    
    # Method: perform analysis on several XAI metrics
    def analysis(self):
        all_metrics = {}
        
        for m in tqdm(METRICS):
            if m == "Focus":
                mosaic_dataset = self.get_data(mosaic=True)
                mosaic_dataloader = DataLoader(mosaic_dataset, batch_size=4, shuffle=False, num_workers=self.config["num_workers"], drop_last=False, collate_fn=MosaicCollator())
                all_metrics[m] = self.compute_metric(m, mosaic_dataloader)
            elif m == "AttributionLocalisation":
                dataset = self.get_data()
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.config["num_workers"], drop_last=False)
                all_metrics[m] = self.compute_metric(m, dataloader, bbox=True)
            else:
                dataset = self.get_data()
                dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=False, num_workers=self.config["num_workers"], drop_last=False)
                all_metrics[m] = self.compute_metric(m, dataloader)

        return all_metrics

    # Method: Save results
    def save_results(self, results, fname=None):
        save_dir: Path = self.experiment.save_dir / "xai_metrics"
        save_dir.mkdir(parents=True, exist_ok=True)

        final_fname = f"{self.config['explainer_name']}_{self.config['explainer_config']}_epoch_{self.get_loaded_epoch()}"
        if fname: final_fname += f"_{fname}"
        final_fname += ".json"
        with open(os.path.join(save_dir, final_fname), "w") as f:
            json.dump(results, f, indent=2)

        return


    # Method: Run
    def run(self):
        
        # Get results
        results = self.analysis()

        # Save results
        self.save_results(results)

        print(f"Finished.")

        return


# Function: Argument parser for the CLI
def argument_parser(multiple_args=False, add_help=True):
    """
    Create a parser with run_experiments arguments.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        description="XAI metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )
    parser.add_argument(
        "--ema",
        default=False,
        action="store_true",
        help="Load EMA weights instead if they exist.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=False,
        help="Loads the provided checkpoint."
    )
    parser.add_argument(
        "--imgs_file",
        type=str,
        required=True,
        help="File with correctly classified images for this dataset."
    )
    parser.add_argument(
        "--explainer_name",
        default="Ours",
        type=str,
        help="Which explainer method to use. Ours uses trainer.attribute.",
    )

    parser.add_argument(
        "--explainer_config",
        default="default",
        type=str,
        help="Which explainer configuration file to load.",
    )

    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size.",
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers.",
    )

    parser.add_argument("--debug", action="store_true", default=False)

    
    return parser



# Function: Get the arguments from the CLI
def get_arguments():
    parser = argument_parser()
    opts = parser.parse_args()

    if opts.explainer_name == "GCam": assert opts.batch_size == 1, "GradCam only supports batch size of 1."
    return opts



# Function: Run main
def main(config):
    print("Starting XAI metrics computation for", config)
    print()
    print()

    config.save_path = os.path.dirname(config.weights)

    experiment = Experiment(config.save_path)

    analyser = XAIMetricsAnalyser(
        config.imgs_file,
        experiment,
        explainer_name=config.explainer_name,
        explainer_config=config.explainer_config,
        ema=config.ema,
        weights=config.weights,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    analyser.run()



# Usage
if __name__ == "__main__":
    params = get_arguments()

    try:
        main(params)
    except Exception:
        if params.debug:
            import pdb

            pdb.post_mortem()
        raise
