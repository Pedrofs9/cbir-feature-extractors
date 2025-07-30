# Imports
import argparse
import os
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import json

# PyTorch Imports
import torch

# Project Imports
from bcos.experiments.utils import Experiment
from interpretability.analyses.utils import Analyser, get_explainer_factory



# Class VisualExplanationsAnalyser: Generate visual explanations using xAI post-hoc methods
class VisualExplanationsAnalyser(Analyser):

    # Class attribute: Default configuration for xAI post-hoc method
    default_config = {
        "explainer_name": "Ours",
        "explainer_config": None
    }


    # Method: __init__
    def __init__(self, experiment: Experiment, verbose: bool = True, **config):
        
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



        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get explainer
        self.get_explainer = get_explainer_factory(supress_import_warnings=True)


        # Load model
        model_data = experiment.load_trained_model(
            reload=self.config["reload"],
            ema=self.config["ema"],
            weights=self.config["weights"] if self.config["weights"] else None,
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


        # Load exaplainer (xAI post-hoc method)
        self.explainer = self.get_explainer(
            model,
            self.config["explainer_name"],
            self.config["explainer_config"],
            batch_size=self.config["batch_size"],
        )
        


        # Get base save folder to save results
        self._base_save_folder = experiment.save_dir

        self.valid_images = None
        if self.config["imgs_file"]:
            with open(self.config["imgs_file"], "r") as f:
                self.valid_images = json.load(f)


    # Method: Get loaded epoch
    def get_loaded_epoch(self) -> Optional[int]:
        return self._epoch


    # Method/Attribute: Save folder
    @property
    def save_folder(self) -> Path:
        save_folder = self._base_save_folder / self.get_save_folder(
            self.get_loaded_epoch()
        )
        if not save_folder.exists():
            save_folder.mkdir(parents=True, exist_ok=True)
        return save_folder


    # Method: Get the save folder
    def get_save_folder(self, epoch: Optional[int] = None):
        """
        'Computes' the folder in which to store the results.
        Args:
            epoch: currently evaluated epoch.

        Returns: Path to save folder.

        """
        if epoch is None:
            epoch = self.get_loaded_epoch()
        return os.path.join(
            "visual_explanations",
            "epoch_{}".format(epoch),
            self.config["explainer_name"],
            self.config["explainer_config"],
        )


    # Method: The method used to generate visual explanations
    def analysis(self):

        # Initialise results dictionary
        results = dict()

        # Get explainer (xAI post-hoc method)
        explainer = self.explainer

        # Get dataset name
        dataset_name = self.experiment.dataset

        # Get datamodules
        datamodule = self.experiment.get_datamodule()
        datamodule.setup("test", True)
        datamodule.batch_size = self.config["batch_size"]
        
        # Get dataloader and dataset
        dataloader = datamodule.test_dataloader()
        if dataset_name == "VinDrCXR":
            dataloader.dataset.return_bbox = True

        # Calculate the attributions for all classes that are participating
        img_idx = 0
        for batch in tqdm(dataloader, desc="Generating visual explanations"):
            if img_idx == 1000: break
            if dataset_name == "ISIC2018CINet":
                image, _, label, _, _, _, _, _, _, imgid = batch
            else:
                if dataset_name == "VinDrCXR":
                    image, label, bboxes, cnames, imgid = batch
                    bboxes = [tuple(b[0].tolist()) for b in bboxes]
                    cnames = list(np.array(cnames).flatten())
                else:
                    image, label, imgid = batch

            if self.valid_images and imgid[0] not in self.valid_images: continue
            
            # Get data
            image = image.to(self.device, non_blocking=True)
            
            # Adjust the shape of the labels according to the database
            if dataset_name == "CXR8" or dataset_name == "VinDrCXR":
                label = torch.argmax(label, dim=-1) 
            
            # Get target
            target = label.item()

            # Get attribution
            attribution = explainer.attribute(image, target)
            attribution = attribution.sum(1, keepdim=True)
            attribution = attribution.detach().cpu().numpy()

            # Get image
            image = image.detach().cpu().numpy()

            # Append to the results dictionary
            if dataset_name == "VinDrCXR":
                results[imgid[0]] = [image, attribution, target, bboxes, cnames]
            else:
                results[imgid[0]] = [image, attribution, target]

            # Update image index
            img_idx += 1


        return results


    # Method: Save results
    def save_results(self, results):
        save_dir: Path = self.experiment.save_dir / self.get_save_folder()
        save_dir.mkdir(parents=True, exist_ok=True)

        for img_idx, img_results in results.items():

            # get image, attributes and target
            image = img_results[0]
            attribution = img_results[1]
            target = img_results[2]

            # Create filenames
            image_fname = f"img_{img_idx}_{target}.npy"
            attribute_fname = f"att_{img_idx}_{target}.npy"

            # Save files
            np.save(
                file=os.path.join(save_dir, image_fname),
                arr=image,
                allow_pickle=True,
                fix_imports=True
            )

            np.save(
                file=os.path.join(save_dir, attribute_fname),
                arr=attribution,
                allow_pickle=True,
                fix_imports=True
            )

            if len(img_results) > 3:
                bboxes = img_results[3]
                cnames = img_results[4]
                bbox_fname = f"bbox_{img_idx}_{target}.npy"
                cnames_fname = f"cname_{img_idx}_{target}.npy"
                np.save(
                    file=os.path.join(save_dir, bbox_fname),
                    arr=np.array(bboxes),
                    allow_pickle=True,
                    fix_imports=True
                )
                np.save(
                    file=os.path.join(save_dir, cnames_fname),
                    arr=np.array(cnames),
                    allow_pickle=True,
                    fix_imports=True
                )  

        print(f"Saved results to '{save_dir}'")

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
    nargs = "+" if multiple_args else None

    parser = argparse.ArgumentParser(
        description="Generation of visual explanations analyser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )
    parser.add_argument(
        "--save_path", default=None, nargs=nargs, help="Path for model checkpoints."
    )
    parser.add_argument(
        "--reload",
        default="last",
        type=str,
        help="Which epoch to load. Options are 'last', 'best', 'best_any' and 'epoch_X',"
        "as long as epoch_X exists.",
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
        help="Used if reload='weights'. Loads the provided checkpoint"
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
        "--imgs_file",
        type=str,
        default=None,
        help="File with correctly classified images for this dataset."
    )

    parser.add_argument("--debug", action="store_true", default=False)
    
    return parser



# Function: Get the arguments from the CLI
def get_arguments():
    parser = argument_parser()
    opts = parser.parse_args()
    return opts



# Function: Run main
def main(config):
    print("Starting visual explanation generation for", config)
    print()
    print()

    experiment = Experiment(config.save_path)

    analyser = VisualExplanationsAnalyser(
        experiment,
        explainer_name=config.explainer_name,
        explainer_config=config.explainer_config,
        reload=config.reload,
        ema=config.ema,
        weights=config.weights,
        batch_size=1,
        imgs_file=config.imgs_file
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
