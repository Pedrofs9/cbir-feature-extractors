# Imports
import os
import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.nn import functional as F
import pandas as pd
import cv2

# Function: Arguments for the CLI
def arguments_cli(add_help=True):

    # Create parser
    parser = argparse.ArgumentParser(
        description="Plot of visual explanations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )

    # Add arguments
    parser.add_argument("--model_type", required=True, type=str, choices=["baseline", "bcos"], help="The type of model you are visualising (relevant for the normalisation of the images).")
    parser.add_argument("--explanations_path", type=str, required=True)
    parser.add_argument("--bbox", action="store_true")
    parser.add_argument("--debug", action="store_true", default=False)

    return parser



# Function: Get arguments
def get_arguments():

    # Get parser and arguments
    parser = arguments_cli()
    args = parser.parse_args()
    
    return args



# Function: Main
def main(args):

    # Get args values
    model_type = args.model_type
    explanations_path = args.explanations_path

    # Load files in the explanations path
    imgs_and_attributions = [f for f in os.listdir(explanations_path) if not f.startswith('.')]
    imgs = [i for i in imgs_and_attributions if i.split('_')[0] == "img"]

    # Generate plots
    for img_fname in tqdm(imgs):

        # Get attribution fname
        att_fname = img_fname.replace("img", "att")

        # Open files
        img = np.load(
            os.path.join(explanations_path, img_fname),
            allow_pickle=True,
            fix_imports=True
        )

        att = np.load(
            os.path.join(explanations_path, att_fname),
            allow_pickle=True,
            fix_imports=True
        )

        if args.bbox:
            bboxes = np.load(
                os.path.join(explanations_path, att_fname.replace("att", "bbox")),
                allow_pickle=True,
                fix_imports=True
            )
            cnames = np.load(
                os.path.join(explanations_path, att_fname.replace("att", "cname")),
                allow_pickle=True,
                fix_imports=True
            )
        

        # Remove batch axis
        img = np.squeeze(img, axis=0)
        att = np.squeeze(att, axis=0)

        att = F.avg_pool2d(
                    torch.from_numpy(att), 15, stride=1, padding=(15 - 1) // 2
                )

        # Check what type of normalisation was applied to the image
        if model_type == "bcos":
            
            # Images
            # We only want the RGB channels
            img = img[0:3, :, :]

            # And then transpose from [C, H, W] to [H, W, C]
            img = np.transpose(img, (1, 2, 0))
            # print(img.shape)
            img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')

            # Attributions
            if len(att.shape) == 3:
                att = np.transpose(att, (1, 2, 0))

                if att.shape[2] == 1:
                    att = np.squeeze(att, axis=2)
            # print(att.shape)

        else:
            
            # Images
            # Transpose from [C, H, W] to [H, W, C]
            img = np.transpose(img, (1, 2, 0))
            img_ = img.copy()
            # Unnormalise
            MEAN = [0.485, 0.456, 0.406]
            STD = [0.229, 0.224, 0.225]

            for c in range(img.shape[2]):
                img[:,:, c] = (img_[:, :, c] * STD[c]) + MEAN[c]
            
            # Scale the pixel range of the array
            img_ = img.copy()
            img = ((img_ - img_.min()) * (1/(img_.max() - img_.min()) * 255)).astype('uint8')

            # Attributions
            if len(att.shape) == 3:
                att = np.transpose(att, (1, 2, 0))

                if att.shape[2] == 1:
                    att = np.squeeze(att, axis=2)
            # print(att.shape)
        att = att.cpu().numpy()
        att = ((att - att.min()) * (1/(att.max() - att.min()) * 255)).astype('uint8')


        # Get plot fname
        plt_fname = img_fname.replace("img", "plt")
        plt_fname = plt_fname.replace("npy", "png")

        plt.imshow(img)
        plt.imshow(att, cmap='seismic', alpha=0.4)
        plt.axis('off')

        # Plots folder
        plots_dir = os.path.join(explanations_path, "plots")
        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)
        
        plt.savefig(
            fname=os.path.join(plots_dir, plt_fname),
            bbox_inches="tight",
            pad_inches=0.0
        )

        if args.bbox:
            if "No finding" in cnames:
                continue
            for c, b in zip(cnames, bboxes):
                img = cv2.rectangle(cv2.UMat(img), (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (36,255,12), 2)
                img = cv2.putText(img, c, (int(b[0]), int(b[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                img = img.get()

        plt.imshow(img)
        plt.axis('off')
        plt.savefig(
            fname=os.path.join(plots_dir, plt_fname.replace(".png", "_og.png")),
            bbox_inches="tight",
            pad_inches=0.0
        )
        
        plt.close('all')

    return



# Usage
if __name__ == "__main__":
    
    # Get arguments
    args = get_arguments()

    try:
        main(args)
    
    except Exception:
        if args.debug:
            pdb.post_mortem()
        raise
