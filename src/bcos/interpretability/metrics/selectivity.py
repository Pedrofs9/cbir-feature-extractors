import numpy as np

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

# Torch imports.
import torch
from torch.nn.functional import conv2d

# Quantus imports.
from quantus.helpers import warn
from quantus.helpers import utils
from quantus.helpers import plotting
from quantus import BatchedPerturbationMetric
from quantus.functions.normalise_func import normalise_by_max
from quantus.helpers.model.model_interface import ModelInterface
from quantus.functions.perturb_func import baseline_replacement_by_indices
from quantus.helpers.enums import (
    ModelType,
    DataType,
    ScoreDirection,
    EvaluationCategory,
)


class Selectivity(BatchedPerturbationMetric):
    """
    Implementation of Selectivity test by Montavon et al., 2018.

    At each iteration, a patch (e.g., of size 4 x 4) corresponding to the region with
    highest relevance is set to black. The plot keeps track of the function value
    as the features are being progressively removed and computes an average over
    a large number of examples.

    Assumptions:
        - The original metric definition relies on perturbation functionality suited only for images.
        Therefore, only apply the metric to 3-dimensional (image) data. To extend the applicablity
        to other data domains, adjustments to the current implementation might be necessary.

    References:
        1) Grégoire Montavon et al.:
        "Methods for interpreting and understanding deep neural networks."
        Digital Signal Processing 73 (2018): 1-15.

    Attributes:
        -  _name: The name of the metric.
        - _data_applicability: The data types that the metric implementation currently supports.
        - _models: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
        - evaluation_category: What property/ explanation quality that this metric measures.
    """

    name = "Selectivity"
    data_applicability = {DataType.IMAGE}
    model_applicability = {ModelType.TORCH, ModelType.TF}
    score_direction = ScoreDirection.LOWER
    evaluation_category = EvaluationCategory.FAITHFULNESS

    def __init__(
        self,
        patch_size: int = 8,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,
        perturb_baseline: str = "black",
        perturb_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Callable = np.mean,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        patch_size: integer
            The patch size for masking, default=8.
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=False.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func: callable
            Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline: string
            Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        perturb_func_kwargs: dict
            Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call.
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        kwargs: optional
            Keyword arguments.
        """
        if normalise_func is None:
            normalise_func = normalise_by_max

        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices
        perturb_func = perturb_func

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

        if default_plot_func is None:
            default_plot_func = plotting.plot_selectivity_experiment

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Save metric-specific attributes.
        self.patch_size = patch_size

        # Asserts and warnings.
        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and the patch size for masking"
                    " 'patch_size'"
                ),
                data_domain_applicability=(
                    f"Also, the current implementation only works for 3-dimensional (image) data."
                ),
                citation=(
                    "Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. 'Methods for "
                    "interpreting and understanding deep neural networks.' Digital Signal "
                    "Processing 73 (2018): 1-15"
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = True,
        device: Optional[str] = None,
        batch_size: int = 64,
        custom_batch: Optional[Any] = None,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to evaluation_scores.
        Calls custom_postprocess() afterwards. Finally returns evaluation_scores.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        explain_func: callable
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        device: string
            Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        kwargs: optional
            Keyword arguments.

        Returns
        -------
        evaluation_scores: list
            a list of Any with the evaluation scores of the concerned batch.

        Examples:
        --------
            # Minimal imports.
            >> import quantus
            >> from quantus import LeNet
            >> import torch

            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/pytests/mnist_model"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = Metric(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency}
        """
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            softmax=softmax,
            device=device,
            batch_size=batch_size,
            custom_batch=custom_batch,
            **kwargs,
        )

    def evaluate_batch(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        s_batch: np.ndarray,
    ) -> list[list[float]]:
        # Rename variables.
        x = x_batch
        y = y_batch
        a = a_batch
        patch_size = self.patch_size

        # x has shape [channel, height, width] or [batch, channel, height, width]
        # Add batch dimension if necessary.
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        if a.ndim == 3:
            a = np.expand_dims(a, axis=0)

        assert (
            x.shape[0] == a.shape[0] == y.shape[0]
        ), "Number of instances in x, a and y must match."

        # Get dimensions.
        _, _, height, width = a.shape

        # Aggregate saliency scores for each patch.
        # To avoid a python loop, we use a convolution with a kernel of ones
        # to calculate the sum of each patch.
        attribution_sums = conv2d(
            torch.from_numpy(a).float(),
            torch.ones(1, 1, patch_size, patch_size).float(),
            padding=(self.patch_size, self.patch_size),
        )

        # Remove padding.
        attribution_sums = attribution_sums[
            :, 0, patch_size : patch_size + height, patch_size : patch_size + width
        ]
        attribution_sums = attribution_sums.numpy()

        # Store perturbed images and batch indices.
        perturbed_images = []
        batch_indices = []

        # Make perturbed images for each image in the batch.
        for batch_idx in range(x.shape[0]):
            # Get the attention map and image for the current batch index.
            batch_attribution_sums = attribution_sums[batch_idx].copy()
            perturbed_image = x[batch_idx].copy()

            # Pad image
            perturbed_image = np.pad(
                perturbed_image,
                ((0, 0), (patch_size, patch_size), (patch_size, patch_size)),
                mode="constant",
                constant_values=0,
            )

            # Find next patch to perturb while there are still non-overlapping patches to available.
            while True:
                # Find the patch with the maximum saliency score.
                i, j = np.unravel_index(
                    np.argmax(batch_attribution_sums), (height, width)
                )

                # If the maximum attention score is -inf, then there are no more patches to perturb.
                if batch_attribution_sums[i, j] == -np.inf:
                    break

                # Remove attribution scores for the current patch and all overlapping patches.
                batch_attribution_sums[
                    max(0, i - patch_size + 1) : i + patch_size,
                    max(0, j - patch_size + 1) : j + patch_size,
                ] = -np.inf

                # Create slice for patch.
                patch_slice = utils.create_patch_slice(
                    patch_size=patch_size,
                    coords=(i + patch_size, j + patch_size),
                )

                # Perturb the current patch (set to black).
                perturbed_image = self.perturb_func(
                    arr=perturbed_image.copy(),
                    indices=patch_slice,
                    indexed_axes=self.a_axes,
                    **self.perturb_func_kwargs,
                )

                # Store perturbed image and batch index.
                perturbed_images.append(
                    perturbed_image[:, patch_size:-patch_size, patch_size:-patch_size]
                )
                batch_indices.append(batch_idx)

        # Stack perturbed images and convert to tensor.
        perturbed_images = np.stack(perturbed_images)
        perturbed_labels = y[batch_indices]

        # Get predictions for perturbed images.
        results = []
        batch_size = x.shape[0]

        for batch_start_index in range(0, len(perturbed_images), batch_size):
            batch_images = perturbed_images[
                batch_start_index : batch_start_index + batch_size
            ]
            batch_labels = perturbed_labels[
                batch_start_index : batch_start_index + batch_size
            ]

            batch_images = model.shape_input(
                batch_images, batch_images.shape, channel_first=True, batched=True
            )
            batch_predictions = model.predict(batch_images)
            batch_predictions = batch_predictions[
                np.arange(batch_images.shape[0]), batch_labels
            ]
            results.extend(batch_predictions.tolist())

        # Group results by batch index.
        results_by_batch_index = defaultdict(list)
        for batch_index, result in zip(batch_indices, results):
            results_by_batch_index[batch_index].append(result)

        return [
            results_by_batch_index[batch_index] for batch_index in range(x.shape[0])
        ]

    @property
    def get_auc_score(self, aggregate=False):
        """Calculate the area under the curve (AUC) score for several test samples."""
        scores = [utils.calculate_auc(np.array(curve)) for curve in self.evaluation_scores]
        if aggregate:
            return np.mean(scores)
        return scores


