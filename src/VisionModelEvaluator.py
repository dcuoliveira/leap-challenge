import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

class VisionModelEvaluator:
    """
    Generic class for evaluating vision models on datasets with optional class translation.
    """

    def __init__(
            self,
            base_dataset: torch.utils.data.Dataset = None,
            class_index_file: str = None
    ):
        """
        Initialize the evaluator.

        Parameters
        ----------
        base_dataset : torch.utils.data.Dataset, optional
            Reference dataset for label mapping (e.g., Tiny ImageNet test set).
        class_index_file : str, optional
            Path to a JSON file mapping model class indices to WNIDs or human-readable labels.
            Example: ImageNet class index JSON file.
        """
        self.base_dataset = base_dataset
        self.base_class_idx = base_dataset.class_to_idx if base_dataset and hasattr(base_dataset, "class_to_idx") else None

        self.model_class_idx = None
        if class_index_file and os.path.exists(class_index_file):
            with open(class_index_file, 'r') as f:
                self.model_class_idx = json.load(f)

    def fit_vision_model(
            self,
            model: torch.nn.Module,
            test_dataset: torch.utils.data.Dataset,
            batch_size: int = 32,
            shuffle: bool = False,
            tag: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference of a pretrained vision model on a test dataset.

        Parameters
        ----------
        model : torch.nn.Module
            Pretrained vision model (e.g., ResNet101)
        test_dataset : torch.utils.data.Dataset
            Dataset for evaluation.
        batch_size : int, default=32
            DataLoader batch size.
        shuffle : bool, default=False
            Shuffle dataset in DataLoader.
        tag : float, default=0.0
            Optional tag (used for progress bar, e.g., noise level).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Predicted labels and ground-truth labels.
        """
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in tqdm(
                    test_loader, 
                    desc=f"Evaluating Model (tag={tag:.1f})", 
                    unit="batch"):
                images = images.to(device)
                outputs = model(images)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        return np.array(all_preds), np.array(all_targets)

    def translate_accuracy(
            self,
            all_preds: np.ndarray,
            all_targets: np.ndarray,
            target_dataset: torch.utils.data.Dataset = None
    ) -> float:
        """
        Compute accuracy after optionally translating model predictions to target dataset labels.

        Parameters
        ----------
        all_preds : np.ndarray
            Model predictions (indices in model's class space, e.g., 0-999 for ImageNet)
        all_targets : np.ndarray
            Ground truth labels (indices in target dataset)
        target_dataset : torch.utils.data.Dataset, optional
            Dataset to extract class mapping from. Defaults to base_dataset.

        Returns
        -------
        float
            Computed accuracy.
        """
        # If no mapping, compute direct accuracy
        if self.model_class_idx is None:
            return (all_preds == all_targets).mean()

        # Determine target class mapping
        target_class_idx = None
        if target_dataset is not None and hasattr(target_dataset, "class_to_idx"):
            target_class_idx = target_dataset.class_to_idx
        elif self.base_class_idx is not None:
            target_class_idx = self.base_class_idx
        else:
            raise ValueError("Target class mapping not available for accuracy translation.")

        new_target = []
        new_preds = []

        for pred_idx, gt_idx in zip(all_preds, all_targets):
            pred_label = self.model_class_idx[str(pred_idx)][0]
            gt_label = list(target_class_idx.keys())[gt_idx]

            if pred_label in target_class_idx:
                new_preds.append(target_class_idx[pred_label])
                new_target.append(target_class_idx[gt_label])

        new_preds = np.array(new_preds)
        new_target = np.array(new_target)
        return (new_preds == new_target).mean()
