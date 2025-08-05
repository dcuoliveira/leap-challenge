import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

class AdversarialDatasetUtility:
    """
    Utility for creating adversarial/noisy datasets by applying noise functions to images.
    """

    @staticmethod
    def add_noise_to_random_pixels(
            image: torch.Tensor,
            num_noise_pixels: int = 1000,
            noise_mean: float = 0.0,
            noise_std: float = 0.1
    ) -> torch.Tensor:
        """
        Adds Gaussian noise to a fixed number of random pixels in a (C,H,W) tensor image.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor of shape (C, H, W) with pixel values in [0, 1].
        num_noise_pixels : int, default=1000
            Number of random pixels to add noise to.
        noise_mean : float, default=0.0
            Mean of the Gaussian noise.
        noise_std : float, default=0.1
            Standard deviation of the Gaussian noise.

        Returns
        -------
        torch.Tensor
            Noisy image tensor of shape (C, H, W).
        """
        C, H, W = image.shape
        noisy_image = image.clone()

        # Cap number of noisy pixels to image size
        num_noise_pixels = min(num_noise_pixels, H * W)

        # Random pixel indices
        pixel_indices = torch.randperm(H * W)[:num_noise_pixels]
        rows = pixel_indices // W
        cols = pixel_indices % W

        # Gaussian noise for all channels
        noise = torch.randn((num_noise_pixels, C), dtype=image.dtype) * noise_std + noise_mean
        noisy_image[:, rows, cols] += noise.T

        return noisy_image

    @staticmethod
    def create_noisy_dataset(
            dataset: torch.utils.data.Dataset,
            noise_fn: callable = None,
            **noise_kwargs
    ) -> torch.utils.data.Dataset:
        """
        Create a noisy version of a dataset by applying a noise function to each image.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Original dataset (assumed to return (image, label) tuples).
        noise_fn : callable, optional
            Function to apply noise to each image. Should take a torch.Tensor image and return a noisy image.
            If None, defaults to `add_noise_to_random_pixels`.
        **noise_kwargs
            Additional keyword arguments to pass to the noise function.

        Returns
        -------
        torch.utils.data.Dataset
            New dataset with noisy images and original labels.
        """
        if noise_fn is None:
            noise_fn = AdversarialDatasetUtility.add_noise_to_random_pixels

        noisy_images = []
        labels = []

        for img_tensor, label in tqdm(dataset, desc="Creating noisy dataset"):
            noisy_img = noise_fn(img_tensor, **noise_kwargs)
            noisy_images.append(noisy_img)
            labels.append(label)

        noisy_images = torch.stack(noisy_images)
        labels = torch.tensor(labels)

        return TensorDataset(noisy_images, labels)
