import torch
from typing import Optional
from torch_geometric.transforms import NormalizeScale, SamplePoints
from torch_geometric.datasets import ModelNet
from torch.utils.data import random_split

from utils.transforms import *
from .dataset import DataModule, DatasetAug


class MNDataModule(DataModule):
    """
    Data module for MNIST dataset.

    Args:
        data_dir (str): The directory where the dataset is stored.
        batch_size (int): The batch size for training and testing.
        name (str): The name of the dataset. Must be either '10' or '40'.
        num_nodes (int): The number of nodes for sampling points.
        train_transform (Optional[dict]): Optional dictionary of transformations for training data.
        test_transform (Optional[dict]): Optional dictionary of transformations for testing data.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        name: str,
        num_nodes: int,
        train_transform: Optional[dict] = None,
        test_transform: Optional[dict] = None,
    ):
        super().__init__(data_dir, batch_size, name, train_transform, test_transform)
        assert name in ["10", "40"]
        self.pre_transformation = NormalizeScale()
        self.transformation = SamplePoints(num_nodes)

    def setup(self, stage: Optional[str] = None):
        """
        Set up the dataset for training and testing.
        For testing, 40 augmentations of the same dataset form the test dataset.

        Args:
            stage (Optional[str]): The stage of setup. Must be either "fit" or "test".
        """
        if stage == "fit" or stage is None:
            data_full = ModelNet(
                self.data_dir,
                self.name,
                train=True,
                pre_transform=self.pre_transformation,
                transform=self.transformation,
            )
            train_size = int(0.9 * len(data_full))
            val_size = len(data_full) - train_size
            data_train, data_val = random_split(
                data_full,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

            self.data_train = DatasetAug(data_train, self._train_transforms)
            self.data_val = DatasetAug(data_val, self._test_transforms)

            print("Dataset used: " + str(data_full))

        if stage == "test" or stage is None:
            data_test = ModelNet(
                self.data_dir,
                self.name,
                train=False,
                pre_transform=self.pre_transformation,
                transform=self.transformation,
            )

            # 40 augmentations for testing, the average is reported
            self.data_test = torch.utils.data.ConcatDataset(
                40 * [DatasetAug(data_test, self._test_transforms)]
            )
