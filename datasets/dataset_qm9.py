import numpy as np
from typing import Optional
from torch_geometric.datasets import QM9

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.transforms import *
from .dataset import DataModule, DatasetAug

feature_dict = {
    "mu": 0,
    "alpha": 1,
    "homo": 2,
    "lumo": 3,
    "delta": 4,
    "r2": 5,
    "ZPVE": 6,
    "U0": 12,
    "U": 13,
    "H": 14,
    "G": 15,
    "cv": 11,
}


class QM9DataModule(DataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        name: str,
        train_transform: Optional[dict] = None,
        test_transform: Optional[dict] = None,
        target: str = "homo",
    ):
        """
        Initializes the QM9DataModule class.

        Args:
        - data_dir (str): The directory where the QM9 dataset is stored.
        - batch_size (int): The batch size for training and testing.
        - name (str): The name of the dataset.
        - train_transform (Optional[dict]): Optional dictionary specifying the transformations to apply to the training data.
        - test_transform (Optional[dict]): Optional dictionary specifying the transformations to apply to the testing data.
        - target (str): The target feature to use for training and inference.
        """
        super().__init__(data_dir, batch_size, name, train_transform, test_transform)
        assert name == "QM9"
        target_idx = feature_dict[target]
        self.pre_transformation = None
        self.transformation = SequentialTransformation(
            [
                TargetSubset(
                    [target_idx]
                ),  # only one target is used for training/inference.
                NodeFeatSubset(
                    [0, 1, 2, 3, 4]
                ),  # only one hot encoding of atom type is used as node features.
                # RemoveEdges() # Brandstetter et al. do not use edge information for training/inference.
            ]
        )

        self.indices = self.get_indices()
        self.target_mean, self.target_std = self.get_preprocess_stats(target_idx)

        self.pos_mean = -0.02
        self.pos_std = 1.7415

    def get_preprocess_stats(self, target_idx):
        """
        Computes the mean and standard deviation of the target feature in the training data.

        Args:
        - target_idx (int): The index of the target feature.

        Returns:
        - target_mean (float): The mean of the target feature.
        - target_std (float): The standard deviation of the target feature.
        """
        train_data = QM9(root=self.data_dir)[self.indices["train"]]
        target_mean = train_data.mean(target_idx)
        target_std = train_data.std(target_idx)
        return target_mean, target_std

    def get_indices(self):
        """
        Computes the indices for training, validation, and testing data.

        Returns:
        - indices (dict): A dictionary containing the indices for training, validation, and testing data.
        """
        Nmols = 130831  # total number of molecules in pyg.QM9 dataset
        Ntrain = 100000  # number of points to train on
        Ntest = int(0.1 * Nmols)  # 10% go to validation
        Nvalid = Nmols - (Ntrain + Ntest)

        np.random.seed(0)
        data_perm = np.random.permutation(Nmols)
        train, valid, test = np.split(data_perm, [Ntrain, Ntrain + Nvalid])
        return {"train": train, "val": valid, "test": test}

    def setup(self, stage: Optional[str] = None):
        """
        Sets up the data for training and testing.

        Args:
        - stage (Optional[str]): The stage of setup, either "fit" for training or "test" for testing.
        """
        data_full = QM9(
            root=self.data_dir,
            pre_transform=self.pre_transformation,
            transform=self.transformation,
        )

        if stage == "fit" or stage is None:
            data_train = data_full[self.indices["train"]]
            data_val = data_full[self.indices["val"]]

            self.data_train = DatasetAug(data_train, self._train_transforms)
            self.data_val = DatasetAug(data_val, self._test_transforms)

        if stage == "test" or stage is None:
            data_test = data_full[self.indices["test"]]

            self.data_test = DatasetAug(data_test, self._test_transforms)
