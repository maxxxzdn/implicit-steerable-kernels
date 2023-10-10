import torch
import numpy as np
import pytorch_lightning as pl
from typing import Optional
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomTranslate, RandomShear, RandomScale, NormalizeScale, SamplePoints
from torch_geometric.datasets import ModelNet, QM9
from torch.utils.data import random_split
import torch.nn.functional as F

from ..utils.transforms import *


class DatasetAug(Dataset):
    """
    Dataset with augmentation.
    """
    def __init__(self, dataset: ModelNet, transforms: SequentialTransformation):
        """
        Args:
            dataset (torch_geometric.datasets.ModelNet): The dataset to augment.
            transforms (SequentialTransformation): The sequence of transformations to apply to each point cloud.
        """
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        
    def augment(self, data: Data):
        return self.transforms(data)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        if self.transforms is None:
            return self.dataset[index]
        else:
            return self.augment(self.dataset[index])
        

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, name: str,
                 train_transform: Optional[dict] = None, test_transform: Optional[dict] = None):
        """Creates PL data module for ModelNet / QM9 datasets from PyG.
        
        The ModelNet10/40 datasets from the `"3D ShapeNets: A Deep Representation for Volumetric Shapes"
        <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,
        containing CAD models of 10 and 40 categories, respectively.
        
        The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
        Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
        about 130,000 molecules with 19 regression targets.
        Each molecule includes complete spatial information for the single low
        energy conformation of the atoms in the molecule.
        In addition, we provide the atom features from the `"Neural Message
        Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.
        
        Args:
            data_dir (str): Root directory where the dataset should be saved.
            batch_size (int): How many point clouds per batch to load in data loaders.
            name (str): The name of the dataset "10" for ModelNet10, "40" for ModelNet40.
            num_nodes (int): [for ModelNet] The number of points in each point cloud (randomly sampled at each call).
            train_transform (dict, optional): a dictionary defining the set of transformations 
                to apply to each point cloud in the training dataset before every access.
            test_transform (dict, optional): a dictionary defining the set of transformations 
                to apply to each point cloud in the val/test/predict dataset before every access.
                
        Transform dictionaries might containg following keys:
            translate (float): maximum translation in each dimension. 
                transformation: torch_geometric.transforms.RandomJitter
            scale (tuple(float, float)): scaling factor interval.
                transformation: torch_geometric.transforms.RandomScale
            shear (float): maximum shearing factor.
                transformation: torch_geometric.transforms.RandomShear
            rotate_axes (list): axes to rotate a point cloud around by random degree.
                transformation: utils.transforms.RandomRotateAxes
            If a key is not given, the corresponding transformation will not be applied.
        """
        super().__init__()
        assert name in ['10', '40', 'QM9', 'atom', 'residue']
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.name = name 
        
        self._train_transforms = self._transform(train_transform)
        self._test_transforms = self._transform(test_transform)
        
    def _transform(self, transform_args: Optional[dict] = None):
        """
        Creates a sequence of transformations to apply to each point cloud.

        Args:
            transform_args (dict): The dictionary of transformation arguments.
                - translate (float): maximum translation in each dimension.
                - scale (tuple(float, float)): scaling factor interval.
                - shear (float): maximum shearing factor.
                - rotate_axes (list): axes to rotate a point cloud around by random degree.

        Returns:
            SequentialTransformation: The sequence of transformations to apply to each point cloud.
        """
        if transform_args is None:
            transform = None
        else:
            transform_args.setdefault('translate', None)
            transform_args.setdefault('scale', None)
            transform_args.setdefault('shear', None)
            transform_args.setdefault('rotate_axes', None)
            transforms = []
            if transform_args['translate'] is not None:
                transforms.append(RandomTranslate(transform_args['translate'])) #0.1
            if transform_args['scale'] is not None:
                transforms.append(RandomScale(transform_args['scale'])) #2/3 3/2
            if transform_args['shear'] is not None:
                transforms.append(RandomShear(transform_args['shear'])) #0.1
            if transform_args['rotate_axes'] is not None:
                transforms.append(RandomRotateAxes(transform_args['rotate_axes'])) #-1
            transform = SequentialTransformation(transforms)
        return transform
             
    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size)
