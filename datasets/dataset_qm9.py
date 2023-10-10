import numpy as np
from typing import Optional

from ..utils.transforms import *
from ..utils.segnn_data import *
from .dataset import DataModule, DatasetAug

feature_dict = {'mu': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 
                'delta': 4, 'r2': 5, 'ZPVE': 6, 'U0': 12, 
                'U': 13, 'H': 14, 'G': 15, 'cv': 11}


class QM9DataModule(DataModule):
    def __init__(self, data_dir: str, batch_size: int, name: str,
                 train_transform: Optional[dict] = None, test_transform: Optional[dict] = None,
                 target: str = 'homo', feature_type: str = 'one_hot'):
        super().__init__(data_dir, batch_size, name, train_transform, test_transform)
        assert name == 'QM9'
        target_idx = feature_dict[target]
        self.pre_transformation = None
        self.transformation = SequentialTransformation(
            [
                TargetSubset([target_idx]), #only one target is used for training/inference.
                NodeFeatSubset([0,1,2,3,4]), #only one hot encoding of atom type is used as node features.
                #RemoveEdges() # Brandstetter et al. do not use edge information for training/inference.
            ]
        )
        
        self.indices = self.get_indices()
        train_data = QM9(root = self.data_dir)[self.indices['train']]
        self.target_mean = train_data.mean(target_idx)
        self.target_std = train_data.std(target_idx)
        self.pos_mean = -0.02
        self.pos_std = 1.7415
        
        
    def get_indices(self):       
        Nmols = 130831 #total number of molecules in pyg.QM9 dataset
        Ntrain = 100000 #number of points to train on
        Ntest = int(0.1*Nmols) #10% go to validation
        Nvalid = Nmols - (Ntrain + Ntest)

        np.random.seed(0)
        data_perm = np.random.permutation(Nmols)
        train, valid, test = np.split(data_perm, [Ntrain, Ntrain+Nvalid])
        return {"train": train, "val": valid, "test": test}

    def setup(self, stage: Optional[str] = None):
        data_full = QM9(root = self.data_dir,
                pre_transform = self.pre_transformation, 
                transform = self.transformation)
        
        if stage == "fit" or stage is None:            
            data_train = data_full[self.indices['train']]
            data_val = data_full[self.indices['val']]

            self.data_train = DatasetAug(data_train, self._train_transforms)
            self.data_val = DatasetAug(data_val, self._test_transforms)
            
        if stage == "test" or stage is None:
            data_test = data_full[self.indices['test']]

            self.data_test = DatasetAug(data_test, self._test_transforms)