import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
from torch_geometric.data import Data


class Model(pl.LightningModule):
    """
    Base class for all models.
    """
    def __init__(self, model: torch.nn.Module, lr: float, path: str = None, **kwargs):
        """
        Args:
            model: model to train
            lr: learning rate
            path: path to model checkpoint
        """
        super().__init__(**kwargs)
        self.model = model
        self.lr = lr
        self.path = path
        if self.path is not None:
            try:
                model.load_state_dict(torch.load(self.path)['model_state_dict'], strict=False)
            except:
                print('Could not load model checkpoint, continue training from scratch.')

    def forward(self, data: Data) -> torch.Tensor:
        pass

    def training_step(self, data: Data, batch_idx: int):
        """
        Training step of a pytorch lightning module.
        Predict on batch of data, calculate loss and log metrics.
        Args:
            data: batch of data
            batch_idx: batch index
        Output:
            loss: loss value
        """
        batch_size = data.batch.max().item() + 1
        pred = self(data)
        loss = self.loss(pred, data.y)
        self.train_metric(pred, data.y)
        self.log_dict(
            {
                'train/metric': self.train_metric, 
                'train/loss': loss.item()
            }, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return loss

    def validation_step(self, data: Data, batch_idx: int):
        """
        Validation step of a pytorch lightning module.
        Predict on batch of data, calculate loss and log metrics.
        Args:
            data: batch of data
            batch_idx: batch index
        Output:
            loss: loss value
        """
        batch_size = data.batch.max().item() + 1
        pred = self(data)
        loss = self.loss(pred, data.y)
        self.val_metric(pred, data.y)
        self.log_dict(
            {
                'val/metric': self.val_metric, 
                'val/loss': loss.item()
            }, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss
    
    def test_step(self, data: Data, batch_idx: int):
        """
        Test step of a pytorch lightning module.
        Predict on batch of data, calculate loss and log metrics.
        Args:
            data: batch of data
            batch_idx: batch index
        Output:
            loss: loss value
        """
        batch_size = data.batch.max().item() + 1
        pred = self(data)
        loss = self.loss(pred, data.y)
        self.test_metric(pred, data.y)
        self.log_dict(
            {
                'test/metric': self.test_metric, 
                'test/loss': loss.item()
            }, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss
    
    def configure_optimizers(self):
        pass
    

class ModelNetClassifier(Model):
    """
    Classifier for ModelNet40 dataset.
    """
    def __init__(self, model: torch.nn.Module, lr: float, **kwargs):
        """
        Args:
            model: classification model
            lr: learning rate
        """
        super().__init__(model, lr, **kwargs)
        self.loss = self.smooth_loss
        self.train_metric = torchmetrics.Accuracy()
        self.val_metric = torchmetrics.Accuracy()
        self.test_metric = torchmetrics.Accuracy()
        
    def forward(self, data: Data) -> torch.Tensor:
        return self.model(data.pos, data.batch)
    
    def configure_optimizers(self):
        """
        Optimizer and learning rate scheduler for pytorch lightning.
        We use AdamW optimizer with LR reducing by 0.5 every 25 epochs.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,75], gamma=0.5)
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler, 
                    "monitor": "train/metric"
                }
               }
    
    @staticmethod
    def smooth_loss(input, target, smoothing=True, eps = 0.2):
        """
        Calculate cross entropy loss, apply label smoothing if needed.
        Used in "Dynamic Graph CNN for Learning on Point Clouds"
        Article: https://arxiv.org/abs/1801.07829
        Source: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/util.py
        """
        target = target.contiguous().view(-1)
        if smoothing:
            n_class = input.size(1)
            one_hot = torch.zeros_like(input).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(input, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(input, target, reduction='mean')
        return loss
    
    
class QM9Regressor(Model):
    """
    Regressor for QM9 dataset.
    """
    def __init__(self, 
            model: torch.nn.Module, 
            lr: float, 
            target_mean: float, 
            target_std: float, 
            pos_mean: float, 
            pos_std: float, 
            **kwargs):
        """
        Args:
            model: regression model
            lr: learning rate
            target_mean: mean of target value
            target_std: std of target value 
            pos_mean: mean of positions 
            pos_std: std of positions 
        Note: 
            We normalize positions to zero mean and unit variance.
            We normalize target value to zero mean and unit variance when training.
        """
        super().__init__(model, lr, **kwargs)
        self.loss = torch.nn.L1Loss()
        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.val_metric = torchmetrics.MeanAbsoluteError()
        self.test_metric = torchmetrics.MeanAbsoluteError()
        self.target_mean = target_mean
        self.target_std = target_std
        self.pos_mean = pos_mean
        self.pos_std = pos_std
                
    def forward(self, data: Data) -> torch.Tensor:
        # normalize positions
        pos_in = (data.pos - self.pos_mean) / self.pos_std
        return self.model(
            pos_in=pos_in, 
            batch=data.batch, 
            node_features=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr)
    
    def training_step(self, data: Data, batch_idx: int):
        batch_size = data.batch.max().item() + 1
        pred = self(data)
        # normalize target value
        target = (data.y - self.target_mean)/self.target_std
        loss = self.loss(pred, target)
        # report metric computed on unnormalized target value
        # multiply by 1000 = eV -> meV
        self.train_metric((pred*self.target_std+self.target_mean)*1000, 
                          data.y*1000)
        self.log_dict(
            {
                'train/metric': self.train_metric, 
                'train/loss': loss.item()
            }, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, data: Data, batch_idx: int):
        batch_size = data.batch.max().item() + 1
        pred = self(data)
        # normalize target value
        target = (data.y - self.target_mean)/self.target_std
        loss = self.loss(pred, target)
        # report metric computed on unnormalized target value
        # multiply by 1000 = eV -> meV
        self.val_metric((pred*self.target_std+self.target_mean)*1000, 
                        data.y*1000)
        self.log_dict(
            {
                'val/metric': self.val_metric, 
                'val/loss': loss.item()
            }, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def test_step(self, data: Data, batch_idx: int):
        batch_size = data.batch.max().item() + 1
        pred = self(data)
        # normalize target value
        target = (data.y - self.target_mean)/self.target_std
        loss = self.loss(pred, target)
        # report metric computed on unnormalized target value
        # multiply by 1000 = eV -> meV
        self.test_metric((pred*self.target_std+self.target_mean)*1000, 
                        data.y*1000)
        self.log_dict(
            {
                'test/metric': self.test_metric, 
                'test/loss': loss.item()
            }, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss
    
    def configure_optimizers(self):
        """
        Optimizer and learning rate scheduler for pytorch lightning.
        We use AdamW optimizer with LR reducing by 0.5 every 25 epochs.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-8)
        if self.path is not None:
            try:
                optimizer.load_state_dict(torch.load(self.path)['optimizer_state_dict'])
            except:
                pass
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,75,100,125], gamma=0.5)
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler, 
                    "monitor": "train/metric"
                }
               }
    

class ManyBodyClassifier(Model):
    """
    Classifier for N-body dataset.
    """
    def __init__(self, model: torch.nn.Module, lr: float, **kwargs):
        """
        Args:
            model: classification model
            lr: learning rate
        """
        super().__init__(model, lr, **kwargs)
        self.loss = torch.nn.L1Loss()
        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.val_metric = torchmetrics.MeanAbsoluteError()
        self.test_metric = torchmetrics.MeanAbsoluteError()
        
    def forward(self, data: Data) -> torch.Tensor:
        return self.model(pos_in=data.pos, 
                          batch=data.batch, 
                          node_features=data.x,
                          edge_index=data.edge_index,
                          edge_attr=data.edge_attr)
    
    def configure_optimizers(self):
        """
        Optimizer and learning rate scheduler for pytorch lightning.
        We use AdamW optimizer with LR reducing by 0.5 every 25 epochs.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,75], gamma=0.5)
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler, 
                    "monitor": "train/metric"
                }
               }
    