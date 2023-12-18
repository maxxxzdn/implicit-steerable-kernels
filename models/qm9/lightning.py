import torch
import torchmetrics
import pytorch_lightning as pl
from torch_geometric.data import Data


class QM9Regressor(pl.LightningModule):
    """
    A PyTorch Lightning module for regression on the QM9 dataset.

    This module normalizes position and target values, computes forward passes,
    and calculates loss and metrics during training, validation, and testing.
    """

    def __init__(
        self, 
        model: torch.nn.Module, 
        lr: float, 
        target_mean: float, 
        target_std: float, 
        pos_mean: float, 
        pos_std: float,
        path_to_state_dict: str = None,
        verbose: bool = False
    ):
        """
        Initialize the QM9Regressor module.

        Args:
            model (torch.nn.Module): The regression model to be used.
            lr (float): Learning rate.
            target_mean (float): Mean of the target value for normalization.
            target_std (float): Standard deviation of the target value for normalization.
            pos_mean (float): Mean of positions for normalization.
            pos_std (float): Standard deviation of positions for normalization.
            path_to_state_dict (str, optional): Path to a pre-trained state dictionary.
            verbose (bool, optional): If True, enables verbose logging.
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss = torch.nn.L1Loss()
        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.val_metric = torchmetrics.MeanAbsoluteError()
        self.test_metric = torchmetrics.MeanAbsoluteError()
        self.target_mean = target_mean
        self.target_std = target_std
        self.pos_mean = pos_mean
        self.pos_std = pos_std
        self.path = path_to_state_dict
        self.verbose = verbose
                
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            data (Data): Input data containing positions, node features, and graph structure.

        Returns:
            torch.Tensor: Model predictions.
        """
        pos_in = (data.pos - self.pos_mean) / self.pos_std
        return self.model(data.x, pos_in, data.edge_index, data.edge_attr, data.batch)
    
    def _step(self, data: Data, stage: str) -> torch.Tensor:
        """
        A helper function for training, validation, and test steps.

        Args:
            data (Data): Batch of data.
            stage (str): Stage of the model ('train', 'val', or 'test').

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        batch_size = data.batch.max().item() + 1
        pred = self(data)
        target = (data.y - self.target_mean) / self.target_std
        loss = self.loss(pred, target)

        metric = getattr(self, f'{stage}_metric')
        metric((pred * self.target_std + self.target_mean) * 1000, data.y * 1000)
        self.log_dict(
            {
                f'{stage}/metric': metric,
                f'{stage}/loss': loss.item()
            },
            on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=self.verbose
        )
        return loss

    def training_step(self, data: Data, batch_idx: int):
        """
        Training step for the model.

        Args:
            data (Data): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        return self._step(data, 'train')
    
    def validation_step(self, data: Data, batch_idx: int):
        """
        Validation step for the model.

        Args:
            data (Data): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        return self._step(data, 'val')

    def test_step(self, data: Data, batch_idx: int):
        """
        Test step for the model.

        Args:
            data (Data): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        return self._step(data, 'test')
    
    def configure_optimizers(self):
        """
        Configures optimizers and learning rate schedulers.

        Returns:
            dict: Dictionary containing optimizer and LR scheduler configurations.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-8)
        if self.path is not None:
            try:
                optimizer.load_state_dict(torch.load(self.path)['optimizer_state_dict'])
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load optimizer state: {e}")

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75, 100, 125], gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train/metric"}}
