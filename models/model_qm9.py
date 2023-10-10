import torch
import torch.distributions as dist
from escnn import gspaces

from models.model import SteerableCNN
from utils import BatchGeometricTensor
from common.blocks import EmbeddingBlock


class SteerableCNN_QM9(SteerableCNN):
    def __init__(self, gspace: gspaces, filter_type: str, non_linearity: str, 
                 n_g_blocks: int, num_nodes: list, num_classes: int, 
                 connectivity: str, channels: list, max_freq: list, node_dim: int, edge_dim: int, edge_att_type: str,
                 use_bn: int, x_dist: dist = None, attr_dist: dist = None, pool_type: str = 'mean', 
                 use_skipcons: bool = False, aggr: str = 'add', **kwargs):
        
        embed_block = EmbeddingBlock(
            non_linearity=non_linearity,
            channels=channels[0],
            L=max_freq[0],
            in_type=gspace.type(*(node_dim * [gspace.trivial_repr]))
        )
        
        super().__init__(embed_block, filter_type, non_linearity, n_g_blocks, num_nodes, num_classes, 
                 connectivity, channels, max_freq, edge_dim, edge_att_type, use_bn,
                 x_dist, attr_dist, pool_type, use_skipcons, aggr, **kwargs)

    def forward(self, pos_in: torch.Tensor, batch: torch.Tensor, 
                node_features: torch.Tensor = None, edge_index: torch.Tensor = None, edge_attr: torch.Tensor = None):
        """
        Forward pass of the network.

        Args:
            pos_in (torch.Tensor): Input positions of shape `(num_nodes, 3)`.
            batch (torch.Tensor): Batch vector of shape `(num_nodes, )`.
            node_features (torch.Tensor, optional): Node features of shape `(num_nodes, num_node_features)`.
                Defaults to None.
            edge_index (torch.Tensor, optional): Edge indices of shape `(2, num_edges)` for graph connectivity.
                Defaults to None.
            edge_attr (torch.Tensor, optional): Edge attributes of shape `(num_edges, num_edge_features)`.
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape `(num_samples, num_classes)`.
        """
        x = self.embed_block(node_features, pos_in)

        # induce connectivity if 'knn' or 'radius' passed
        if self.connectivity != "given":
            edge_index = None

        x = BatchGeometricTensor(
            x,
            batch,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        x = self.g_blocks(x)
        x, batch = self.inv_block(x)
        x = self.pooling(x, batch)
        x = self.fully_net(x)

        return x