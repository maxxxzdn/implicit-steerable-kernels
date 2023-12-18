import torch
from escnn import gspaces, group
from torch_geometric.nn import global_mean_pool

from models.qm9.blocks import EmbeddingBlock, ResidualBlock, InvariantMap

class SteerableCNN_QM9(torch.nn.Module):
    """
    Steerable CNN for QM9 dataset.

    This model comprises an embedding block, a series of residual blocks, an invariant mapping,
    and a fully connected layer for final predictions.
    """
    def __init__(
        self, 
        gspace: gspaces.GSpace, 
        num_layers: int, 
        num_features: int, 
        num_inv_features: int, 
        mlp_params: dict, 
        L: int = 1, 
        edge_distr: list = [None, None]
    ):
        """
        Initialize the SteerableCNN_QM9 model.

        Args:
            gspace (gspaces.GSpace): The gspace in which data lives.
            num_layers (int): Number of residual layers.
            num_features (int): Number of features for the layers.
            num_inv_features (int): Number of invariant features.
            mlp_params (dict): Parameters for the MLP layers.
            L (int, optional): Maximum frequency of the representations. Defaults to 1.
            edge_distr (list, optional): List of distributions for the edge features. Defaults to [None, None].
        """
        super().__init__()

        self.gspace = gspace
        self.num_layers = num_layers
        self.num_features = num_features
        self.num_inv_features = num_inv_features
        self.L = L
        self.mlp_params = mlp_params
        self.edge_distr = edge_distr

        node_repr = 5 * [gspace.trivial_repr]
        edge_repr = 4 * [gspaces.no_base_space(group.o3_group()).trivial_repr]

        self.embedding = EmbeddingBlock(
            gspace=self.gspace,
            in_repr=node_repr,
            out_channels=self.num_features,
            L=self.L
        )
        self.residual_blocks = torch.nn.ModuleList(
            [ResidualBlock(
                gspace=self.gspace,
                edge_repr=edge_repr,
                out_channels=self.num_features,
                L=self.L,
                mlp_params=self.mlp_params,
                edge_distr=self.edge_distr
            ) for _ in range(self.num_layers)]
        )
        self.invariant_map = InvariantMap(
            gspace=self.gspace,
            in_repr=self.embedding.out_type.representations,
            edge_repr=edge_repr,
            L=self.L,
            mlp_params=self.mlp_params,
            num_inv_features=self.num_inv_features,
            edge_distr=self.edge_distr
        )
        self.pool = global_mean_pool
        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_inv_features),
            torch.nn.ELU(),
            torch.nn.Linear(num_inv_features, num_inv_features),
            torch.nn.BatchNorm1d(num_inv_features),
            torch.nn.ELU(),
            torch.nn.Linear(num_inv_features, 1)
        )

    def forward(self, x, pos, edge_index, edge_attr, batch):
        """
        Forward pass of the SteerableCNN_QM9 model.

        Args:
            x (torch.Tensor): Input node features.
            pos (torch.Tensor): Node positions.
            edge_index (torch.Tensor): Graph connectivity.
            edge_attr (torch.Tensor): Edge attributes.
            batch (torch.Tensor): Batch index for each node.

        Returns:
            torch.Tensor: Output of the model.
        """
        edge_delta = pos[edge_index[1]] - pos[edge_index[0]]
        x = self.embedding(x=x, coords=pos)
        for residual_block in self.residual_blocks:
            x = residual_block(x=x, edge_index=edge_index, edge_delta=edge_delta, edge_attr=edge_attr)
        x = self.invariant_map(x=x, edge_index=edge_index, edge_delta=edge_delta, edge_attr=edge_attr)
        x = self.pool(x=x.tensor, batch=batch)
        x = self.fc(x)
        return x
