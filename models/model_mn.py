import torch
import torch.distributions as dist
from escnn import group, gspaces, nn

from models.model import SteerableCNN
from utils import BatchGeometricTensor
from .core.blocks import SteerableBlock
from .core.helpers import get_connector, get_convolution
  
class SteerableCNN_MN(SteerableCNN):
    def __init__(self, gspace: gspaces, filter_type: str, non_linearity: str, 
                 n_g_blocks: int, num_nodes: list, num_classes: int, 
                 connectivity: str, channels: list, max_freq: list, edge_dim: int, edge_att_type: str,
                 use_bn: int, x_dist: dist = None, attr_dist: dist = None, pool_type: str = 'mean', 
                 use_skipcons: bool = False, aggr: str = 'add', **kwargs):
                
        # Get connector and point convolution objects based on given connectivity and filter types
        connector, con_params = get_connector(connectivity, n_g_blocks, kwargs)
        point_conv, conv_params = get_convolution(filter_type, n_g_blocks, edge_dim, x_dist, attr_dist, aggr, kwargs)

        subgroup_id = gspace._sg_id[:-1]
        std_rep = group.o3_group().standard_representation()
        
        # the input is a vector of positions
        self.in_type = gspace.type(std_rep.restrict(subgroup_id))

        embed_block = SteerableBlock(
            point_conv=point_conv,
            non_linearity=non_linearity,
            channels=channels[0],
            L=max_freq[0],
            in_type=self.in_type,
            connector=connector,
            con_params=con_params[0],
            num_nodes_in=num_nodes[0],
            num_nodes_out=num_nodes[1],
            edge_att_type=edge_att_type,
            conv_params=conv_params[0],
            gated_type=None,
            use_bn=use_bn,
        )

        super().__init__(embed_block, filter_type, non_linearity, n_g_blocks, num_nodes, num_classes, 
                 connectivity, channels, max_freq, edge_dim, edge_att_type, use_bn,
                 x_dist, attr_dist, pool_type, use_skipcons, aggr, **kwargs)
        
    def forward(self, pos_in: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            pos_in (torch.Tensor): Input positions of shape `(num_nodes, 3)`.
            batch (torch.Tensor): Batch vector of shape `(num_nodes, )`.

        Returns:
            torch.Tensor: Output tensor of shape `(num_samples, num_classes)`.
        """
        x = nn.GeometricTensor(pos_in, self.in_type, pos_in)
        x = BatchGeometricTensor(x, batch)
        x = self.embed_block(x)
        x = self.g_blocks(x)
        x, batch = self.inv_block(x)
        x = self.pooling(x, batch)
        x = self.fully_net(x)
        return x