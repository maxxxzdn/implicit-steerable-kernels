import torch.distributions as dist
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from escnn import nn

from common.blocks import SteerableBlock, InvariantMap, ResidualBlock
from common.helpers import get_connector, get_convolution, get_fc

pool_dict = {'max': global_max_pool, 'mean': global_mean_pool, 'add': global_add_pool}

  
class SteerableCNN(nn.EquivariantModule):
    """
    SteerableCNN module to perform classification on 3D shapes using equivariant neural networks.        
    """
    def __init__(
        self,
        embed_block,
        filter_type: str,
        non_linearity: str,
        n_g_blocks: int,
        num_nodes: eval,
        num_classes: int,
        connectivity: str,
        channels: eval,
        max_freq: eval,
        edge_dim: int,
        edge_att_type: str,
        use_bn: int,
        x_dist: dist = None,
        attr_dist: dist = None,
        pool_type: str = "mean",
        use_skipcons: bool = False,
        aggr: str = "add",
        **kwargs):
        """
        Args:
            embed_block (nn.Module): Embedding block to create an initial embedding for the shape
            filter_type (str): Type of filter used in the convolution
            non_linearity (str): Type of non-linearity function used in the model
            n_g_blocks (int): Number of G-blocks in the model
            num_nodes (list): Number of nodes in each G-block
            num_classes (int): Number of output classes
            connectivity (str): Type of connectivity used in the model
            channels (list): Number of output channels in each G-block
            max_freq (list): Maximum frequency for each G-block
            edge_dim (int): Dimensionality of edge features
            edge_att_type (str): Type of edge attention mechanism used in the model
            use_bn (int): Whether to use batch normalization or not
            x_dist (dist): Distribution of node features
            attr_dist (dist): Distribution of edge features
            pool_type (str): Type of pooling to be used in the model
            use_skipcons (bool): Whether to use skip connections or not
            aggr (str): Type of aggregation used in the model
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.connectivity = connectivity
        # Get connector and point convolution objects based on given connectivity and filter types
        connector, con_params = get_connector(connectivity, n_g_blocks, kwargs)
        point_conv, conv_params = get_convolution(filter_type, n_g_blocks, edge_dim, x_dist, attr_dist, aggr, kwargs)
        # Check the inputs and assert their validity
        if len(num_nodes) == 1:
            num_nodes = num_nodes*(n_g_blocks+2)
        else:
            assert len(num_nodes) == n_g_blocks+2, f"number of nodes ({len(num_nodes)}) != number of layers ({n_g_blocks+2})"               
        if len(max_freq) == 1:
            max_freq = max_freq*n_g_blocks
        else:
            assert len(max_freq) == n_g_blocks, f"number of max.freq ({len(max_freq)}) != number of layers ({n_g_blocks})"  
        if len(channels) == 1:
            channels = channels*n_g_blocks
        else:
            assert len(channels) == n_g_blocks, f"number of channels ({len(channels)}) != number of layers ({n_g_blocks})"              
        if use_skipcons:
            assert len(set(num_nodes[:-1])) == 1 # same number of channels in G-equivariant convolutions
            assert len(set(max_freq[1:])) == 1 # same frequency in G-equivariant convolutions
            GBlock = ResidualBlock
        else:
            GBlock = SteerableBlock

        # Encoder
        self.embed_block = embed_block

        # G-equivariant message passing in hidden space
        self.g_blocks = nn.SequentialModule()
        hid_type = self.embed_block.out_type
        gated_type = self.embed_block.activation.in_type
        for i in range(1, n_g_blocks):
            self.g_blocks.append(
                GBlock(
                    point_conv=point_conv,
                    non_linearity=non_linearity,
                    channels=channels[i],
                    L=max_freq[i],
                    in_type=hid_type,
                    connector=connector,
                    con_params=con_params[i],
                    num_nodes_in=num_nodes[i],
                    num_nodes_out=num_nodes[i+1],
                    edge_att_type=edge_att_type,
                    conv_params=conv_params[i],
                    gated_type=gated_type,
                    use_bn=use_bn,
                )
            )
            hid_type = self.g_blocks.out_type
            gated_type = self.g_blocks[-1].activation.in_type
        
        # Decoder
        self.inv_block = InvariantMap(
            point_conv=point_conv,
            non_linearity=non_linearity,
            num_inv_channels=kwargs['num_inv_channels'],
            in_type=self.g_blocks.out_type,
            connector=connector,
            con_params=con_params[n_g_blocks],
            num_nodes_in=num_nodes[n_g_blocks],
            num_nodes_out=num_nodes[n_g_blocks],
            edge_att_type=edge_att_type,
            conv_params=conv_params[n_g_blocks],
            use_bn=use_bn,
            ) 
        self.fully_net = get_fc(non_linearity, num_classes, use_bn, kwargs)  

        self.pooling = pool_dict[pool_type] 
            
    def forward(x):
        pass

    def evaluate_output_shape(self, input_shape: tuple):
        pass
