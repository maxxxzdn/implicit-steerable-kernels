import torch

from escnn import group, gspaces, nn

from ..utils import BatchGeometricTensor
from common.helpers import get_connector, get_convolution
from common.blocks import SteerableBlock, ResidualBlock, EmbeddingBlock, SteerableBlockOutLin


class SteerableCNN_MB(nn.EquivariantModule):
    """
    Steerable CNN for N-body problems
    """
    def __init__(
        self,
        gspace: gspaces,
        filter_type: str,
        non_linearity: str,
        n_g_blocks: int,
        num_nodes: eval,
        connectivity: str,
        channels: eval,
        max_freq: eval,
        use_bn: int,
        use_skipcons: bool = False,
        aggr: str = "mean",
        **kwargs):
        """
        Args:
            gspace (gspaces): the input gspace
            filter_type (str): the type of filter to use
            non_linearity (str): the type of non-linearity to use
            n_g_blocks (int): number of G-blocks
            num_nodes (eval): number of nodes in each G-block
            connectivity (str): the type of connectivity to use
            channels (eval): number of channels in each G-block
            max_freq (eval): maximum frequency in each G-block
            use_bn (int): whether to use batch normalization
            use_skipcons (bool): whether to use skip connections
            aggr (str): the type of aggregation to use
            **kwargs: additional arguments
        """
        super().__init__()
        self.connectivity = connectivity
        # Get connector and point convolution objects based on given connectivity and filter types
        self.connector, self.con_params = get_connector(connectivity, n_g_blocks, kwargs)
        self.point_conv, self.conv_params = get_convolution(filter_type, n_g_blocks, aggr, kwargs)
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
        
        # subgroup_id asserts that the correct G is used in the model
        subgroup_id = gspace._sg_id[:-1]
        # input: positions + velocities
        self.in_type = gspace.type(*(2*[group.o3_group().standard_representation().restrict(subgroup_id)]))
        # condition: velocity 1, velocity 2, charge 1 * charge 2, distance
        self.edge_repr = 2*[gspaces.no_base_space(group.o3_group()).trivial_repr] \
                         + 2*([group.o3_group().standard_representation()] \
                              + [gspaces.no_base_space(group.o3_group()).trivial_repr])
        # output: position update   
        self.out_type = gspace.type(group.o3_group().standard_representation().restrict(subgroup_id))
        
        # Encoder
        self.embed_block = EmbeddingBlock(
            channels=channels[0],
            L=max_freq[0],
            in_type=self.in_type,
            gspace_out=gspace,
        )
        # G-equivariant message passing in hidden space
        self.g_blocks = nn.SequentialModule()
        hid_type = self.embed_block.out_type
        gated_type = self.embed_block.activation.in_type
        for i in range(1, n_g_blocks):
            print(i)
            self.g_blocks.append(
                GBlock(
                    point_conv=self.point_conv,
                    non_linearity=non_linearity,
                    channels=channels[i],
                    L=max_freq[i],
                    in_type=hid_type,
                    edge_repr=self.edge_repr,
                    connector=self.connector,
                    con_params=self.con_params[i],
                    num_nodes_in=num_nodes[i],
                    num_nodes_out=num_nodes[i+1],
                    conv_params=self.conv_params[i],
                    gated_type=gated_type,
                    use_bn=use_bn,
                )
            )
            hid_type = self.g_blocks.out_type
            gated_type = self.g_blocks[-1].activation.in_type

        # Decoder
        self.decoder_block = SteerableBlockOutLin(
            L=max_freq[-1],
            channels=channels[-1],
            in_type=self.g_blocks.out_type,
            out_type=self.out_type,)
        
    def evaluate_output_shape(self, input_shape: tuple):
        pass
    
    def forward(
        self, 
        pos_in: torch.Tensor, 
        batch: torch.Tensor, 
        node_features: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor):
        """
        Args:
            pos_in (torch.Tensor): input positions
            batch (torch.Tensor): batch indices
            node_features (torch.Tensor): node features
            edge_index (torch.Tensor): edge indices
            edge_attr (torch.Tensor): edge attributes
        Returns:    
            torch.Tensor: output positions
        """
        x = torch.cat([pos_in, node_features], -1) # pos, vel
        x = nn.GeometricTensor(x, self.in_type, pos_in)
        x = BatchGeometricTensor(
            x,
            batch,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        x.gt = self.embed_block(x.gt)
        x = self.g_blocks(x)
        x = self.decoder_block(x)
        return x.tensor