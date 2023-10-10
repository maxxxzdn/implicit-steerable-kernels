import torch
from typing import Union, Tuple
from torch_geometric.nn import knn, radius
from torch_geometric.utils import add_self_loops
from escnn import nn, gspaces

from utils import sample_indices, BatchGeometricTensor, get_elu, get_gated
from .point_convolution import R3PointConv


class BaseBlock(nn.EquivariantModule):
    """
    A base block for equivariant point convolutions.
    Base functionality:
        - batch-wise downsampling of point clouds;
        - connecting an original point cloud to its downsampled version;
        - computing edge features.
    """
    def __init__(
            self,
            in_type: nn.FieldType,
            connector: Union[knn, radius],
            con_params: list,
            num_nodes_in: int,
            num_nodes_out: int,
            edge_att_type: str
        ):
        """
        Args:
            in_type: input type;
            connector: a connector module [knn, radius];
            con_params: parameters of the connector module, e.g. k for knn;
            num_nodes_in: number of nodes in the input point cloud;
            num_nodes_out: number of nodes in the output point cloud (downsampled);
            edge_att_type: type of edge features.
        """
        super().__init__()
        self.in_type = in_type
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out
        self.connector = connector
        self.con_params = con_params
        self.edge_att_type = edge_att_type

    def downsample(
            self, 
            x: BatchGeometricTensor
        ) -> torch.Tensor:
        """
        For each point cloud in a batch returns indices of randomly sampled points.
        """
        return sample_indices(x.batch, self.num_nodes_out)

    def connect(
            self, 
            x: BatchGeometricTensor,
            idx_downsampled: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Connect a point cloud (source) to its downsampled version (target).
        """
        edge_index = self.connector(
            x=x.coords[idx_downsampled],
            batch_x=x.batch[idx_downsampled],
            y=x.coords,
            batch_y=x.batch,
            **self.con_params
        )
        return edge_index

    def compute_edge_info(
            self,
            x: BatchGeometricTensor,
            idx_downsampled: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute edge features for edges going from source graph to target downsampled graph.
        """
        row, cols = edge_index
        pos_j = x.coords[row]
        pos_i = x.coords[idx_downsampled][cols]
        # edge_delta = torch.cat((pos_j[row], pos_i[cols]), dim=-1)
        edge_delta = pos_j - pos_i

        # native edge features + features of source and target nodes
        if self.edge_att_type == 'edge_attr_z':
            assert x.node_features is not None and x.edge_attr is not None
            features_j = x.node_features[row]
            features_i = x.node_features[cols]
            edge_attr = torch.cat([edge_attr, features_j, features_i], 1)
            # edge_attr = torch.cat([pos_j, pos_i, edge_attr, features_j, features_i], 1)
        # only features of source and target nodes
        elif self.edge_att_type == 'z':
            assert x.node_features is not None
            features_j = x.node_features[row]
            features_i = x.node_features[cols]
            edge_attr = torch.cat([features_j, features_i], 1)
        # no edge features
        elif self.edge_att_type == 'none':
            edge_attr = None

        return edge_delta, edge_attr

    def evaluate_output_shape(
            self, 
            input_shape: tuple
        ) -> list:
        shape = list(input_shape)
        assert len(shape) == 2, shape
        assert shape[1] == self.in_type.size, shape
        shape[1] = self.out_type.size
        return shape


class EmbeddingBlock(BaseBlock):
    """
    Equivariant linear embedding from input trivial type to N sphere representations with frequency up to L.
    """
    def __init__(
            self,
            non_linearity: str,
            channels: int,
            L: int,
            in_type: nn.FieldType):
        """
        Args:
            non_linearity: activation function type [swish_gate, elu];
            channels: number of channels in the hidden layer;
            L: maximum frequency of the input representations;
            in_type: input type.
        """
        # only trivial representations supported
        assert False not in [rep.is_trivial()
                             for rep in in_type.representations]
        super().__init__(in_type, None, None, None, None, None)

        gspace = in_type.gspace
        self.in_type = in_type

        if non_linearity == 'swish_gate':
            self.activation = get_gated(self.in_type, channels)
        elif non_linearity == 'elu':
            self.activation = get_elu(gspace=gspace, L=L, channels=channels)
        else:
            raise NotImplementedError
        
        # linear transformation of trivial representations
        self.fc = torch.nn.Linear(len(in_type), channels)
        self.hid_type = gspace.type(*(channels * [gspace.trivial_repr]))
        # tensor product module from trivial representations to N sphere representations
        self.tp_module = nn.TensorProductModule(
            in_type=self.hid_type,
            out_type=self.activation.in_type,
            initialize=False)
        
        # weight initialization
        nn.init.deltaorthonormal_init(
            self.tp_module.weights.data,
            self.tp_module.basisexpansion)
        
        self.out_type = self.activation.out_type

    def forward(self, x: torch.Tensor,
                coords: torch.Tensor) -> nn.GeometricTensor:
        x = self.fc(x)
        x = nn.GeometricTensor(x, self.hid_type, coords)
        x = self.tp_module(x)
        x = self.activation(x)
        return x


class SteerableBlock(BaseBlock):
    """
    Equivariant point convolution block: 
        downsample -> connect -> compute edge features ->
        -> convolution -> batch normalization -> activation.
    """
    def __init__(
            self,
            point_conv,
            non_linearity: str,
            channels: int,
            L: int,
            in_type: nn.FieldType,
            connector: Union[knn, radius],
            con_params: list,
            num_nodes_in: int,
            num_nodes_out: int,
            edge_att_type: str,
            use_bn: int,
            conv_params: dict = None,
            gated_type=None):
        """
        Args:
            point_conv: a point convolution module;
            non_linearity: activation function type [swish_gate, elu];
            channels: number of channels in the hidden layer;
            L: maximum frequency of the input representations;
            in_type: input type;
            connector: a connector module [knn, radius];
            con_params: parameters of the connector module, e.g. k for knn;
            num_nodes_in: number of nodes in the input point cloud;
            num_nodes_out: number of nodes in the output point cloud (downsampled);
            edge_att_type: type of edge features;
            use_bn: whether to use batch normalization;
            conv_params: parameters of the point convolution module;
            gated_type: type of the gated activation function (for swish_gate non-linearity)
        """
        super().__init__(
            in_type,
            connector,
            con_params,
            num_nodes_in,
            num_nodes_out,
            edge_att_type)

        self.conv_params = conv_params

        if non_linearity == 'swish_gate':
            self.activation = get_gated(
                in_type=self.in_type, 
                channels=channels, 
                gated_type=gated_type)
        elif non_linearity == 'elu':
            self.activation = get_elu(
                gspace=self.in_type.gspace, 
                L=L, 
                channels=channels)
        else:
            raise NotImplementedError

        self.conv = point_conv(
            in_type=self.in_type,
            out_type=self.activation.in_type,
            **conv_params)
        self.bnorm = nn.IIDBatchNorm1d(self.activation.in_type) if use_bn else None
        self.out_type = self.activation.out_type

    def forward(self, x: BatchGeometricTensor) -> BatchGeometricTensor:
        """
        downsample -> connect -> compute edge features ->
        -> convolution -> batch normalization -> activation.
        """
        # downsample
        idx_downsampled = self.downsample(x)
        # connect
        if x.edge_index is None:
            edge_index = self.connect(x, idx_downsampled)
            edge_attr = None
        else:
            edge_index = x.edge_index
            edge_attr = x.edge_attr
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=0.)
        # compute edge features
        edge_delta, edge_attr = self.compute_edge_info(
            x=x,
            idx_downsampled=idx_downsampled,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        # convolution from original to downsampled point cloud
        if isinstance(self.conv, R3PointConv):
            x.gt = self.conv(
                x=(x.gt, idx_downsampled),
                edge_index=edge_index,
                edge_delta=edge_delta)
        else:
            x.gt = self.conv(
                x=(x.gt, idx_downsampled),
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_delta=edge_delta)
        # batch normalization and activation
        if self.bnorm is not None:
            x.gt = self.bnorm(x.gt)
        x.gt = self.activation(x.gt)
        # update batch indices
        x.batch = x.batch[idx_downsampled]
        return x


class InvariantMap(BaseBlock):
    """
    Invariant point convolution block:
        downsample -> connect -> compute edge features ->
        -> convolution -> batch normalization -> activation.
    """
    def __init__(
            self,
            point_conv,
            non_linearity: str,
            num_inv_channels: int,
            in_type: nn.FieldType,
            connector: Union[knn, radius],
            con_params: list,
            num_nodes_in: int,
            num_nodes_out: int,
            edge_att_type: str,
            use_bn: int,
            conv_params: dict = None):
        """
        Args:
            point_conv: a point convolution module;
            non_linearity: activation function type [swish_gate, elu];
            num_inv_channels: number of channels in the hidden layer;
            in_type: input type;
            connector: a connector module [knn, radius];
            con_params: parameters of the connector module, e.g. k for knn;
            num_nodes_in: number of nodes in the input point cloud;
            num_nodes_out: number of nodes in the output point cloud (downsampled);
            edge_att_type: type of edge features;
            use_bn: whether to use batch normalization;
            conv_params: parameters of the point convolution module.
        """
        super().__init__(
            in_type,
            connector,
            con_params,
            num_nodes_in,
            num_nodes_out,
            edge_att_type)
        
        self.out_type = nn.FieldType(
            in_type.gspace, num_inv_channels * [in_type.gspace.trivial_repr])
        self.conv = point_conv(in_type=self.in_type,
                               out_type=self.out_type,
                               **conv_params)

        self.bnorm = torch.nn.BatchNorm1d(num_inv_channels) if use_bn else None

        if non_linearity == 'swish_gate':
            self.activation = torch.nn.SiLU()
        elif non_linearity == 'elu':
            self.activation = torch.nn.ELU()
        else:
            raise NotImplementedError

    def forward(self, x: BatchGeometricTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        downsample -> connect -> compute edge features ->
        -> convolution -> batch normalization -> activation.
        """
        # original batch indices
        batch = x.batch
        # downsample
        idx_downsampled = self.downsample(x)
        # connect
        if x.edge_index is None:
            edge_index = self.connect(x, idx_downsampled)
            edge_attr = None
        else:
            edge_index = x.edge_index
            edge_attr = x.edge_attr
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=0.)
        # compute edge features
        edge_delta, edge_attr = self.compute_edge_info(
            x=x,
            idx_downsampled=idx_downsampled,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        # convolution from original to downsampled point cloud
        if isinstance(self.conv, nn.R3PointConv):
            x.gt = self.conv(
                x=(x.gt, idx_downsampled),
                edge_index=edge_index,
                edge_delta=edge_delta)
        else:
            x.gt = self.conv(
                x=(x.gt, idx_downsampled),
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_delta=edge_delta)
        # batch normalization and activation
        x = x.tensor
        if self.bnorm is not None:
            x = self.bnorm(x)
        x = self.activation(x)
        # batch is returned since x doesn't store it anymore
        return x, batch[idx_downsampled]
    

class SteerableBlockOutLin(nn.EquivariantModule):
    """
    Equivariant MLP block.
    """
    def __init__(
        self,
        L : int,
        channels: int,
        in_type: nn.FieldType,
        out_type: nn.FieldType):
        """
        Args:
            L: maximum frequency of the input representations;
            channels: number of channels in the hidden layer;
            in_type: input type;
            out_type: output type.
        """
        super().__init__()
        G = in_type.gspace.fibergroup
        gspace = gspaces.no_base_space(G)
        self.hid_type = gspace.type(*in_type.representations)
        self.out_type = gspace.type(*out_type.representations)
        self.activation = get_elu(gspace = gspace, L = L, channels = channels)
        self.linear1 = nn.Linear(self.hid_type, self.activation.in_type)
        self.linear2 = nn.Linear(self.activation.out_type, self.activation.in_type)
        self.linear3 = nn.Linear(self.activation.out_type, self.out_type)
        
        nn.init.deltaorthonormal_init(self.linear1.weights.data, self.linear1.basisexpansion)
        nn.init.deltaorthonormal_init(self.linear2.weights.data, self.linear2.basisexpansion)
        nn.init.deltaorthonormal_init(self.linear3.weights.data, self.linear3.basisexpansion)
        
    def forward(self, x: BatchGeometricTensor) -> BatchGeometricTensor:
        x = x.tensor
        x = self.hid_type(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x

    def evaluate_output_shape(self, input_shape: tuple):
        shape = list(input_shape)
        assert len(shape) == 2, shape
        assert shape[1] == self.in_type.size, shape
        shape[1] = self.out_type.size
        return shape


class ResidualBlock(SteerableBlock):
    """
    Equivariant point convolution block with skip-connections:
        downsample -> connect -> compute edge features ->
        -> convolution -> batch normalization -> activation -> skip-connection.
    """
    def __init__(
            self,
            point_conv,
            non_linearity: str,
            channels: int,
            L: int,
            in_type: nn.FieldType,
            connector: Union[knn, radius],
            con_params: list,
            num_nodes_in: int,
            num_nodes_out: int,
            edge_att_type: str,
            use_bn: int,
            conv_params: dict = None,
            gated_type=None):
        """
        Args:
            point_conv: a point convolution module;
            non_linearity: activation function type [swish_gate, elu];
            channels: number of channels in the hidden layer;
            L: maximum frequency of the input representations;
            in_type: input type;
            connector: a connector module [knn, radius];
            con_params: parameters of the connector module, e.g. k for knn;
            num_nodes_in: number of nodes in the input point cloud;
            num_nodes_out: number of nodes in the output point cloud (downsampled);
            edge_att_type: type of edge features;
            use_bn: whether to use batch normalization;
            conv_params: parameters of the point convolution module;
            gated_type: type of the gated activation function (for swish_gate non-linearity)
        """
        assert gated_type is not None  # must be the same as for the previous block hence given
        super().__init__(point_conv, non_linearity, channels, L, in_type, connector, con_params,
                         num_nodes_in, num_nodes_out, edge_att_type, use_bn, conv_params, gated_type)
        assert self.in_type == self.out_type

    def forward(self, x: BatchGeometricTensor) -> BatchGeometricTensor:
        """
        downsample -> connect -> compute edge features ->
        -> convolution -> batch normalization -> activation -> skip-connection.
        """
        # downsample
        idx_downsampled = self.downsample(x)
        # connect
        if x.edge_index is None:
            edge_index = self.connect(x, idx_downsampled)
            edge_attr = None
        else:
            edge_index = x.edge_index
            edge_attr = x.edge_attr
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=0.)
        # compute edge features
        edge_delta, edge_attr = self.compute_edge_info(
            x=x,
            idx_downsampled=idx_downsampled,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        # convolution from original to downsampled point cloud
        if isinstance(self.conv, R3PointConv):
            out = self.conv(
                x=(x.gt, idx_downsampled),
                edge_index=edge_index,
                edge_delta=edge_delta)
        else:
            out = self.conv(
                x=(x.gt, idx_downsampled),
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_delta=edge_delta)
        # batch normalization and activation
        if self.bnorm is not None:
            out = self.bnorm(out)
        out = self.activation(out)
        # skip-connection
        x.gt += out
        x.batch = x.batch[idx_downsampled]
        return x
