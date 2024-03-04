import torch
from torch import nn

from escnn import nn, group

from utils.rd_point_convolution import _RdPointConv
from models.core.implicit_kernel import ImplicitKernel


class ImplicitPointConv(_RdPointConv):
    def __init__(
        self,
        in_type: nn.FieldType,
        out_type: nn.FieldType,
        edge_attr_type: nn.FieldType,
        n_layers: int,
        n_channels: int,
        init_scheme: str = "he",
    ):
        """
        Point convolution with an implicit kernel. See https://arxiv.org/abs/2212.06096, Section 3.4 for more details.
        It inherits from _RdPointConv, which we upgraded to support implicit kernels.

        Args:
            in_type (nn.FieldType): input field type.
            out_type (nn.FieldType): output field type.
            edge_attr_type (nn.FieldType): field type of the edge attributes.
            n_layers (int): number of layers in the MLP.
            n_channels (int): number of hidden channels in the MLP.
            init_scheme (str): initialization scheme of the implicit kernel.
                - 'deltaorthonormal': delta-orthonormal initialization scheme.
                - 'he': generalized He initialization scheme.
        """
        dimensionality = in_type.gspace.dimensionality

        super().__init__(
            in_type=in_type,
            out_type=out_type,
            d=dimensionality,
            groups=1,
            bias=True,
            basis_filter=False,
            recompute=False,
            use_implicit=True,
        )

        self.implicit_kernel = ImplicitKernel(
            conv_in_type=in_type,
            conv_out_type=out_type,
            kernel_attr_type=edge_attr_type,
            n_layers=n_layers,
            n_channels=n_channels,
            init_scheme=init_scheme,
        )

    def message(
        self, x_j: torch.Tensor, edge_delta: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the message passed between nodes.

        Args:
            x_j (torch.Tensor): features of the neighboring nodes of shape (batch_size * num_edges, in_channels).
            edge_delta (torch.Tensor): relative positions of the neighboring nodes of shape (batch_size * num_edges, d).
            edge_attr (torch.Tensor): edge attributes of shape (batch_size * num_edges, edge_attr_dim).

        Returns:
            torch.Tensor: The computed message tensor of shape (batch_size, num_nodes, out_channels).
        """
        _filter = self.implicit_kernel(edge_delta, edge_attr)
        return torch.einsum("noi,ni->no", _filter, x_j)

    def forward(
        self,
        x: nn.GeometricTensor,
        edge_index: torch.Tensor,
        edge_delta: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ):
        """
        Performs a forward pass of the ImplicitPointConv layer.

        Args:
            x (nn.GeometricTensor): The feature tensor of shape (batch_size * num_nodes, in_channels).
            edge_index (torch.Tensor): The edge index tensor of shape (2, num_edges).
            edge_delta (torch.Tensor): The relative position tensor of shape (batch_size * num_edges, d).
                - warning: it needs to be given explicitly due to the bug in escnn. In the future, it will be computed from the coordinates of x.
            edge_attr (torch.Tensor): The edge attribute tensor of shape (batch_size * num_edges, edge_attr_dim).

        Returns:
            nn.GeometricTensor: convolution output.
        """
        assert isinstance(x, nn.GeometricTensor)
        assert x.type == self.in_type

        assert len(edge_index.shape) == 2
        assert edge_index.shape[0] == 2

        coords = x.coords

        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.shape[1], 0, device=edge_index.device)

        out = self.propagate(
            edge_index, x=x.tensor, edge_delta=edge_delta, edge_attr=edge_attr
        )

        _bias = self.expanded_bias if not self.training else self.expand_bias()

        if _bias is not None:
            out += _bias

        out = nn.GeometricTensor(out, self.out_type, coords=coords)

        return out

    def _build_kernel_basis(self):
        """
        Builds the kernel basis. Dummy method for compatibility with _RdPointConv.
        """
        pass


class ImplicitPointConvWithDownsampling(ImplicitPointConv):
    """
    ImplicitPointConv with downsampling.
    It takes indices of the downsampled points, and performs the convolution only on these points and their neighbors.
    """

    def forward(
        self,
        x: nn.GeometricTensor,
        edge_index: torch.Tensor,
        edge_delta: torch.Tensor,
        edge_attr: torch.Tensor,
        idx_downsampled: torch.Tensor,
    ):
        """
        Performs the forward pass of the convolution operation.

        Args:
            x (nn.GeometricTensor): The feature tensor of shape (batch_size * num_nodes, in_channels).
            edge_index (torch.Tensor): The edge index tensor of shape (2, num_edges).
            edge_delta (torch.Tensor): The edge delta tensor of shape (batch_size * num_edges, d).
            edge_attr (torch.Tensor): The edge attribute tensor of shape (batch_size * num_edges, edge_attr_dim).
            idx_downsampled (torch.Tensor): The indices of the downsampled points.
                - if x has N points, idx_downsampled has M points, and M <= N.

        Returns:
            nn.GeometricTensor: downsampled convolution output.
        """

        if idx_downsampled is not None:
            coords = x.coords[idx_downsampled]
            x = (x.tensor, x.tensor[idx_downsampled])
        else:
            return super().forward(x, edge_index, edge_delta, edge_attr=edge_attr)

        out = super().propagate(
            edge_index, x=x, edge_delta=edge_delta, edge_attr=edge_attr
        )

        _bias = self.expanded_bias if not self.training else self.expand_bias()

        if _bias is not None:
            out += _bias

        out = nn.GeometricTensor(out, self.out_type, coords=coords)

        return out
