import torch
from torch import nn
from torch.nn.functional import conv2d, conv3d, pad

from escnn import nn, group

from utils.rd_convolution import _RdConv
from models.core.implicit_kernel import ImplicitKernel
from .kernel_utils import generate_kernel_grid


class ImplicitConv(_RdConv):
    def __init__(
        self,
        in_type: nn.FieldType,
        out_type: nn.FieldType,
        kernel_attr_type: nn.FieldType,
        kernel_size: int,
        n_layers: int,
        n_channels: int,
        init_scheme: str = "he",
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        padding_mode: str = "zeros",
    ):
        """
        Grid convolution with an implicit kernel. See https://arxiv.org/abs/2212.06096, Section 3.4 for more details.

        Args:
            in_type (nn.FieldType): input field type.
            out_type (nn.FieldType): output field type.
            kernel_attr_type (nn.FieldType): field type of the kernel attributes.
            kernel_size (int): size of the kernel.
            padding (int): padding of the input.
            stride (int): stride of the convolution.
            n_layers (int): number of layers in the MLP.
            n_channels (int): number of hidden channels in the MLP.
            init_scheme (str): initialization scheme of the implicit kernel.
                - 'deltaorthonormal': delta-orthonormal initialization scheme.
                - 'he': generalized He initialization scheme.
            subgroup_id (tuple): id of the subgroup of the full orthogonal group O(n) to restrict the kernel to.
        """
        dimensionality = in_type.gspace.dimensionality

        super().__init__(
            in_type=in_type,
            out_type=out_type,
            d=dimensionality,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            padding_mode=padding_mode,
            groups=1,
            bias=True,
            basis_filter=False,
            recompute=False,
            use_implicit=True,
        )

        self.kernel_size = kernel_size
        self.grid = generate_kernel_grid(kernel_size, dimensionality)
        self.uses_kernel_attr = True if kernel_attr_type is not None else False

        self.implicit_kernel = ImplicitKernel(
            conv_in_type=in_type,
            conv_out_type=out_type,
            kernel_attr_type=kernel_attr_type,
            n_layers=n_layers,
            n_channels=n_channels,
            init_scheme=init_scheme,
        )

        self.conv_operator = conv2d if dimensionality == 2 else conv3d

    def expand_filter(self):
        """
        Expands the implicit kernel to the grid if no attributes are used.
        """
        if not self.uses_kernel_attr:
            return self.implicit_kernel(self.grid)
        else:
            return None

    def expand_filter_attr(self, attr):
        """
        Expands the implicit kernel to the grid if attributes are used.

        Args:
            attr (torch.Tensor): global attribute vector of shape (1, attr_dim).
        """
        attr = attr.repeat(self.grid.shape[0], 1)
        return self.implicit_kernel(self.grid, attr)

    def forward(
        self,
        x: nn.GeometricTensor,
        attr: torch.Tensor = None,
    ):
        """
        Performs a forward pass of the ImplicitConv layer.

        Args:
            x (nn.GeometricTensor): The feature tensor of shape (B, C, X_1, ..., X_D).
            attr (torch.Tensor): global attribute vector of shape (B, attr_dim).

        Returns:
            nn.GeometricTensor: convolution output.
        """
        assert isinstance(x, nn.GeometricTensor)
        assert x.type == self.in_type
        assert attr is None or len(attr.shape) == 2

        if not self.training:
            if not self.uses_kernel_attr:
                _filter = self.filter
            else:
                _filter = self.expand_filter_attr(attr)
            _bias = self.expanded_bias
        else:
            # retrieve the filter and the bias
            _filter = (
                self.expand_filter_attr(attr)
                if self.uses_kernel_attr
                else self.expand_filter()
            )
            _bias = self.expand_bias()

        # reshape filter from (n, c_out, c_in) to (c_out, c_in, X_1, ..., X_D)
        _filter = _filter.permute(1, 2, 0).reshape(
            _filter.shape[1], _filter.shape[2], *(self.d * [self.kernel_size])
        )

        if self.padding_mode == "zeros":
            out = self.conv_operator(
                x.tensor,
                _filter,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=_bias,
            )
        else:
            out = self.conv_operator(
                pad(
                    x.tensor,
                    self._reversed_padding_repeated_twice,
                    self.padding_mode,
                ),
                _filter,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
                bias=_bias,
            )

        out = nn.GeometricTensor(out, self.out_type)

        return out

    def _build_kernel_basis(self):
        """
        Builds the kernel basis. Dummy method for compatibility with _RdPointConv.
        """
        pass
