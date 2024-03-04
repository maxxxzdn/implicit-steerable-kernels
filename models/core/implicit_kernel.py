import torch
import math
from escnn import nn, group, gspaces

from .mlp import MLP
from .kernel_utils import compute_scalar_shell
from utils.escnn_utils import get_tensor_representation


class ImplicitKernel(torch.nn.Module):
    def __init__(
        self,
        conv_in_type: nn.FieldType,
        conv_out_type: nn.FieldType,
        kernel_attr_type: nn.FieldType = None,
        n_layers: int = 2,
        n_channels: int = 16,
        init_scheme: str = "he",
    ):
        """
        Steerable kernel. See https://arxiv.org/abs/2212.06096, Section 3 for more details.
            - input: ND positions + other attributes.
            - output: steerable kernel value at each coordinate.

        Args:
            conv_in_type (nn.FieldType): field type of the convolution input.
            conv_out_type (nn.FieldType): field type of the convolution output.
            kernel_attr_type (nn.FieldType): field type of the kernel attributes.
                - e.g. for point clouds, those attributes could be the distances.
                - if None, the kernel is only a function of the positions.
            n_layers (int): number of layers in the MLP.
            n_channels (int): number of hidden channels in the MLP.
            init_scheme (str): initialization scheme name.
                - 'deltaorthonormal': delta-orthonormal initialization scheme.
                - 'he': generalized He initialization scheme.
        """

        super().__init__()
        assert (
            conv_in_type.uniform and conv_out_type.uniform
        ), f"Input and output fields must have uniform representations, but got {conv_in_type.representations} and {conv_out_type.representations}."
        if kernel_attr_type is not None:
            assert (
                kernel_attr_type.representations[0].group
                == conv_in_type.representations[0].group
                == conv_out_type.representations[0].group
            ), f"Kernel attributes must have the same group as the input and output fields, but got {kernel_attr_type.representations[0].group}, {conv_in_type.representations[0].group}, and {conv_out_type.representations[0].group}."

        # we start with the full orthogonal group, which will be restricted to the input field group
        G = group.O3() if conv_in_type.gspace.dimensionality == 3 else group.O2()

        # no base space for an MLP
        kernel_gspace = gspaces.no_base_space(G)

        # positions have standard representation
        pos_repr = G.standard_representation()

        # now we restrict the gspace to the input field group
        if conv_in_type.gspace.fibergroup != G:
            subgroup_id = conv_in_type.gspace._sg_id
            kernel_gspace = kernel_gspace.restrict(subgroup_id)[0]
            pos_repr = pos_repr.restrict(subgroup_id)

        # kernel input representation: positions + attributes (if any)
        kernel_in_repr = [pos_repr]
        if kernel_attr_type is not None:
            kernel_in_repr += kernel_attr_type.representations

        # define the kernel input (positions + attributes) and output (tensor representation) types
        kernel_in_type = kernel_gspace.type(*kernel_in_repr)
        kernel_out_type = kernel_gspace.type(
            *get_tensor_representation(
                conv_in_type.representations, conv_out_type.representations
            )
        )

        # define the MLP
        self.mlp = MLP(
            kernel_in_type, kernel_out_type, n_layers, n_channels, init_scheme
        )

        # number of input and output channels + their sizes for reshaping later
        self.c_in, self.c_out = len(conv_in_type.representations), len(
            conv_out_type.representations
        )
        self.in_size = conv_in_type.representations[0].size
        self.out_size = conv_in_type.representations[0].size

        # normalization factor
        self.factor = 1 / math.sqrt(self.c_out)

        # radial shell to cut off the kernel at the corners of the grid (improves equivariance)
        self.shell_width = torch.nn.Parameter(0.5 * torch.ones(1, 1, 1))

    def forward(self, pos: torch.Tensor, attrs: torch.Tensor = None):
        """
        Compute kernel values at each position + attribute.

        1. compute the kernel values at each position.
        2. reshape the kernel values to matrix form.
        3. apply the radial shell to cut off the kernel at the corners of the grid.
        """
        shell = compute_scalar_shell(pos[:, None, :], self.shell_width)

        if attrs is None:
            attrs = torch.zeros(pos.shape[0], 0, device=pos.device)
        x = nn.GeometricTensor(torch.cat([pos, attrs], 1), self.mlp.in_type)

        k = self.mlp(x).tensor

        k = k.reshape((-1, self.c_out, self.c_in, self.out_size, self.in_size))
        k = k.permute((0, 1, 3, 2, 4))
        k = k.reshape((-1, self.c_out * self.out_size, self.c_in * self.in_size))

        return k * shell * self.factor
