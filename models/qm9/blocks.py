import torch
from escnn import nn, gspaces

from models.core.point_convolution import ImplicitPointConv
from utils.nonlinearities import get_elu


class EmbeddingBlock(nn.EquivariantModule):
    """
    Equivariant linear embedding from input trivial type to N sphere representations with frequency up to L.
    """

    def __init__(
        self,
        in_type: nn.FieldType,
        out_channels: int,
        L: int,
    ):
        """
        Initialize the EmbeddingBlock.

        Args:
            in_type (nn.FieldType): Input field type.
            out_channels (int): Number of output channels.
            L (int): Maximum frequency of the output representations.
        """
        if not all(rep.is_trivial() for rep in in_type.representations):
            raise ValueError("Only trivial representations are supported as input.")

        super().__init__()
        gspace = in_type.gspace
        self.in_type = in_type
        self.activation = get_elu(gspace=in_type.gspace, L=L, channels=out_channels)
        self.fc = torch.nn.Linear(len(self.in_type), out_channels)
        self.hid_type = gspace.type(*(out_channels * [gspace.trivial_repr]))

        self.tp_module = nn.TensorProductModule(
            in_type=self.hid_type, out_type=self.activation.in_type, initialize=False
        )

        nn.init.deltaorthonormal_init(
            self.tp_module.weights.data, self.tp_module.basisexpansion
        )

        self.out_type = self.activation.out_type

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> nn.GeometricTensor:
        """
        Forward pass of the EmbeddingBlock.

        Args:
            x (torch.Tensor): Input field.
            coords (torch.Tensor): Coordinates of the input field.

        Returns:
            nn.GeometricTensor: Output field.
        """
        x = self.fc(x)
        x = nn.GeometricTensor(x, self.hid_type, coords)
        x = self.tp_module(x)
        x = self.activation(x)
        return x

    def evaluate_output_shape(self):
        # Implement the method if necessary
        pass


class ResidualBlock(nn.EquivariantModule):
    """
    Equivariant residual block, part of a neural network architecture.

    This block applies two convolutional layers with intermediate activations and batch normalization,
    followed by a residual connection and another activation.
    """

    def __init__(
        self,
        type: nn.FieldType,
        edge_attr_repr: list,
        out_channels: int,
        subgroup_id: tuple,
        n_layers: int,
        n_channels: int,
        L: int,
        init_scheme: str = "he",
    ):
        """
        Initialize the ResidualBlock.

        Args:
            type (nn.FieldType): Input field type.
            edge_attr_repr (list): Edge representation.
            out_channels (int): Number of output channels.
            subgroup_id (tuple): id of the subgroup of the full orthogonal group O(n) to restrict the kernel to.
            n_layers (int): Number of layers in the implicit kernels.
            n_channels (int): Number of hidden channels in the implicit kernels.
            L (int): Maximum frequency of the output representations.
            init_scheme (str): Initialization scheme of the implicit kernels.
                - 'deltaorthonormal': delta-orthonormal initialization scheme.
                - 'he': generalized He initialization scheme.
        """
        super().__init__()
        gspace = type.gspace
        self.activation = get_elu(gspace=gspace, L=L, channels=out_channels)
        assert (
            type == self.activation.in_type
        ), f"Input and hidden types must be the same, got input {type} and hidden {self.activation.in_type}."

        self.in_type = type

        mlp_params = {
            "n_layers": n_layers,
            "n_channels": n_channels,
            "init_scheme": init_scheme,
        }

        self.conv1 = ImplicitPointConv(
            dimensionality=gspace.dimensionality,
            in_type=self.in_type,
            out_type=self.activation.out_type,
            edge_attr_repr=edge_attr_repr,
            subgroup_id=subgroup_id,
            **mlp_params,
        )

        self.conv2 = ImplicitPointConv(
            dimensionality=gspace.dimensionality,
            in_type=self.in_type,
            out_type=self.activation.out_type,
            edge_attr_repr=edge_attr_repr,
            subgroup_id=subgroup_id,
            **mlp_params,
        )

        self.bnorm1 = nn.IIDBatchNorm1d(self.activation.in_type)
        self.bnorm2 = nn.IIDBatchNorm1d(self.activation.in_type)
        self.out_type = self.activation.out_type

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_delta: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the ResidualBlock.

        Args:
            x (torch.Tensor): Input field.
            edge_index (torch.Tensor): Graph connectivity.
            edge_delta (torch.Tensor): Edge deltas.
            edge_attr (torch.Tensor): Edge attributes.

        Returns:
            torch.Tensor: Output field after the residual block.
        """
        out = self.activation(
            self.bnorm1(
                self.conv1(
                    x=x,
                    edge_index=edge_index,
                    edge_delta=edge_delta,
                    edge_attr=edge_attr,
                )
            )
        )
        out = self.bnorm2(
            self.conv2(
                x=out, edge_index=edge_index, edge_delta=edge_delta, edge_attr=edge_attr
            )
        )
        out = out + x
        out = self.activation(out)
        return out

    def evaluate_output_shape(self):
        # Implement the method if necessary
        pass


class InvariantMap(nn.EquivariantModule):
    """
    Equivariant invariant map from non-trivial representations to trivial representations.

    This module is designed to transform fields with non-trivial representations into fields with
    trivial representations, using a convolutional layer.
    """

    def __init__(
        self,
        type: nn.FieldType,
        edge_attr_repr: list,
        out_channels: int,
        subgroup_id: tuple,
        n_layers: int,
        n_channels: int,
        init_scheme: str = "he",
    ):
        """
        Initialize the InvariantMap.

        Args:
            type (nn.FieldType): Input field type.
            edge_attr_repr (list): Edge representation.
            out_channels (int): Number of output channels.
            subgroup_id (tuple): id of the subgroup of the full orthogonal group O(n) to restrict the kernel to.
            n_layers (int): Number of layers in the implicit kernels.
            n_channels (int): Number of hidden channels in the implicit kernels.
            L (int): Maximum frequency of the output representations.
            init_scheme (str): Initialization scheme of the implicit kernels.
                - 'deltaorthonormal': delta-orthonormal initialization scheme.
                - 'he': generalized He initialization scheme.
        """
        super().__init__()
        gspace = type.gspace
        self.in_type = type
        self.out_type = nn.FieldType(gspace, out_channels * [gspace.trivial_repr])
        self.conv = ImplicitPointConv(
            dimensionality=gspace.dimensionality,
            in_type=self.in_type,
            out_type=self.out_type,
            edge_attr_repr=edge_attr_repr,
            subgroup_id=subgroup_id,
            n_layers=n_layers,
            n_channels=n_channels,
            init_scheme=init_scheme,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_delta: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the InvariantMap.

        Args:
            x (torch.Tensor): Input field.
            edge_index (torch.Tensor): Graph connectivity.
            edge_delta (torch.Tensor): Edge deltas.
            edge_attr (torch.Tensor): Edge attributes.

        Returns:
            torch.Tensor: Output field after the invariant map transformation.
        """
        return self.conv(
            x=x, edge_index=edge_index, edge_delta=edge_delta, edge_attr=edge_attr
        )

    def evaluate_output_shape(self):
        # Implement the method if necessary
        pass
