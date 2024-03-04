from escnn import nn, gspaces

from utils.escnn_utils import repr_max_freq
from utils.nonlinearities import get_elu


class MLP(nn.EquivariantModule):
    def __init__(
        self,
        in_type: nn.FieldType,
        out_type: nn.FieldType,
        n_layers: int = 2,
        n_channels: int = 16,
        init_scheme: str = "deltaorthonormal",
        norm: bool = True,
    ):
        """
        MLP that is equivariant to subgroups of O(N).

        Args:
            in_type (nn.FieldType): type of the input field.
            out_type (nn.FieldType): type of the output field.
            n_layers (int): number of layers in the MLP.
            n_channels (int): number of hidden channels in the MLP.
            initialize (bool): whether to initialize the MLP
                - warning: might take a while for large MLPs.
        """
        super().__init__()
        self.G = in_type.fibergroup
        self.gspace = gspaces.no_base_space(self.G)
        self.in_type = in_type
        self.out_type = out_type

        # INITIALIZATION
        self.mlp = nn.SequentialModule()
        hid_type = in_type

        if n_layers > 1:
            L = (
                repr_max_freq(out_type.representations) + 1
            )  # bandlimit angular frequency as the maximum frequency of the input representation
            activation = get_elu(
                gspace=self.gspace, L=L, channels=n_channels
            )  # ELU-like activation function

        for _ in range(n_layers - 1):
            self.mlp.append(
                nn.Linear(hid_type, activation.in_type, bias=True, initialize=False)
            )
            self.mlp.append(activation)

            if norm:
                self.mlp.append(nn.IIDBatchNorm1d(activation.out_type))

            hid_type = activation.in_type

        self.mlp.append(nn.Linear(hid_type, out_type, bias=True, initialize=False))

        self.initialize(init_scheme)

    def initialize(self, init_scheme):
        """
        Initializes each linear map using either delta-orthonormal or generalized He initialization scheme.
        More details: https://quva-lab.github.io/escnn/api/escnn.nn.html#module-escnn.nn.init

        Args:
            init_scheme (str): initialization scheme name.
                - 'deltaorthonormal': delta-orthonormal initialization scheme.
                - 'he': generalized He initialization scheme.
        """
        for i in range(len(self.mlp)):
            if self.mlp[i].__class__.__name__ == "Linear":
                if init_scheme == "deltaorthonormal":
                    nn.init.deltaorthonormal_init(
                        self.mlp[i].weights.data, self.mlp[i].basisexpansion
                    )
                else:
                    nn.init.generalized_he_init(
                        self.mlp[i].weights.data, self.mlp[i].basisexpansion
                    )

    def forward(self, x: nn.GeometricTensor):
        """
        Forward pass of the MLP.

        Args:
            x (nn.GeometricTensor): input field with type self.in_type.

        Returns:
            nn.GeometricTensor: output field with type self.out_type.
        """
        assert isinstance(x, nn.GeometricTensor)
        assert x.type == self.in_type
        return self.mlp(x)

    def evaluate_output_shape(self):
        pass
