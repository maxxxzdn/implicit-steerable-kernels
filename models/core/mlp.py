import random
import torch
import numpy as np
from escnn import nn, group, gspaces

from utils.utils import get_elu


class SONMLPTensor(nn.EquivariantModule):
    def __init__(self, in_type: nn.FieldType, in_repr: nn.FieldType, out_repr: nn.FieldType, initialize: bool, **mlp_kwargs):
        """
        MLP that is equivariant to the group SO(N). It takes the coordinates of a point in the 3D space as input.
        Note: we assume that in_repr and out_repr contain c1, c2 copies of the same representations (ψ1, ψ2 respectively).
        Args:
            in_type (nn.FieldType): type of the input field.
            in_repr (nn.FieldType): representation of the input field.
            out_repr (nn.FieldType): representation of the output field.
            initialize (bool): whether to initialize the MLP.
            **mlp_kwargs (dict): keyword arguments for the MLP.
        """
        super().__init__()
        assert in_repr.uniform and out_repr.uniform
        self.G = in_type.fibergroup
        self.gspace = gspaces.no_base_space(self.G)
        self.in_type = in_type
        self.mlp = self.set_mlp(
            in_repr=in_repr, 
            out_repr=out_repr, 
            n_layers=mlp_kwargs['n_layers'], 
            n_channels=mlp_kwargs['n_channels'])
        self.out_type = self.mlp.out_type
        
        if initialize:
            self.initialize() 
  
    def set_mlp(self, in_repr, out_repr, n_layers, n_channels):
        """
        Initializes the MLP that spits out a steerable filter.
        Args:
            in_repr (nn.FieldType): representation of the input field.
            out_repr (nn.FieldType): representation of the output field.
            n_layers (int): number of layers in the MLP.
            n_channels (int): number of channels in the MLP.
            use_tp (bool): whether to use tensor product layers.
        """
        mlp = nn.SequentialModule()
        tmp = out_repr.representations[0].tensor(in_repr.representations[0]) # ψ1 ⊗ ψ2
        out_type = self.gspace.type(*[tmp] * (len(out_repr) * len(in_repr))) # ⊗^(c1*c2) ψ1 ⊗ ψ2
        
        hid_type = self.in_type     
         
        if n_layers > 1:
            L = len(set(self.in_type.irreps)) - 1
            activation = get_elu(gspace=self.gspace, L=L, channels=n_channels)
            
        for _ in range(n_layers-1):
            mlp.append(nn.Linear(hid_type, activation.in_type, initialize = False, bias=True))
            mlp.append(activation)
            hid_type = activation.in_type
        
        mlp.append(nn.Linear(hid_type, out_type, initialize=False, bias=True))
        
        return mlp
    
    def initialize(self):
        """
        Initializes each linear map using delta-orthonormal initialization scheme.
        More details: https://quva-lab.github.io/escnn/api/escnn.nn.html#module-escnn.nn.init
        """
        for i in range(len(self.mlp)):
            try:
                nn.init.deltaorthonormal_init(self.mlp[i].weights.data, self.mlp[i].basisexpansion)
                #nn.init.generalized_he_init(self.mlp[i].weights.data, self.mlp[i].basisexpansion)
            except:
                pass
     
    def forward(self, x: nn.GeometricTensor):
        assert isinstance(x, nn.GeometricTensor)
        assert x.type == self.in_type
        return self.mlp(x)
    
    def evaluate_output_shape(self, input_shape: tuple):
        shape = list(input_shape)
        assert len(shape) == 2, shape
        assert shape[1] == self.in_type.size, shape
        shape[1] = self.out_type.size
        return shape
    
    def assert_equivariance(self, atol: float = 5e-5, rtol: float = 1e-5):
        """
        Asserts the equivariance of the MLP to randomly sampled group elements.
        Args:
            atol (float): absolute tolerance.
            rtol (float): relative tolerance.
        """
        device = next(self.parameters()).device
        # random input coordinates uniformly sampled from [-1,1]^3
        x = 2*torch.rand(100, self.in_type.size) - 1
        x = nn.GeometricTensor(x.to(device), self.in_type)
        for _ in range(10):
            with torch.no_grad():
                # we check the equivarince for a randomly chosen field due to the high computation cost
                field_id = random.choice(range(len(self.out_type)))
                g_el = self.G.sample()
                # g @ (MLP(x) [:, random field])
                out1 = self(x)[:, field_id].transform(g_el).tensor
                # MLP(g @ x) [:, random field]
                out2 = self(x.transform(g_el))[:, field_id].tensor
                # assert that g @ MLP(x) ≈ MLP(g @ x) w.r.t L1 distance 
                assert torch.allclose(out1, out2, atol=atol, rtol=rtol), print(torch.max(out1-out2))