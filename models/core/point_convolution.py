import torch
from escnn import nn
from escnn.group import Representation
from escnn.kernels import KernelBasis
from escnn.nn import GeometricTensor, FieldType
from escnn.nn.modules.pointconv.r2_point_convolution import compute_basis_params

from typing import Callable, Tuple, Dict, Union, List, Optional
from torch_geometric.typing import OptTensor, Size

from models.core.implicit_kernel import ImplicitKernelSON
from models.core.rd_point_convolution import _RdPointConv

OptTensor = Optional[torch.Tensor]
OptPairGeometricTensor = Tuple[GeometricTensor, Optional[GeometricTensor]]


class ImplicitPointConv(_RdPointConv):
    def __init__(self, in_type: nn.FieldType, out_type: nn.FieldType, 
                 edge_repr: nn.FieldType, hp_order: int, **mlp_kwargs):
        """Implementation of a G-steerable convolution with an implicit kernel."""
        assert in_type.gspace == out_type.gspace
        super().__init__(in_type=in_type, 
                         out_type=out_type, 
                         d=in_type.gspace.dimensionality, 
                         groups=1, 
                         bias=True, 
                         basis_filter=False, 
                         recompute=False,
                         use_implicit=True)
        self.implicit_kernel = ImplicitKernelSON(in_repr=in_type, out_repr=out_type, 
                                                 edge_repr=edge_repr, hp_order=hp_order,
                                                 **mlp_kwargs)
        
    def _build_kernel_basis(self, in_repr: Representation, out_repr: Representation) -> KernelBasis:
        pass

    def message(self, x_j: torch.Tensor, edge_delta: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        _filter = self.implicit_kernel(edge_delta, edge_attr)
        return torch.einsum('noi,ni->no', _filter, x_j)
    
    def forward(self, x: GeometricTensor, edge_index: torch.Tensor, edge_delta: OptTensor, 
                idx_downsampled: torch.Tensor = None, edge_attr: OptTensor = None, size: Size = None):
                
        assert isinstance(x, GeometricTensor)
        assert x.type == self.in_type

        assert len(edge_index.shape) == 2
        assert edge_index.shape[0] == 2

        if idx_downsampled is not None:
            coords = x.coords[idx_downsampled]
            x = (x.tensor, x.tensor[idx_downsampled])
        else:
            coords = x.coords          
            x = (x.tensor, x.tensor)

        out = self.propagate(
            edge_index, 
            x=x, 
            edge_delta=edge_delta, 
            edge_attr=edge_attr, 
            size=size
        )

        if not self.training:
            _bias = self.expanded_bias
        else:
            # retrieve the bias
            _bias = self.expand_bias()

        if _bias is not None:
            out += _bias

        out = GeometricTensor(out, self.out_type, coords=coords)
        
        return out
    
class R3PointConv(_RdPointConv):
    
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 groups: int = 1,
                 bias: bool = True,
                 sigma: Union[List[float], float] = None,
                 width: float = None,
                 n_rings: int = None,
                 frequencies_cutoff: Union[float, Callable[[float], int]] = None,
                 rings: List[float] = None,
                 basis_filter: Callable[[dict], bool] = None,
                 recompute: bool = False,
                 initialize: bool = True,
                 ):

        basis_filter, self._rings, self._sigma, self._maximum_frequency = compute_basis_params(
            frequencies_cutoff, rings, sigma, width, n_rings, basis_filter
        )

        super(R3PointConv, self).__init__(
            in_type, out_type,
            d=3,
            groups=groups,
            bias=bias,
            basis_filter=basis_filter,
            recompute=recompute
        )

        if initialize:
            # by default, the weights are initialized with a generalized form of He's weight initialization
            nn.init.generalized_he_init(self.weights.data, self.basissampler)

    def _build_kernel_basis(self, in_repr: Representation, out_repr: Representation) -> KernelBasis:
        return self.space.build_kernel_basis(in_repr, out_repr,
                                             self._sigma, self._rings,
                                             maximum_frequency=self._maximum_frequency
                                             )
    
    def forward(self, x: GeometricTensor, idx_downsampled: torch.Tensor, edge_index: torch.Tensor, 
                edge_delta: OptTensor, edge_attr: OptTensor = None, size: Size = None):
        
        assert isinstance(x, GeometricTensor)
        assert x.type == self.in_type

        assert len(edge_index.shape) == 2
        assert edge_index.shape[0] == 2

        out = self.propagate(
            edge_index, 
            x=(x.tensor, x.tensor[idx_downsampled]), 
            edge_delta=edge_delta, 
            size=size
        )

        if not self.training:
            _bias = self.expanded_bias
        else:
            # retrieve the bias
            _bias = self.expand_bias()

        if _bias is not None:
            out += _bias

        out = GeometricTensor(out, self.out_type, coords=x.coords[idx_downsampled])
        
        return out