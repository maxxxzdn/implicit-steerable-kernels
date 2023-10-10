import numpy as np
import torch

from escnn import nn, gspaces, group

from .mlp import SONMLPTensor

        
class ImplicitKernelSON(torch.nn.Module):
    """
    Steerable kernel: 
        3D coordinates -> harmonic polynomicals -> + edge attributes -> implicit kernel.
    """
    def __init__(
        self, 
        in_repr: nn.FieldType, 
        out_repr: nn.FieldType, 
        edge_repr: nn.FieldType, 
        hp_order: int = None, 
        edge_distr: list = [None, None],
        **mlp_kwargs):
        """
        Args:
            in_repr (nn.FieldType): representation of the convolution input.
            out_repr (nn.FieldType): representation of the convolution output.
            edge_repr (nn.FieldType): representation of the edge attributes.
            hp_order (int): order of harmonic polynomials.
            mlp_kwargs (dict): keyword arguments for the MLP.
        """
        super().__init__()
        if hp_order is None:
            # if the order of HPs is not given, we take the maximum frequency of the input representation
            ik_irreps = out_repr.representations[0].tensor(in_repr.representations[0]).irreps # ψ1 ⊗ ψ2
            hp_order = max(ik_irreps, key=lambda x: x[1])
        self.edge_distr = edge_distr
        self.G = group.o3_group()
        self.gspace = gspaces.no_base_space(self.G)
        # input: 3D coordinates + additional edge attributes
        self.pos_type = self.gspace.type(self.G.standard_representation())
        self.edge_repr = edge_repr
        self.hp = nn.HarmonicPolynomialR3(hp_order, 'o3')        
        # restrict output of HPs to a subgroup defined by the input representation
        self.subgroup_id = in_repr.gspace._sg_id[:-1]
        # harmonic polynomial output + edge attributes
        mlp_in_repr = [i for i in self.hp.out_type.representations] + edge_repr
        # we restrict the representation of input type and add attribute representations if needed
        mlp_in_type = self.gspace.type(*mlp_in_repr).restrict(self.subgroup_id)
        # batch normalization for input coordinates
        self.bn_coords = nn.IIDBatchNorm1d(self.pos_type)
        # irrep-wise batch normalization for HPs output
        decomposed_type = self.gspace.type(*[self.hp.out_type.fibergroup.irrep(*id) for id in self.hp.out_type.irreps])
        self.bn_hp = nn.IIDBatchNorm1d(decomposed_type)
        # function that applies batch normalization to each irrep independently
        self.scale_hp = lambda x: self.hp.out_type(self.bn_hp(decomposed_type(x.tensor)).tensor)   
        self.mlp = SONMLPTensor(in_type=mlp_in_type, 
                                in_repr=in_repr, 
                                out_repr=out_repr, 
                                initialize=True,
                                **mlp_kwargs)               
        self.c_in, self.c_out  = len(in_repr), len(out_repr)
        self.delta_in, self.delta_out = int(in_repr.size / len(in_repr)), int(out_repr.size / len(out_repr))
        # learn scale of the kernel
        self.inv_scale = torch.nn.Parameter(torch.tensor(1.0))
        # kernel's output initial standard deviation (for generalized He initialization)
        kstd = 1 / self.kernel_std(1000)
        self.register_buffer("inv_std", kstd)  
        # scale parameter          
        self.init_factor = 2 / np.sqrt(self.c_in * self.c_out)

    def forward(self, coords: torch.Tensor, edge_attr: torch.Tensor, init: bool = True):
        """
        Forward pass of the kernel:
            Input -> Scale coords -> HPs -> MLP -> He init -> filter.
        Args:
            coords (torch.Tensor): 3D coordinates of the input points.
            edge_attr (torch.Tensor): edge attributes.
            init (bool): whether to re-scale the filter (corresponds to He initialization).
        """
        x = self.transform_coords(coords)
        if edge_attr is not None:
            x = nn.GeometricTensor(torch.cat([x.tensor, edge_attr], 1), self.mlp.in_type)
        else:
            x = nn.GeometricTensor(x.tensor, self.mlp.in_type) 
        # compute c_in * delta_in * c_out * delta_out filter based on input features
        x = self.mlp(x)
        x = self.transform_mlp(x, coords, init)
        # reshape the vector to d_out x d_in filter at last
        return x.reshape((-1, self.c_out * self.delta_out, self.c_in * self.delta_in))
    
    def transform_coords(self, coords: torch.Tensor):
        """
        Transform coordinates of the input points:
            BatchNorm -> Harmonic polynomials -> scaling.
        """
        coords = nn.GeometricTensor(coords, self.pos_type)
        coords = self.bn_coords(coords) # normalize input coordinates
        x = self.compute_hp(coords)
        x = self.scale_hp(x)
        return x
    
    def transform_mlp(self, x: nn.GeometricTensor, coords: torch.Tensor, init: bool):
        """
        Transform the output of MLP:
            Scale MLP's output -> reshape the output to c_out x delta_out x c_in x delta_in filter -> He init.
        Args:
            x (nn.GeometricTensor): output of the MLP.
            coords (torch.Tensor): 3D coordinates of the input points.
            init (bool): whether to re-scale the filter (corresponds to He initialization).
        """
        # multiply with exp radial function
        x = x.tensor
        x = self.scale_mlp(x, coords)
        x = x.view((-1, self.c_out, self.c_in, self.delta_out, self.delta_in))
        # normalize the filter to yield He weight initialization
        x = self.he_initialize(x) if init else x
        return x.permute((0, 1, 3, 2, 4))
    
    def scale_mlp(self, x, coords):
        """
        Scale MLP's output according to the coordinates of the input point.
        Similar to radial function of SO(3) steerable kernels: https://arxiv.org/abs/1807.02547.
        """
        squared_norms = torch.sum(coords**2, dim=1, keepdim=True)
        # Compute the exponential factor of the operation
        exponential_factor = torch.exp(-0.5 * squared_norms * self.inv_scale**2)
        # Apply the element-wise multiplication of `x` and the exponential factor
        return x * exponential_factor
    
    def compute_hp(self, x: nn.GeometricTensor):
        """
        Compute scaled harmonic polynomials that are restricted to the subgroup of the input representation.
        """
        return self.hp(x).restrict(self.subgroup_id)
    
    def he_initialize(self, x: torch.Tensor):
        """ 
        Generalized He weight initialization of filters calculated by the implicit kernel.
        The initialization condition is taken from https://arxiv.org/pdf/1711.07289.pdf (see Appendix B).
        For the forward pass: Var[w] = 2 / (C * Q) where 
            C is the number of input channels, 
            Q is the number of basis functions,
            filters are normalized to unit norm.
        """
        return x * self.inv_std * self.init_factor
    
    def sample_kernel(self, n_points: int):
        """
        Sample points in space randomly and compute the value of kernel at each point.
        Draws samples from a normal distribution with mean 0 and variance 1.
        """
        device = next(self.parameters()).device
        with torch.no_grad():
            # randomly generated coordinates in 3D
            if self.edge_distr[0] is None:
                coords = torch.randn(n_points, 3).to(device)
            else: 
                coords = self.edge_distr[0].sample([n_points]).to(device)
                
            if self.edge_distr[1] is None:
                edge_attr = torch.randn(n_points, sum(reps.size for reps in self.edge_repr)).to(device)
            else:
                edge_attr = self.edge_distr[1].sample([n_points]).to(device)
                
            return self(coords, edge_attr, init=False)
    
    def kernel_std(self, n_points: int):
        """
        Computes channel-wise variance for the implicit kernel;
        asserts that there are no zero variance channels.
        """
        x = self.sample_kernel(n_points)
        x = x.reshape((-1, self.c_out, self.delta_out, self.c_in, self.delta_in))
        return x.std(dim=(0,2,4)).reshape(1, self.c_out, self.c_in, 1, 1)