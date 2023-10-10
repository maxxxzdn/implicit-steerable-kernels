import torch
import torch.distributions as dist
import numpy as np
from escnn import nn, gspaces, group
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme


class Projector(nn.EquivariantModule):
    def __init__(self, in_type: nn.FieldType, out_type: nn.FieldType):
        super().__init__()
        G = in_type.gspace.fibergroup
        gspace = gspaces.no_base_space(G)
        self.hid_type1 = gspace.type(*in_type.representations)
        self.hid_type2 = gspace.type(*out_type.representations)
        self.linear = nn.Linear(self.hid_type1, self.hid_type2) 
        self.out_type = out_type
    
    def forward(self, x):
        x, coords = x.tensor, x.coords
        x = self.hid_type1(x)
        x = self.linear(x)
        x = nn.GeometricTensor(x.tensor, self.out_type, coords)
        return x
    
    def evaluate_output_shape(self, input_shape):
        return super().evaluate_output_shape(input_shape)
    

class BatchGeometricTensor():
    def __init__(self, gt: nn.GeometricTensor, batch: torch.Tensor, node_features: torch.Tensor = None,
                 edge_index: torch.Tensor = None, edge_attr: torch.Tensor = None):
        assert batch.dtype == torch.int64
        assert len(batch) == len(gt.tensor)
        self.gt = gt
        self.batch = batch
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_attr = edge_attr
           
    @property
    def tensor(self):
        return self.gt.tensor
    
    @property
    def device(self):
        return self.gt.tensor.device
    
    @property
    def type(self):
        return self.gt.type
    
    @property
    def coords(self):
        return self.gt.coords
    
    @property
    def batch_size(self):
        return self.batch.max().item() + 1
    
    def clone(self):
        return BatchGeometricTensor(self.gt, self.batch, self.edge_index, self.node_features)
    
    def __add__(self, x):
        if self.edge_index is not None or x.edge_index is not None: 
            assert torch.allclose(self.edge_index, x.edge_index)
        if self.node_features is not None or x.node_features is not None: 
            assert torch.allclose(self.node_features, x.node_features)
        if self.edge_attr is not None or x.edge_attr is not None: 
            assert torch.allclose(self.edge_attr, x.edge_attr)
        assert torch.allclose(self.batch, x.batch)
        return BatchGeometricTensor(self.gt + x.gt, self.batch, self.node_features, self.edge_index, self.edge_attr)
    
    def __getitem__(self, indices):
        # slicing works only when 1) edge_index is not given (point cloud) or 
        #                         2) no downsampling is applied.
        assert self.edge_index is None or len(indices) == len(self.gt.coords)
        tensor = self.gt.tensor[indices]
        if self.gt.coords is not None:
            coords = self.gt.coords[indices]
        batch = self.batch[indices]
        gt = nn.GeometricTensor(tensor, self.gt.type, coords)
        return BatchGeometricTensor(gt, batch, self.node_features, self.edge_index, self.edge_attr)

progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="black",
        progress_bar="green",
        progress_bar_finished="green",
        progress_bar_pulse="#6206E0",
        batch_progress="green",
        time="gray48",
        processing_speed="gray48",
        metrics="gray48",
    )
)

def get_gated(in_type, channels, gated_type=None):
    if gated_type is None:      
        G = in_type.fibergroup
        gs = in_type.gspace
        repr = in_type.representation
        gates_and_repr = group.directsum(([G.trivial_representation]*len(repr.irreps))) + repr
        gated_type = in_type.gspace.type(*[gates_and_repr]*channels)
    return nn.GatedNonLinearityUniform(gated_type, gate=torch.nn.functional.silu)

def get_elu(gspace: gspaces, L: int, channels: int):
    G = gspace.fibergroup
    N = 40 if L == 3 else 25
    if G == group.so3_group():
        irreps = G.bl_sphere_representation(L=L).irreps
        activation = nn.QuotientFourierELU(
            gspace=gspace, subgroup_id=(False,-1), channels=channels, 
            irreps=irreps, grid=G.sphere_grid(type='thomson', N=N), inplace=True)
    elif G == group.so2_group():
        irreps = G.bl_regular_representation(L=L).irreps
        irreps = list(set(irreps))
        activation = nn.FourierELU(
            gspace=gspace, channels=channels,
            irreps=irreps, inplace=True, type='regular', N=N)
    elif G == group.o2_group():
        irreps = G.bl_regular_representation(L=L).irreps
        irreps = list(set(irreps))
        activation = nn.FourierELU(
            gspace=gspace, channels=channels,
            irreps=irreps, inplace=True, type='regular', N=N)
    elif G == group.o3_group():
        irreps = G.bl_sphere_representation(L=L).irreps
        activation = nn.QuotientFourierELU(
            gspace=gspace, subgroup_id=('cone', -1), channels=channels, 
            irreps=irreps, grid=G.sphere_grid(type='thomson', N=N), inplace=True)
    elif G == group.cyclic_group(1) or G == group.cyclic_group(2):
        irreps = [ir.id for ir in G.irreps()]
        activation = nn.FourierPointwise(
            gspace=gspace, channels=channels, irreps=irreps, inplace=True, type='regular', N=len(irreps))
    elif G == group.cylinder_group(maximum_frequency=3):
        irreps = [
            ((f,), (l,))
            for f in range(2)
            for l in range(L)
        ]
        activation = nn.FourierELU(
            gspace=gspace, channels=channels,
            irreps=irreps, inplace=True, type='regular', N=N, G1_type='regular', G1_N=2, G2_type='regular', G2_N=N)
    else:
        raise NotImplementedError("{} is not supported! Only O(3), SO(2) and SO(3) are supported.".format(G))
    return activation

class sumNormal():
    def __init__(self, mean=list, var=list):
        self.dists = [dist.Normal(mean, var) for mean, var in zip(mean,var)]
        
    def sample(self, sample_shape=torch.Size([])):
        sample_shape[0] = int(sample_shape[0] / len(self.dists)) # half from each distribution
        return torch.cat([dist.sample(sample_shape).reshape(sample_shape[0], -1) for dist in self.dists],0)
    
    def __str__(self):
        return "Joint distribution of {}".format(self.dists)
    
class sumDist():
    def __init__(self, dists):
        self.dists = dists
        
    def sample(self, sample_shape=torch.Size([])):
        return torch.cat([dist.sample(sample_shape) for dist in self.dists],1)
    
    def __str__(self):
        return "Joint distribution of {}".format(self.dists)
    
class normBeta():
    def __init__(self, a, b):
        self.dist = dist.Beta(a, b)
        
    def sample(self, sample_shape=torch.Size([])):
        sample = self.dist.sample(sample_shape)
        return 2*sample - 1
    
    def __str__(self):
        return "Normalized {}".format(self.dist)
    
def f1_max(pred, target):
    """
    F1 score with the optimal threshold.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()
