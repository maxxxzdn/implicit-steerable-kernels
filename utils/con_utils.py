import torch
from typing import Tuple, Union

def sample_indices(batch, N):
    """
    Sample N indices from each group in a batch.
    """
    indices = torch.arange(len(batch))
    groups = torch.unique(batch)
    if N != -1:
        return torch.cat([indices[batch == group][torch.randperm(len(indices[batch == group]))[:N]] for group in groups])
    else:
        return torch.cat([indices[batch == group] for group in groups])


def get_batch(batch_size: int, num_nodes_in: int) -> torch.Tensor:
    """
    Returns a column vector which maps each point to its respective point cloud in the batch.
    Note: it assumes that each point cloud has the same number of points 'num_nodes_in'.
    """
    return torch.repeat_interleave(torch.arange(batch_size), num_nodes_in)


def get_downsampled_idx(batch_size: int, num_nodes_in: int, num_nodes_out: int) -> torch.Tensor:
    """For each point cloud in a batch returns indices of randomly sampled points."""
    assert num_nodes_out <= num_nodes_in, "output node number shoud be equal ot smaller than input."
    if num_nodes_in > num_nodes_out:
        node_idx = torch.randperm(num_nodes_in)[:num_nodes_out]
    else:
        node_idx = torch.arange(num_nodes_in)
    return node_idx.repeat(batch_size) + \
        get_batch(batch_size, num_nodes_out) * num_nodes_in


def get_delta(coords: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
              edge_index: torch.Tensor) -> torch.Tensor:
    """
    Calculate distance between connected nodes of a graph.
    Coordinates of nodes of a bipartite graph must be given in a tuple (from, to).
    """
    rows, cols = edge_index
    if isinstance(coords, tuple):
        return coords[0][rows] - coords[1][cols]
    else:
        return coords[rows] - coords[cols]