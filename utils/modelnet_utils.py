import torch


def sample_indices(batch, N):
    """
    Sample N indices from each group in a batch.

    Args:
        batch (torch.Tensor): batch vector indicating the group of each point.
        N (int): number of indices to sample from each group.

    Returns:
        torch.Tensor: indices of sampled points.
    """
    indices = torch.arange(len(batch))
    groups = torch.unique(batch)
    if N != -1:
        return torch.cat(
            [
                indices[batch == group][
                    torch.randperm(len(indices[batch == group]))[:N]
                ]
                for group in groups
            ]
        )
    else:
        return torch.cat([indices[batch == group] for group in groups])


def get_batch(batch_size: int, num_nodes_in: int) -> torch.Tensor:
    """
    Returns a column vector which maps each point to its respective point cloud in the batch.
    Note: it assumes that each point cloud has the same number of points 'num_nodes_in'.

    Args:
        batch_size (int): number of point clouds in the batch.
        num_nodes_in (int): number of points in each point cloud.

    Returns:
        torch.Tensor: column vector of size (batch_size * num_nodes_in).
    """
    return torch.repeat_interleave(torch.arange(batch_size), num_nodes_in)


def get_downsampled_idx(
    batch_size: int, num_nodes_in: int, num_nodes_out: int
) -> torch.Tensor:
    """
    For each point cloud in a batch returns indices of randomly sampled points.

    Args:
        batch_size (int): number of point clouds in the batch.
        num_nodes_in (int): number of points in each point cloud.
        num_nodes_out (int): number of points in each downsampled point cloud.
    """
    assert (
        num_nodes_out <= num_nodes_in
    ), "output node number shoud be equal ot smaller than input."
    if num_nodes_in > num_nodes_out:
        node_idx = torch.randperm(num_nodes_in)[:num_nodes_out]
    else:
        node_idx = torch.arange(num_nodes_in)
    return (
        node_idx.repeat(batch_size)
        + get_batch(batch_size, num_nodes_out) * num_nodes_in
    )
