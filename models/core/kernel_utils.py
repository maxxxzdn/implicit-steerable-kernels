import torch


def compute_scalar_shell(x, sigma):
    """
    Compute scalar shell for the output of the kernel given a vector.

    Args:
        x (torch.Tensor): The input vector of shape (N, 1, dim).
        sigma (torch.Tensor): The array of kernel widths of shape (1,1,*).
            - the last dimension can be 2**algebra.dim or 1.

    Returns:
        torch.Tensor: The output scalar of shape (N, 1, 1).
    """
    assert len(x.shape) == 3, "The input tensor must have shape (N, 1, dim)."
    assert (
        len(sigma.shape) == 3 and sigma.shape[0] == 1
    ), "The sigma tensor must have shape (1, 1, *)."
    norm = torch.norm(x, dim=-1, keepdim=True)
    return torch.exp(-0.5 * (norm / sigma) ** 2)


def generate_kernel_grid(kernel_size, dim):
    """
    Generate the 2D or 3D grid for a given kernel size in PyTorch.

    Args:
        kernel_size (int): The size of the kernel.
        dim (int): The dimension of the grid.

    Returns:
        torch.Tensor: The grid of shape (kernel_size ** dim, dim) defined on the range [-1, 1]^dim.
    """
    axes = [torch.arange(0, kernel_size) for _ in range(dim)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    grid = grid - kernel_size // 2
    return grid.reshape(-1, dim) / max(kernel_size // 2, 1.0)
