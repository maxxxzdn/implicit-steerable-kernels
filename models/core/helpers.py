import torch
from torch_geometric.nn import knn, radius

from .point_convolution import ImplicitPointConv, R3PointConv


def get_connector(connectivity: str, n_g_blocks: int, kwargs: dict):
    """
    Returns a connector function and its parameters depending on the connectivity type and its arguments.

    Args:
        connectivity (str): Type of connectivity. Can be one of: 'knn', 'radius', 'given'.
        n_g_blocks (int): Number of G-blocks in the model.
        kwargs (dict): Arguments for the connector function.

    Returns:
        tuple: A tuple containing the connector function and its parameters. If connectivity is 'given',
        the connector function is None and the parameters are None for all graph blocks. Otherwise, the
        connector function is either knn or radius, and the parameters are a list of dictionaries. Each
        dictionary corresponds to a graph block and contains the parameters for the connector function.

    Raises:
        NotImplementedError: If the connectivity type is not one of: 'knn', 'radius', 'given'.
        AssertionError: If the required arguments are not provided.

    """
    if connectivity == 'knn':
        k = kwargs.get('k')
        assert k is not None, "number of neighbors 'k' is not given"
        connector = knn
        # If k is not a list, define a list with n_g_blocks+1 elements where each element is k.
        if len(k) == 1:
            k = k * (n_g_blocks + 1)
        # Create a list of dictionaries, where each dictionary contains the parameter k for a graph block.
        con_params = [{'k': ki} for ki in k]
    elif connectivity == 'radius':
        r = kwargs.get('r')
        max_num_neighbors = kwargs.get('max_num_neighbors')
        assert r is not None, "cut off radius 'r' is not given"
        connector = radius
        # If r is not a list, define a list with n_g_blocks+1 elements where each element is r.
        if len(r) == 1:
            r = r * (n_g_blocks + 1)
        # If max_num_neighbors is not a list, define a list with n_g_blocks+1 elements where each element is max_num_neighbors.
        if len(max_num_neighbors) == 1:
            max_num_neighbors = max_num_neighbors * (n_g_blocks + 1)
        # Create a list of dictionaries, where each dictionary contains the parameters r and max_num_neighbors for a graph block.
        con_params = [{'r': ri, 'max_num_neighbors': mi} for (ri, mi) in zip(r, max_num_neighbors)]
    elif connectivity == 'given':
        connector = None
        con_params = [None] * (n_g_blocks + 1)
    else:
        raise NotImplementedError("Unknown connectivity type: {}".format(connectivity))

    return connector, con_params

def get_convolution(filter_type: str, n_g_blocks: int, edge_dim: int, x_dist, attr_dist, aggr: str, kwargs: dict):
    """
    Returns the point convolution and its parameters based on the filter type.

    Args:
        filter_type (str): The type of filter.
        n_g_blocks (int): Number of G-blocks in the model.
        kwargs (dict): The keyword arguments for the filter.
        edge_dim (int): Dimensionality of edge features.
        x_dist (dist): Distribution of node features.
        attr_dist (dist): Distribution of edge features.        
        aggr (str): Type of aggregation used in the model.

    Returns:
        tuple: A tuple containing the point convolution and its parameters.
    """
    # Check if required arguments are present for the given filter type.
    if filter_type == 'conv':
        assert 'n_rings' in kwargs.keys(), "number of rings for a steerable kernel is not given"
        assert 'width' in kwargs.keys(), "radius of the support of the learnable filters is not given"
        assert 'sigma' in kwargs.keys(), "width of rings for a steerable kernel is not given"
        # If 'n_rings' is a list, create a dictionary of convolution parameters for each element of the list.
        if len(kwargs['n_rings']) == 1:
            conv_params = [{'n_rings': n_i, 'sigma': s_i, 'width': w_i} for (
                n_i, s_i, w_i) in zip(kwargs['n_rings'], kwargs['sigma'], kwargs['width'])]
        # If 'n_rings' is not a list, create a dictionary of convolution parameters for each element of n_g_blocks+1.
        else:
            conv_params = [{'n_rings': kwargs['n_rings'], 'sigma': kwargs['sigma'], 'width': kwargs['width']} for _ in range(n_g_blocks+1)]
        point_conv = R3PointConv
    else:
        assert 'hp_order' in kwargs.keys(), "order of harmonic polynomials for a steerable kernel is not given"
        # Create a dictionary of convolution parameters for each element of hp_order.
        if len(kwargs['hp_order']) != 1:
            hp_order = kwargs['hp_order']
        else:
            hp_order = kwargs['hp_order']*(n_g_blocks+1)
        conv_params = [{'hp_order': h_i, 
                        'scale_coords': kwargs['scale_coords'], 
                        'scale_hp': kwargs['scale_hp'],
                        'n_layers': kwargs['mlp_n_layers'], 
                        'n_channels': kwargs['mlp_n_channels'],
                        'act_fn': kwargs['mlp_act_fn'],
                        'use_tp': kwargs['mlp_use_tp'],
                        'edge_dim': edge_dim,
                        'x_dist': x_dist,
                        'attr_dist': attr_dist,
                        'aggr': aggr,
                        'squash_coords': kwargs['squash_coords'],
                        'squash_hp': kwargs['squash_hp'],
                        'squash_mlp': kwargs['squash_mlp'],
                       } for h_i in hp_order]
        point_conv = ImplicitPointConv
    assert len(conv_params) == n_g_blocks+1, f"number of parameters ({len(conv_params)}) != number of layers ({n_g_blocks+1})"
    return point_conv, conv_params

def get_fc(non_linearity, num_classes, use_bn, kwargs):
    """Constructs a fully connected neural network using PyTorch.

    Args:
        non_linearity (str): Activation function to use after each linear layer.
        num_classes (int): Number of output classes.
        kwargs (dict): Additional keyword arguments:
            - num_inv_channels (int): Number of channels in the input tensor.
            - p_drop (list): Dropout probabilities for the input and hidden layers.
            - use_bn (bool): Whether to use batch normalization after each linear layer.

    Returns:
        torch.nn.Sequential: Fully connected neural network.
    """
    p_drop1, p_drop2 = kwargs['p_drop']
    num_inv_channels = kwargs['num_inv_channels']
    # non-linearity depends on the choice in G-blocks
    if non_linearity == 'swish_gate':
        activation = torch.nn.SiLU()
    elif non_linearity == 'elu':
        activation = torch.nn.ELU()
    else:
        raise NotImplementedError
    # Build fully connected network
    fully_net = torch.nn.Sequential(
        torch.nn.Dropout(p_drop1),
        torch.nn.Linear(num_inv_channels, num_inv_channels),
        torch.nn.BatchNorm1d(num_inv_channels) if use_bn else torch.nn.Identity(),
        activation,
        torch.nn.Dropout(p_drop2),
        torch.nn.Linear(num_inv_channels, num_inv_channels),
        torch.nn.BatchNorm1d(num_inv_channels) if use_bn else torch.nn.Identity(),
        activation,
        torch.nn.Linear(num_inv_channels, num_classes)
    )  
    return fully_net