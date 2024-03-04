from escnn import nn, gspaces, group


def get_elu(gspace: gspaces, L: int, channels: int):
    """
    Given group space, maximum frequence of representation and the number of channels, returns group-specific ELU-like nonlinearity.

    Args:
        gspace (gspaces.GSpace): group space.
        L (int): maximum frequency of the representation.
        channels (int): number of channels.

    Returns:
        nn.Module: group-specific ELU-like nonlinearity.
    """
    G = gspace.fibergroup
    N = 40 if L == 3 else 125

    if G == group.so3_group():
        irreps = G.bl_sphere_representation(L=L).irreps
        irreps = list(set(irreps))
        activation = nn.QuotientFourierELU(
            gspace=gspace,
            channels=channels,
            subgroup_id=(False, -1),
            irreps=irreps,
            inplace=True,
            type="ico",
            N=N,
        )
    elif G == group.o3_group():
        irreps = G.bl_sphere_representation(L=L).irreps
        irreps = list(set(irreps))
        activation = nn.QuotientFourierELU(
            gspace=gspace,
            channels=channels,
            subgroup_id=(False, False, -1),
            irreps=irreps,
            inplace=True,
            type="ico",
            N=N,
        )
    elif G == group.so2_group():
        irreps = G.bl_regular_representation(L=L).irreps
        irreps = list(set(irreps))
        activation = nn.FourierELU(
            gspace=gspace,
            channels=channels,
            irreps=irreps,
            inplace=True,
            type="regular",
            N=N,
        )
    elif G == group.o2_group():
        irreps = G.bl_regular_representation(L=L).irreps
        irreps = list(set(irreps))
        activation = nn.FourierELU(
            gspace=gspace,
            channels=channels,
            irreps=irreps,
            inplace=True,
            type="regular",
            N=N,
        )
    elif G == group.cyclic_group(1) or G == group.cyclic_group(2):
        irreps = [ir.id for ir in G.irreps()]
        irreps = list(set(irreps))
        activation = nn.FourierPointwise(
            gspace=gspace,
            channels=channels,
            irreps=irreps,
            inplace=True,
            type="regular",
            N=len(irreps),
        )
    elif G == group.cylinder_group(maximum_frequency=3):
        irreps = [((f,), (l,)) for f in range(2) for l in range(L)]
        irreps = list(set(irreps))
        activation = nn.FourierELU(
            gspace=gspace,
            channels=channels,
            irreps=irreps,
            inplace=True,
            type="regular",
            N=N,
            G1_type="regular",
            G1_N=2,
            G2_type="regular",
            G2_N=N,
        )
    else:
        raise NotImplementedError(
            "{} is not supported! Only O(3), SO(2) and SO(3) are supported.".format(G)
        )
    return activation
