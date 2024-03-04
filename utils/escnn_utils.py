def get_tensor_representation(in_repr, out_repr):
    """
    Computes the output representation of an implicit kernel.
    It assumes that the input representation is uniform, i.e. it consists of N copies of the same representation.

    Args:
        in_repr (group.Representation): input field representation ⊗^(c_in) ψ_in
        out_repr (group.Representation): output field representation ⊗^(c_out) ψ_out

    Returns:
        group.Representation: output field representation ⊗^(c_out*c_in) ψ_out ⊗ ψ_in
    """
    c_in, c_out = len(in_repr), len(out_repr)
    psi_out = out_repr[0]
    psi_in = in_repr[0]

    # ψ1 ⊗ ψ2
    out_repr_single = psi_out.tensor(psi_in)
    # ⊗^(c_out*c_in) ψ1 ⊗ ψ2
    out_repr = [out_repr_single] * (c_out * c_in)
    return out_repr


def repr_max_freq(rep_list):
    """
    Computes the maximum frequency appearing in the input representation.

    Args:
        rep_list (list): list of representation (group.Representation).

    Returns:
        int: maximum frequency.
    """
    L_max = -1
    for rep in rep_list:
        for irrep in rep.irreps:

            if rep.group.name in ["O(2)", "O(3)"]:
                if irrep[1] > L_max:
                    L_max = irrep[1]

            elif rep.group.name in ["SO(2)", "SO(3)"]:
                if irrep[0] > L_max:
                    L_max = irrep[0]

    return L_max
