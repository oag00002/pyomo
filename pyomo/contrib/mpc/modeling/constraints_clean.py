# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Safe variant of get_piecewise_constant_constraints for clean_model='delete' models.

See pyomo/contrib/mpc/interfaces/load_data_clean.py for background.

With clean_model='delete', VarData entries at non-collocation (finite-element
boundary) points are removed.  The standard piecewise_constant_rule tries to form
``var[t] - var[t_next] == 0`` at every non-sample-point, which re-creates deleted
entries when t or t_next is a deleted FE boundary.

The clean variant handles this in two ways:
  1. If ``t not in var``, skip the constraint at t (the deleted entry cannot
     participate in any equality chain).
  2. If ``t in var`` but the immediate neighbor (t_next/t_prev) has been deleted,
     walk forward (or backward) through the time set until an existing entry or a
     sample-point boundary is found.  The found point becomes the partner in the
     equality constraint.

This preserves the mathematical meaning of piecewise-constant inputs: all existing
variable entries within a sample interval are constrained equal to one another.
"""

from pyomo.core.base.constraint import Constraint
from pyomo.core.base.set import Set


def get_piecewise_constant_constraints_clean(
    inputs, time, sample_points, use_next=True
):
    """Safe variant of get_piecewise_constant_constraints for clean_model='delete'.

    Returns an IndexedConstraint that constrains the provided variables to be
    constant between sample points, skipping any time points whose VarData entries
    have been deleted and walking over deleted neighbors to maintain the equality
    chain.

    Arguments
    ---------
    inputs: list of variables
        Time-indexed variables that will be constrained piecewise constant.
    time: Set
        Set of points at which provided variables will be constrained.
    sample_points: List of floats
        Points at which "constant constraints" will be omitted; these are points
        at which the provided variables may vary.
    use_next: Bool (default True)
        Whether the next time point will be used in the constant constraint at
        each point in time.  Otherwise, the previous time point is used.

    Returns
    -------
    Set, IndexedConstraint
        A RangeSet indexing the list of variables provided and a Constraint
        indexed by the product of this RangeSet and time.

    """
    input_set = Set(initialize=range(len(inputs)))
    sample_point_set = set(sample_points)

    def piecewise_constant_rule(m, i, t):
        if t in sample_point_set:
            return Constraint.Skip
        var = inputs[i]
        # If this time point's VarData was deleted, there is nothing to constrain.
        if t not in var:
            return Constraint.Skip
        if use_next:
            t_next = time.next(t)
            # Walk forward past deleted interior points.  Stop when we hit an
            # existing entry, a sample boundary, or the end of the time set.
            while (
                t_next not in var
                and t_next not in sample_point_set
                and t_next != time.last()
            ):
                t_next = time.next(t_next)
            # If the walk ended on another deleted point (e.g. time.last() is also
            # deleted, which can happen for some variable types), skip.
            if t_next not in var:
                return Constraint.Skip
            return var[t] - var[t_next] == 0
        else:
            t_prev = time.prev(t)
            # Walk backward past deleted interior points.
            while (
                t_prev not in var
                and t_prev not in sample_point_set
                and t_prev != time.first()
            ):
                t_prev = time.prev(t_prev)
            if t_prev not in var:
                return Constraint.Skip
            return var[t_prev] - var[t] == 0

    pwc_con = Constraint(input_set, time, rule=piecewise_constant_rule)
    return input_set, pwc_con
