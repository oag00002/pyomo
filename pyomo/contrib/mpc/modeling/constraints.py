# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.core.base.constraint import Constraint
from pyomo.core.base.set import Set


def get_piecewise_constant_constraints(inputs, time, sample_points, use_next=True):
    """Returns an IndexedConstraint that constrains the provided variables
    to be constant between the provided sample points

    Arguments
    ---------
    inputs: list of variables
        Time-indexed variables that will be constrained piecewise constant
    time: Set
        Set of points at which provided variables will be constrained
    sample_points: List of floats
        Points at which "constant constraints" will be omitted; these are
        points at which the provided variables may vary.
    use_next: Bool (default True)
        Whether the next time point will be used in the constant constraint
        at each point in time. Otherwise, the previous time point is used.

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
        if t not in var:
            # This variable has no entry at t (see load_data.py), so there is
            # nothing to hold constant here.
            return Constraint.Skip
        # I think whether we want prev or next here depends on whether
        # we use an explicit or implicit time discretization. I.e. whether
        # an input is applied to the finite element in front of or behind
        # its time point. If the wrong direction for a discretization
        # is used, we could have different inputs applied within the same
        # finite element, which I think we never want.
        if use_next:
            t_next = time.next(t)
            # Walk past any neighbors this variable has no entry at, so that
            # the existing entries within a sample interval remain chained
            # together. On a variable with every entry this loop never runs.
            while (
                t_next not in var
                and t_next not in sample_point_set
                and t_next != time.last()
            ):
                t_next = time.next(t_next)
            if t_next not in var:
                return Constraint.Skip
            return var[t] - var[t_next] == 0
        else:
            t_prev = time.prev(t)
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
