# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Safe variant of copy_values_at_time for use with clean_model='delete' models.

See load_data_clean.py for background.  The standard copy_values_at_time accesses
source and target variables with var[t], which re-creates deleted VarData entries.
This variant guards both source and target accesses with ``t in var``.
"""

iterable_scalars = (str, bytes)


def _to_iterable(item):
    if hasattr(item, "__iter__"):
        if isinstance(item, iterable_scalars):
            yield item
        else:
            for obj in item:
                yield obj
    else:
        yield item


def copy_values_at_time_clean(
    source_vars, target_vars, source_time_points, target_time_points
):
    """Safe variant of copy_values_at_time for clean_model='delete' models.

    Identical to copy_values_at_time except that each (source_time, target_time)
    pair is only copied when both VarData entries exist in their respective
    variables.  Pairs involving deleted entries are silently skipped.
    """
    source_time_points = list(_to_iterable(source_time_points))
    target_time_points = list(_to_iterable(target_time_points))
    if (
        len(source_time_points) != len(target_time_points)
        and len(source_time_points) != 1
    ):
        raise ValueError(
            "copy_values_at_time can only copy values when lists of time\n"
            "points have the same length or the source list has length one."
        )
    n_points = len(target_time_points)
    if len(source_time_points) == 1:
        source_time_points = source_time_points * n_points
    for s_var, t_var in zip(source_vars, target_vars):
        for s_t, t_t in zip(source_time_points, target_time_points):
            if s_t in s_var and t_t in t_var:
                t_var[t_t].set_value(s_var[s_t].value)
