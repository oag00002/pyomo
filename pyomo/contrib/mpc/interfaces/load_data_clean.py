# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Safe variants of load_data functions for use with clean_model='delete' models.

When dae.collocation is called with clean_model='delete', VarData entries at
non-collocation (finite-element boundary) points are permanently removed from the
model.  The standard load functions in load_data.py access variables with var[t],
which triggers Pyomo's __missing__ protocol and silently re-creates deleted entries.

The functions here guard every write with ``t in var`` (which is safe for both plain
IndexedVar and Reference objects) so that deleted entries are simply skipped rather
than re-created.

These functions are intended to be routed to by DynamicModelInterface when its
clean_model flag is True.  They are parallel implementations — the originals in
load_data.py are left completely unchanged.
"""

from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable
from pyomo.contrib.mpc.data.find_nearest_index import (
    find_nearest_index,
    find_nearest_interval_index,
)


def _raise_invalid_cuid(cuid, model):
    raise RuntimeError("Cannot find a component %s on block %s" % (cuid, model))


def load_data_from_scalar_clean(data, model, time):
    """Safe variant of load_data_from_scalar for clean_model='delete' models.

    Identical to load_data_from_scalar except that, for indexed variables, each
    time point is checked with ``t in var`` before writing.  Time points whose
    VarData entries were deleted by clean_model='delete' are silently skipped.

    Arguments
    ---------
    data: ~scalar_data.ScalarData
    model: BlockData
    time: Iterable

    """
    data = data.get_data()
    t_iter = time if _is_iterable(time) else (time,)
    for cuid, val in data.items():
        var = model.find_component(cuid)
        if var is None:
            _raise_invalid_cuid(cuid, model)
        if var.is_indexed():
            for t in t_iter:
                if t in var:
                    var[t].set_value(val)
        else:
            var.set_value(val)


def load_data_from_series_clean(data, model, time, tolerance=0.0):
    """Safe variant of load_data_from_series for clean_model='delete' models.

    Identical to load_data_from_series except that each time point is checked
    with ``t in var`` before writing.  Deleted entries are silently skipped.

    Arguments
    ---------
    data: TimeSeriesData
    model: BlockData
    time: Iterable

    """
    time_list = list(time)
    time_indices = [
        find_nearest_index(time_list, t, tolerance=tolerance)
        for t in data.get_time_points()
    ]
    for idx, t in zip(time_indices, data.get_time_points()):
        if idx is None:
            raise RuntimeError("Time point %s not found time set" % t)
    if len(time_list) != len(data.get_time_points()):
        raise RuntimeError(
            "TimeSeriesData object and model must have same number"
            " of time points to load data from series"
        )
    data = data.get_data()
    for cuid, vals in data.items():
        var = model.find_component(cuid)
        if var is None:
            _raise_invalid_cuid(cuid, model)
        for idx, val in zip(time_indices, vals):
            t = time_list[idx]
            if t in var:
                var[t].set_value(val)


def load_data_from_interval_clean(
    data,
    model,
    time,
    tolerance=0.0,
    prefer_left=True,
    exclude_left_endpoint=True,
    exclude_right_endpoint=False,
):
    """Safe variant of load_data_from_interval for clean_model='delete' models.

    Identical to load_data_from_interval except that each time point is checked
    with ``t in var`` before writing.  Deleted entries are silently skipped.

    Arguments
    ---------
    data: IntervalData
    model: BlockData
    time: Iterable
    tolerance: Float
    prefer_left: Bool
    exclude_left_endpoint: Bool
    exclude_right_endpoint: Bool

    """
    if prefer_left and exclude_right_endpoint and not exclude_left_endpoint:
        raise RuntimeError(
            "Cannot use prefer_left=True with exclude_left_endpoint=False"
            " and exclude_right_endpoint=True."
        )
    elif not prefer_left and exclude_left_endpoint and not exclude_right_endpoint:
        raise RuntimeError(
            "Cannot use prefer_left=False with exclude_left_endpoint=True"
            " and exclude_right_endpoint=False."
        )
    intervals = data.get_intervals()
    left_endpoints = [t for t, _ in intervals]
    right_endpoints = [t for _, t in intervals]
    # NOTE: O(len(time)*log(len(intervals)))
    idx_list = [
        find_nearest_interval_index(
            intervals, t, tolerance=tolerance, prefer_left=prefer_left
        )
        for t in time
    ]
    left_endpoint_indices = [
        find_nearest_index(left_endpoints, t, tolerance=tolerance)
        for t in time
    ]
    right_endpoint_indices = [
        find_nearest_index(right_endpoints, t, tolerance=tolerance)
        for t in time
    ]

    # Post-process indices to exclude endpoints
    for i, t in enumerate(time):
        if (
            exclude_left_endpoint
            and left_endpoint_indices[i] is not None
            and right_endpoint_indices[i] is None
        ):
            idx_list[i] = None
        elif (
            exclude_right_endpoint
            and right_endpoint_indices[i] is not None
            and left_endpoint_indices[i] is None
        ):
            idx_list[i] = None
        elif (
            exclude_left_endpoint
            and exclude_right_endpoint
            and right_endpoint_indices[i] is not None
            and left_endpoint_indices[i] is not None
        ):
            idx_list[i] = None

    data = data.get_data()
    for cuid, vals in data.items():
        var = model.find_component(cuid)
        if var is None:
            _raise_invalid_cuid(cuid, model)
        for i, t in zip(idx_list, time):
            if i is None:
                continue
            elif t in var:
                var[t].set_value(vals[i])
