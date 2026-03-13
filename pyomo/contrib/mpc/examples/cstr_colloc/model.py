# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""CSTR model discretized with dae.collocation (LAGRANGE-RADAU).

This module mirrors pyomo/contrib/mpc/examples/cstr/model.py but uses
``dae.collocation`` instead of ``dae.finite_difference``.  It exists to
demonstrate and test the ``clean_model`` flag on ``DynamicModelInterface``.

The ``clean_model`` parameter accepted by ``initialize_model`` and
``create_instance`` is passed directly to ``dae.collocation``'s ``clean_model``
CONFIG option ('none', 'deactivate', or 'delete').
"""

import pyomo.environ as pyo
import pyomo.dae as dae


def _flow_eqn_rule(m, t):
    return m.flow_in[t] - m.flow_out[t] == 0


def _conc_out_eqn_rule(m, t, j):
    return m.conc[t, j] - m.conc_out[t, j] == 0


def _rate_eqn_rule(m, t, j):
    return m.rate_gen[t, j] - m.stoich[j] * m.k_rxn * m.conc[t, "A"] == 0


def _conc_diff_eqn_rule(m, t, j):
    return (
        m.dcdt[t, j]
        - (
            m.flow_in[t] * m.conc_in[t, j]
            - m.flow_out[t] * m.conc_out[t, j]
            + m.rate_gen[t, j]
        )
        == 0
    )


def _conc_steady_eqn_rule(m, t, j):
    return (
        m.flow_in[t] * m.conc_in[t, j]
        - m.flow_out[t] * m.conc_out[t, j]
        + m.rate_gen[t, j]
    ) == 0


def make_model(dynamic=True, horizon=10.0):
    m = pyo.ConcreteModel()
    m.comp = pyo.Set(initialize=["A", "B"])
    if dynamic:
        m.time = dae.ContinuousSet(initialize=[0, horizon])
    else:
        m.time = pyo.Set(initialize=[0])
    time = m.time
    comp = m.comp

    m.stoich = pyo.Param(m.comp, initialize={"A": -1, "B": 1}, mutable=True)
    m.k_rxn = pyo.Param(initialize=1.0, mutable=True)

    m.conc = pyo.Var(m.time, m.comp)
    if dynamic:
        m.dcdt = dae.DerivativeVar(m.conc, wrt=m.time)

    m.flow_in = pyo.Var(time, bounds=(0, None))
    m.flow_out = pyo.Var(time, bounds=(0, None))
    m.flow_eqn = pyo.Constraint(time, rule=_flow_eqn_rule)

    m.conc_in = pyo.Var(time, comp, bounds=(0, None))
    m.conc_out = pyo.Var(time, comp, bounds=(0, None))
    m.conc_out_eqn = pyo.Constraint(time, comp, rule=_conc_out_eqn_rule)

    m.rate_gen = pyo.Var(time, comp)
    m.rate_eqn = pyo.Constraint(time, comp, rule=_rate_eqn_rule)

    if dynamic:
        m.conc_diff_eqn = pyo.Constraint(time, comp, rule=_conc_diff_eqn_rule)
    else:
        m.conc_steady_eqn = pyo.Constraint(time, comp, rule=_conc_steady_eqn_rule)

    return m


def initialize_model(m, dynamic=True, ntfe=None, ncp=3, clean_model='none'):
    """Discretize and initialize the CSTR model using collocation.

    Parameters
    ----------
    m: ConcreteModel
    dynamic: bool
    ntfe: int (optional)
        Number of finite elements.  Defaults to 10 for dynamic models.
    ncp: int (optional)
        Number of collocation points per finite element.  Default 3.
    clean_model: str (optional)
        Passed to dae.collocation's clean_model option: 'none' (default),
        'deactivate', or 'delete'.  Use 'delete' together with
        ``DynamicModelInterface(model, time, clean_model=True)`` to exercise
        the safe-access code paths in contrib.mpc.

    """
    if ntfe is not None and not dynamic:
        raise RuntimeError("Cannot provide ntfe to initialize steady model")
    elif dynamic and ntfe is None:
        ntfe = 10
    if dynamic:
        disc = pyo.TransformationFactory("dae.collocation")
        disc.apply_to(
            m,
            wrt=m.time,
            nfe=ntfe,
            ncp=ncp,
            scheme="LAGRANGE-RADAU",
            clean_model=clean_model,
        )

    t0 = m.time.first()

    # Fix inputs
    m.conc_in[:, "A"].fix(5.0)
    m.conc_in[:, "B"].fix(0.01)
    m.flow_in[:].fix(1.0)
    m.flow_in[t0].fix(0.1)

    if dynamic:
        # Fix initial conditions
        m.conc[t0, "A"].fix(1.0)
        m.conc[t0, "B"].fix(0.0)


def create_instance(dynamic=True, horizon=None, ntfe=None, ncp=3, clean_model='none'):
    """Create and return a discretized, initialized CSTR instance.

    Parameters
    ----------
    dynamic: bool
    horizon: float (optional)
    ntfe: int (optional)
    ncp: int (optional)
    clean_model: str (optional)
        Passed through to initialize_model / dae.collocation.

    """
    if horizon is None and dynamic:
        horizon = 10.0
    if ntfe is None and dynamic:
        ntfe = 10
    m = make_model(horizon=horizon, dynamic=dynamic)
    initialize_model(m, ntfe=ntfe, dynamic=dynamic, ncp=ncp, clean_model=clean_model)
    return m
