# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""MPC example using a CSTR model discretized with dae.collocation.

This module mirrors pyomo/contrib/mpc/examples/cstr/run_mpc.py but uses
the collocation-based CSTR model from cstr_colloc/model.py and demonstrates
the ``clean_model`` flag on ``DynamicModelInterface``.

The ``dae_clean_model`` parameter controls both the dae.collocation cleanup mode
and (when 'delete') the safe-access routing in DynamicModelInterface.  Pass
``dae_clean_model='delete'`` to exercise the full clean_model code path.
"""

import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr_colloc.model import (
    create_instance,
)
# Steady-state solve uses dae.finite_difference via the plain cstr model.
from pyomo.contrib.mpc.examples.cstr.model import (
    create_instance as create_ss_instance,
)


def get_steady_state_data(target, tee=False):
    """Solve steady-state CSTR to get initial or setpoint data."""
    m = create_ss_instance(dynamic=False)
    interface = mpc.DynamicModelInterface(m, m.time)
    var_set, tr_cost = interface.get_penalty_from_target(target)
    m.target_set = var_set
    m.tracking_cost = tr_cost
    m.objective = pyo.Objective(expr=sum(m.tracking_cost[:, 0]))
    m.flow_in[:].unfix()
    solver = pyo.SolverFactory("ipopt")
    solver.solve(m, tee=tee)
    return interface.get_data_at_time(0)


def run_cstr_mpc_colloc(
    initial_data,
    setpoint_data,
    samples_per_controller_horizon=5,
    sample_time=2.0,
    ntfe_per_sample_controller=2,
    ncp=3,
    ntfe_plant=5,
    simulation_steps=5,
    dae_clean_model='none',
    tee=False,
):
    """Run the CSTR MPC loop with a collocation-discretized controller model.

    Parameters
    ----------
    initial_data: ScalarData
    setpoint_data: ScalarData
    samples_per_controller_horizon: int
    sample_time: float
    ntfe_per_sample_controller: int
        Number of finite elements per sample interval in the controller model.
    ncp: int
        Collocation points per finite element.
    ntfe_plant: int
        Finite elements in the plant model (uses finite difference).
    simulation_steps: int
    dae_clean_model: str
        Passed to dae.collocation clean_model: 'none', 'deactivate', or 'delete'.
        When 'delete', DynamicModelInterface is created with clean_model=True so
        that all variable accesses in the MPC loop use the safe code paths.
    tee: bool

    Returns
    -------
    tuple: (m_controller, m_plant, sim_data)
        The controller model, plant model, and accumulated simulation data.
        Returning m_controller allows callers to inspect variable sizes after
        the loop (useful for verifying that clean_model='delete' did not
        silently re-create deleted entries).

    """
    controller_horizon = sample_time * samples_per_controller_horizon
    ntfe = ntfe_per_sample_controller * samples_per_controller_horizon

    # mpc_clean_model=True routes DynamicModelInterface to safe access paths.
    # This is only needed (and only correct) when dae_clean_model='delete'.
    mpc_clean_model = dae_clean_model == 'delete'

    m_controller = create_instance(
        horizon=controller_horizon,
        ntfe=ntfe,
        ncp=ncp,
        clean_model=dae_clean_model,
    )
    controller_interface = mpc.DynamicModelInterface(
        m_controller, m_controller.time, clean_model=mpc_clean_model
    )
    t0_controller = m_controller.time.first()

    # Plant uses finite difference (no clean_model concern).
    from pyomo.contrib.mpc.examples.cstr.model import (
        create_instance as create_fd_instance,
    )
    m_plant = create_fd_instance(horizon=sample_time, ntfe=ntfe_plant)
    plant_interface = mpc.DynamicModelInterface(m_plant, m_plant.time)

    # Load initial conditions and initialize
    controller_interface.load_data(initial_data)
    plant_interface.load_data(initial_data)

    #
    # Add objective to controller model
    #
    setpoint_variables = [m_controller.conc[:, "A"], m_controller.conc[:, "B"]]
    vset, tr_cost = controller_interface.get_penalty_from_target(
        setpoint_data, variables=setpoint_variables
    )
    m_controller.setpoint_set = vset
    m_controller.tracking_cost = tr_cost
    m_controller.objective = pyo.Objective(
        expr=sum(
            m_controller.tracking_cost[i, t]
            for i in m_controller.setpoint_set
            for t in m_controller.time
            if t != m_controller.time.first()
        )
    )

    #
    # Unfix input in controller model and add piecewise-constant constraints
    #
    m_controller.flow_in[:].unfix()
    m_controller.flow_in[t0_controller].fix()
    sample_points = [
        i * sample_time for i in range(samples_per_controller_horizon + 1)
    ]
    input_set, pwc_con = controller_interface.get_piecewise_constant_constraints(
        [m_controller.flow_in], sample_points
    )
    m_controller.input_set = input_set
    m_controller.pwc_con = pwc_con

    sim_t0 = 0.0

    #
    # Data structure to accumulate rolling-horizon simulation results
    #
    sim_data = plant_interface.get_data_at_time([sim_t0])

    solver = pyo.SolverFactory("ipopt")
    non_initial_plant_time = list(m_plant.time)[1:]
    ts = sample_time + t0_controller

    for i in range(simulation_steps):
        sim_t0 = i * sample_time

        #
        # Solve controller to get inputs
        #
        res = solver.solve(m_controller, tee=tee)
        pyo.assert_optimal_termination(res)
        ts_data = controller_interface.get_data_at_time(ts)
        input_data = ts_data.extract_variables([m_controller.flow_in])

        plant_interface.load_data(input_data)

        #
        # Simulate plant
        #
        res = solver.solve(m_plant, tee=tee)
        pyo.assert_optimal_termination(res)

        #
        # Accumulate plant data
        #
        m_data = plant_interface.get_data_at_time(non_initial_plant_time)
        m_data.shift_time_points(sim_t0 - m_plant.time.first())
        sim_data.concatenate(m_data)

        #
        # Re-initialize plant from end state
        #
        tf_data = plant_interface.get_data_at_time(m_plant.time.last())
        plant_interface.load_data(tf_data)

        #
        # Re-initialize controller: shift horizon and load new initial condition
        #
        controller_interface.shift_values_by_time(sample_time)
        controller_interface.load_data(tf_data, time_points=t0_controller)

    return m_controller, m_plant, sim_data


def main():
    init_steady_target = mpc.ScalarData({"flow_in[*]": 0.3})
    init_data = get_steady_state_data(init_steady_target, tee=False)
    setpoint_target = mpc.ScalarData({"flow_in[*]": 1.2})
    setpoint_data = get_steady_state_data(setpoint_target, tee=False)

    m_controller, m_plant, sim_data = run_cstr_mpc_colloc(
        init_data, setpoint_data, dae_clean_model='delete', tee=False
    )


if __name__ == "__main__":
    main()
