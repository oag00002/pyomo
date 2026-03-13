# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.run_mpc import get_steady_state_data, run_cstr_mpc
from pyomo.contrib.mpc.examples.cstr_colloc.run_mpc import (
    get_steady_state_data as get_steady_state_data_colloc,
    run_cstr_mpc as run_cstr_mpc_colloc,
)
from pyomo.contrib.mpc.examples.cstr_colloc.model import create_instance

ipopt_available = pyo.SolverFactory("ipopt").available()


@unittest.skipIf(not ipopt_available, "ipopt is not available")
class TestCSTRMPC(unittest.TestCase):
    # This data was obtained from a run of this code. The test is
    # intended to make sure that values do not change, not that
    # they are correct in some absolute sense.
    _pred_A_data = [
        1.15385,
        2.11629,
        2.59104,
        2.82521,
        2.94072,
        2.99770,
        2.84338,
        2.76022,
        2.71541,
        2.69127,
        2.67826,
        2.70659,
        2.72163,
        2.72961,
        2.73384,
        2.73609,
        2.73100,
        2.72830,
        2.72686,
        2.72609,
        2.72568,
        2.72660,
        2.72709,
        2.72735,
        2.72749,
        2.72756,
    ]
    _pred_B_data = [
        3.85615,
        2.89371,
        2.41896,
        2.18479,
        2.06928,
        2.01230,
        2.16662,
        2.24978,
        2.29459,
        2.31873,
        2.33174,
        2.30341,
        2.28837,
        2.28039,
        2.27616,
        2.27391,
        2.27900,
        2.28170,
        2.28314,
        2.28391,
        2.28432,
        2.28340,
        2.28291,
        2.28265,
        2.28251,
        2.28244,
    ]

    def _get_initial_data(self):
        initial_data = mpc.ScalarData({"flow_in[*]": 0.3})
        return get_steady_state_data(initial_data)

    def _get_setpoint_data(self):
        setpoint_data = mpc.ScalarData({"flow_in[*]": 1.2})
        return get_steady_state_data(setpoint_data)

    def test_mpc_simulation(self):
        initial_data = self._get_initial_data()
        setpoint_data = self._get_setpoint_data()
        sample_time = 2.0
        samples_per_horizon = 5
        ntfe_per_sample = 2
        ntfe_plant = 5
        simulation_steps = 5
        m_plant, sim_data = run_cstr_mpc(
            initial_data,
            setpoint_data,
            samples_per_controller_horizon=samples_per_horizon,
            sample_time=sample_time,
            ntfe_per_sample_controller=ntfe_per_sample,
            ntfe_plant=ntfe_plant,
            simulation_steps=simulation_steps,
        )
        sim_time_points = [
            sample_time / ntfe_plant * i
            for i in range(simulation_steps * ntfe_plant + 1)
        ]

        AB_data = sim_data.extract_variables(
            [m_plant.conc[:, "A"], m_plant.conc[:, "B"]]
        )

        A_cuid = sim_data.get_cuid(m_plant.conc[:, "A"])
        B_cuid = sim_data.get_cuid(m_plant.conc[:, "B"])
        pred_data = {A_cuid: self._pred_A_data, B_cuid: self._pred_B_data}

        self.assertStructuredAlmostEqual(pred_data, AB_data.get_data(), delta=1e-3)
        self.assertStructuredAlmostEqual(
            sim_time_points, AB_data.get_time_points(), delta=1e-7
        )


@unittest.skipIf(not ipopt_available, "ipopt is not available")
class TestCSTRCollocationMPC(unittest.TestCase):
    """Tests for the collocation-based CSTR MPC example."""

    # Small problem sizes for fast test execution
    _mpc_kwargs = dict(
        samples_per_controller_horizon=5,
        sample_time=2.0,
        ntfe_per_sample_controller=2,
        ntfe_plant=5,
        simulation_steps=3,
    )

    def _get_data(self):
        init_data = get_steady_state_data_colloc(
            mpc.ScalarData({"flow_in[*]": 0.3})
        )
        setpoint_data = get_steady_state_data_colloc(
            mpc.ScalarData({"flow_in[*]": 1.2})
        )
        return init_data, setpoint_data

    def _get_variable_sizes(self, m):
        return {
            "flow_in": len(m.flow_in),
            "conc_in": len(m.conc_in),
            "flow_out": len(m.flow_out),
            "conc": len(m.conc),
        }

    def test_model_size_consistent_delete(self):
        """Model variable sizes must not change during MPC when clean_model='delete'.

        Checks that no deleted VarData entries are silently re-created by
        DynamicModelInterface operations (load_data, shift_values_by_time,
        get_data_at_time, get_piecewise_constant_constraints, etc.).
        """
        ntfe = (
            self._mpc_kwargs["ntfe_per_sample_controller"]
            * self._mpc_kwargs["samples_per_controller_horizon"]
        )
        horizon = (
            self._mpc_kwargs["sample_time"]
            * self._mpc_kwargs["samples_per_controller_horizon"]
        )
        # Reference model: clean sizes immediately after discretization
        m_ref = create_instance(horizon=horizon, ntfe=ntfe, clean_model='delete')
        expected_sizes = self._get_variable_sizes(m_ref)

        init_data, setpoint_data = self._get_data()
        m_controller, _, _ = run_cstr_mpc_colloc(
            init_data, setpoint_data, clean_model='delete', **self._mpc_kwargs
        )
        end_sizes = self._get_variable_sizes(m_controller)

        self.assertEqual(end_sizes, expected_sizes)

    def test_mpc_runs_clean_model_none(self):
        """MPC with clean_model=None (no VarData deletion) runs without errors
        and model variable sizes remain stable throughout the loop.
        """
        ntfe = (
            self._mpc_kwargs["ntfe_per_sample_controller"]
            * self._mpc_kwargs["samples_per_controller_horizon"]
        )
        horizon = (
            self._mpc_kwargs["sample_time"]
            * self._mpc_kwargs["samples_per_controller_horizon"]
        )
        # Reference model: full ContinuousSet sizes (no cleanup applied)
        m_ref = create_instance(horizon=horizon, ntfe=ntfe, clean_model=None)
        expected_sizes = self._get_variable_sizes(m_ref)

        init_data, setpoint_data = self._get_data()
        m_controller, m_plant, sim_data = run_cstr_mpc_colloc(
            init_data, setpoint_data, clean_model=None, **self._mpc_kwargs
        )
        end_sizes = self._get_variable_sizes(m_controller)

        self.assertEqual(end_sizes, expected_sizes)
        self.assertGreater(len(sim_data.get_time_points()), 1)

    def test_mpc_runs_clean_model_deactivate(self):
        """MPC with clean_model='deactivate' (constraints deactivated, no VarData
        deletion) runs without errors and model variable sizes remain stable.
        """
        ntfe = (
            self._mpc_kwargs["ntfe_per_sample_controller"]
            * self._mpc_kwargs["samples_per_controller_horizon"]
        )
        horizon = (
            self._mpc_kwargs["sample_time"]
            * self._mpc_kwargs["samples_per_controller_horizon"]
        )
        m_ref = create_instance(horizon=horizon, ntfe=ntfe, clean_model='deactivate')
        expected_sizes = self._get_variable_sizes(m_ref)

        init_data, setpoint_data = self._get_data()
        m_controller, m_plant, sim_data = run_cstr_mpc_colloc(
            init_data, setpoint_data, clean_model='deactivate', **self._mpc_kwargs
        )
        end_sizes = self._get_variable_sizes(m_controller)

        self.assertEqual(end_sizes, expected_sizes)
        self.assertGreater(len(sim_data.get_time_points()), 1)


if __name__ == "__main__":
    unittest.main()
