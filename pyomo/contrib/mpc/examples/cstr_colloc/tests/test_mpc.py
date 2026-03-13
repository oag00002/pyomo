# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tests for the collocation CSTR MPC example with the clean_model flag.

Three test groups:

1. Baseline tests (clean_model='none' and 'deactivate') — verify that the MPC
   loop runs to completion with no regressions in the standard (non-delete) cases.

2. Size-consistency test (clean_model='delete') — after discretization, record
   the size of each model variable.  Run the full MPC loop.  Assert that variable
   sizes are unchanged, confirming that no deleted VarData entries were re-created
   by any MPC operation.

3. Numerical-equivalence test — verify that 'delete' and 'none' produce
   numerically identical plant simulation trajectories (to within solver tolerance),
   confirming that the safe code paths preserve mathematical correctness.
"""

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr_colloc.model import create_instance
from pyomo.contrib.mpc.examples.cstr_colloc.run_mpc import (
    get_steady_state_data,
    run_cstr_mpc_colloc,
)

ipopt_available = pyo.SolverFactory("ipopt").available()


def _var_sizes(m):
    """Return a dict of variable-name -> number of populated VarData entries."""
    return {
        v.local_name: sum(1 for _ in v._data)
        for v in m.component_objects(pyo.Var, active=True)
    }


@unittest.skipIf(not ipopt_available, "ipopt is not available")
class TestCSTRCollocationMPC(unittest.TestCase):
    """Tests for the CSTR collocation MPC example."""

    # Common MPC parameters used across tests
    _samples_per_horizon = 5
    _sample_time = 2.0
    _ntfe_per_sample = 2
    _ncp = 3
    _ntfe_plant = 5
    _simulation_steps = 5

    def _get_initial_data(self):
        return get_steady_state_data(mpc.ScalarData({"flow_in[*]": 0.3}))

    def _get_setpoint_data(self):
        return get_steady_state_data(mpc.ScalarData({"flow_in[*]": 1.2}))

    def _run_mpc(self, dae_clean_model='none'):
        """Helper: run full MPC loop and return (m_controller, m_plant, sim_data)."""
        return run_cstr_mpc_colloc(
            self._get_initial_data(),
            self._get_setpoint_data(),
            samples_per_controller_horizon=self._samples_per_horizon,
            sample_time=self._sample_time,
            ntfe_per_sample_controller=self._ntfe_per_sample,
            ncp=self._ncp,
            ntfe_plant=self._ntfe_plant,
            simulation_steps=self._simulation_steps,
            dae_clean_model=dae_clean_model,
        )

    # ------------------------------------------------------------------
    # Baseline: default (no cleanup)
    # ------------------------------------------------------------------

    def test_mpc_runs_clean_model_none(self):
        """MPC loop runs to completion with dae_clean_model='none'."""
        m_controller, m_plant, sim_data = self._run_mpc('none')
        expected_n_time_points = self._simulation_steps * self._ntfe_plant + 1
        self.assertEqual(len(sim_data.get_time_points()), expected_n_time_points)

    # ------------------------------------------------------------------
    # Baseline: constraint deactivation (no deletion)
    # ------------------------------------------------------------------

    def test_mpc_runs_clean_model_deactivate(self):
        """MPC loop runs to completion with dae_clean_model='deactivate'."""
        m_controller, m_plant, sim_data = self._run_mpc('deactivate')
        expected_n_time_points = self._simulation_steps * self._ntfe_plant + 1
        self.assertEqual(len(sim_data.get_time_points()), expected_n_time_points)

    # ------------------------------------------------------------------
    # Core: delete mode — variable sizes must not change during MPC loop
    # ------------------------------------------------------------------

    def test_model_size_consistent_delete(self):
        """With clean_model='delete', no deleted VarData entries are re-created.

        Strategy:
          1. Build a fresh discretized controller model and record variable sizes.
          2. Run the full MPC loop with dae_clean_model='delete'.
          3. Assert that the controller model's variable sizes at the end of the
             loop are identical to those recorded at step 1.
        """
        nfe = self._ntfe_per_sample * self._samples_per_horizon
        ncp = self._ncp
        # Reference sizes: taken from a freshly discretized model (before any MPC).
        m_ref = create_instance(
            horizon=self._sample_time * self._samples_per_horizon,
            ntfe=nfe,
            ncp=ncp,
            clean_model='delete',
        )
        ref_sizes = _var_sizes(m_ref)

        # Verify that 'delete' actually removed some entries relative to 'none'.
        m_none = create_instance(
            horizon=self._sample_time * self._samples_per_horizon,
            ntfe=nfe,
            ncp=ncp,
            clean_model='none',
        )
        none_sizes = _var_sizes(m_none)
        # At least one variable (dcdt) must have fewer entries after deletion.
        self.assertTrue(
            any(ref_sizes[k] < none_sizes[k] for k in ref_sizes),
            "Expected clean_model='delete' to reduce at least one variable's size",
        )

        # Run MPC loop with delete mode.
        m_controller, m_plant, sim_data = self._run_mpc('delete')
        post_loop_sizes = _var_sizes(m_controller)

        # Sizes must be unchanged: no deleted entry was re-created.
        self.assertEqual(
            ref_sizes,
            post_loop_sizes,
            "Variable sizes changed during MPC loop — deleted entries were re-created",
        )

        # MPC ran to completion.
        expected_n_time_points = self._simulation_steps * self._ntfe_plant + 1
        self.assertEqual(len(sim_data.get_time_points()), expected_n_time_points)

    # ------------------------------------------------------------------
    # Numerical equivalence: 'delete' vs 'none' must give same trajectory
    # ------------------------------------------------------------------

    def test_results_match_between_delete_and_none(self):
        """clean_model='delete' and 'none' produce numerically identical trajectories.

        This verifies that the safe code paths in DynamicModelInterface preserve
        the mathematical correctness of the MPC loop.
        """
        m_controller_none, m_plant, sim_none = self._run_mpc('none')
        m_controller_del, _,    sim_delete  = self._run_mpc('delete')

        A_cuid = sim_none.get_cuid(m_plant.conc[:, 'A'])
        B_cuid = sim_none.get_cuid(m_plant.conc[:, 'B'])

        A_none   = sim_none.get_data_from_key(A_cuid)
        A_delete = sim_delete.get_data_from_key(A_cuid)
        B_none   = sim_none.get_data_from_key(B_cuid)
        B_delete = sim_delete.get_data_from_key(B_cuid)

        self.assertStructuredAlmostEqual(A_none, A_delete, delta=1e-4)
        self.assertStructuredAlmostEqual(B_none, B_delete, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
