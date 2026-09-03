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
from pyomo.contrib.mpc.examples.cstr.model import create_instance
from pyomo.contrib.mpc.examples.cstr.run_mpc import get_steady_state_data, run_cstr_mpc

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
        _, m_plant, sim_data = run_cstr_mpc(
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
    """MPC with a controller model discretized by collocation

    With clean_model=True the controller model has no entries at
    non-collocation points. These tests check that the MPC loop neither
    re-creates those entries nor changes the answer.
    """

    _sample_time = 2.0
    _samples_per_horizon = 5
    _ntfe_per_sample = 2
    _ntfe_plant = 5
    _simulation_steps = 5
    _ncp = 3

    def _run_mpc(self, clean_model):
        return run_cstr_mpc(
            get_steady_state_data(mpc.ScalarData({"flow_in[*]": 0.3})),
            get_steady_state_data(mpc.ScalarData({"flow_in[*]": 1.2})),
            samples_per_controller_horizon=self._samples_per_horizon,
            sample_time=self._sample_time,
            ntfe_per_sample_controller=self._ntfe_per_sample,
            ntfe_plant=self._ntfe_plant,
            simulation_steps=self._simulation_steps,
            discretizer="dae.collocation",
            ncp=self._ncp,
            clean_model=clean_model,
        )

    def _make_controller(self, clean_model):
        return create_instance(
            horizon=self._sample_time * self._samples_per_horizon,
            ntfe=self._ntfe_per_sample * self._samples_per_horizon,
            discretizer="dae.collocation",
            ncp=self._ncp,
            clean_model=clean_model,
        )

    @staticmethod
    def _var_sizes(m):
        return {v.local_name: len(v) for v in m.component_objects(pyo.Var)}

    def test_model_size_consistent(self):
        # The variable entries deleted at construction must still be absent
        # after a full MPC loop. Every access in contrib.mpc goes through
        # var[t], which would silently re-create them.
        ref_sizes = self._var_sizes(self._make_controller(clean_model=True))
        full_sizes = self._var_sizes(self._make_controller(clean_model=False))
        self.assertTrue(
            any(ref_sizes[key] < full_sizes[key] for key in ref_sizes),
            "clean_model=True did not remove any variable entries",
        )

        m_controller, _, _ = self._run_mpc(clean_model=True)
        self.assertEqual(ref_sizes, self._var_sizes(m_controller))

    def test_simulation_matches_full_model(self):
        # Removing entries that participate in no equation must not change
        # the trajectory the controller produces.
        _, m_plant, sim_full = self._run_mpc(clean_model=False)
        _, _, sim_clean = self._run_mpc(clean_model=True)

        n_time_points = self._simulation_steps * self._ntfe_plant + 1
        self.assertEqual(len(sim_full.get_time_points()), n_time_points)
        self.assertEqual(len(sim_clean.get_time_points()), n_time_points)

        for var in (m_plant.conc[:, "A"], m_plant.conc[:, "B"], m_plant.flow_in[:]):
            cuid = sim_full.get_cuid(var)
            self.assertStructuredAlmostEqual(
                sim_full.get_data_from_key(cuid),
                sim_clean.get_data_from_key(cuid),
                delta=1e-4,
            )


if __name__ == "__main__":
    unittest.main()
