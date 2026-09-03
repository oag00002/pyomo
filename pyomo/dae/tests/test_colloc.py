# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________


import pyomo.common.unittest as unittest

from pyomo.environ import (
    Block,
    ConcreteModel,
    Constraint,
    Objective,
    Set,
    SolverFactory,
    Suffix,
    TransformationFactory,
    Var,
    value,
    pyomo,
)
from pyomo.core.expr import identify_variables
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error

from pyomo.repn import generate_standard_repn

from io import StringIO

from pyomo.common.log import LoggingIntercept

from os.path import abspath, dirname, normpath, join

currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir, '..', '..', '..', 'examples', 'dae'))


def repn_to_rounded_dict(repn, digits):
    temp = dict()
    for i, v in enumerate(repn.linear_vars):
        temp[id(v)] = round(repn.linear_coefs[i], digits)
    return temp


class TestCollocation(unittest.TestCase):
    """
    Class for testing the pyomo.DAE collocation discretization
    """

    def setUp(self):
        """
        Setting up testing model
        """
        self.m = m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.v1 = Var(m.t)
        m.dv1 = DerivativeVar(m.v1)
        m.s = Set(initialize=[1, 2, 3], ordered=True)

    # test collocation discretization with radau points
    # on var indexed by single ContinuousSet
    def test_disc_single_index_radau(self):
        m = self.m.clone()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3)

        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertTrue(len(m.dv1_disc_eq) == 15)
        self.assertTrue(len(m.v1) == 16)

        expected_tau_points = [0.0, 0.1550510257216822, 0.64494897427831788, 1.0]
        expected_disc_points = [
            0,
            0.310102,
            1.289898,
            2.0,
            2.310102,
            3.289898,
            4.0,
            4.310102,
            5.289898,
            6.0,
            6.310102,
            7.289898,
            8.0,
            8.310102,
            9.289898,
            10,
        ]
        disc_info = m.t.get_discretization_info()

        self.assertTrue(disc_info['scheme'] == 'LAGRANGE-RADAU')

        for idx, val in enumerate(disc_info['tau_points']):
            self.assertAlmostEqual(val, expected_tau_points[idx])

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])
            self.assertTrue(type(val) in [float, int])

        self.assertTrue(hasattr(m, '_pyomo_dae_reclassified_derivativevars'))
        self.assertTrue(m._pyomo_dae_reclassified_derivativevars[0] is m.dv1)

        repn_baseline = {
            id(m.dv1[2.0]): 1.0,
            id(m.v1[0]): 1.5,
            id(m.v1[0.310102]): -2.76599,
            id(m.v1[1.289898]): 3.76599,
            id(m.v1[2.0]): -2.5,
        }

        repn = generate_standard_repn(m.dv1_disc_eq[2.0].body)
        repn_gen = repn_to_rounded_dict(repn, 5)
        self.assertEqual(repn_baseline, repn_gen)

        repn_baseline = {
            id(m.dv1[4.0]): 1.0,
            id(m.v1[2.0]): 1.5,
            id(m.v1[2.310102]): -2.76599,
            id(m.v1[3.289898]): 3.76599,
            id(m.v1[4.0]): -2.5,
        }

        repn = generate_standard_repn(m.dv1_disc_eq[4.0].body)
        repn_gen = repn_to_rounded_dict(repn, 5)
        self.assertEqual(repn_baseline, repn_gen)

    # test collocation discretization with radau points
    # second order derivative indexed by single ContinuousSet
    def test_disc_second_order_radau(self):
        m = self.m.clone()
        m.dv1dt2 = DerivativeVar(m.v1, wrt=(m.t, m.t))
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=2, ncp=2)

        self.assertTrue(hasattr(m, 'dv1dt2_disc_eq'))
        self.assertTrue(len(m.dv1dt2_disc_eq) == 4)
        self.assertTrue(len(m.v1) == 5)

        self.assertTrue(hasattr(m, '_pyomo_dae_reclassified_derivativevars'))
        self.assertTrue(m.dv1 in m._pyomo_dae_reclassified_derivativevars)
        self.assertTrue(m.dv1dt2 in m._pyomo_dae_reclassified_derivativevars)

        repn_baseline = {
            id(m.dv1dt2[5.0]): 1,
            id(m.v1[0]): -0.24,
            id(m.v1[1.666667]): 0.36,
            id(m.v1[5.0]): -0.12,
        }

        repn = generate_standard_repn(m.dv1dt2_disc_eq[5.0].body)
        repn_gen = repn_to_rounded_dict(repn, 5)
        self.assertEqual(repn_baseline, repn_gen)

        repn_baseline = {
            id(m.dv1dt2[10]): 1,
            id(m.v1[5.0]): -0.24,
            id(m.v1[6.666667]): 0.36,
            id(m.v1[10]): -0.12,
        }

        repn = generate_standard_repn(m.dv1dt2_disc_eq[10.0].body)
        repn_gen = repn_to_rounded_dict(repn, 5)
        self.assertEqual(repn_baseline, repn_gen)

    # test second order derivative with single collocation point
    def test_disc_second_order_1cp(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 1))
        m.t2 = ContinuousSet(bounds=(0, 10))
        m.v = Var(m.t, m.t2)
        m.dv = DerivativeVar(m.v, wrt=(m.t, m.t2))
        TransformationFactory('dae.collocation').apply_to(m, nfe=2, ncp=1)

        self.assertTrue(hasattr(m, 'dv_disc_eq'))
        self.assertTrue(len(m.dv_disc_eq) == 4)
        self.assertTrue(len(m.v) == 9)

    # test collocation discretization with legendre points
    # on var indexed by single ContinuousSet
    def test_disc_single_index_legendre(self):
        m = self.m.clone()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3, scheme='LAGRANGE-LEGENDRE')

        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertTrue(hasattr(m, 'v1_t_cont_eq'))
        self.assertTrue(len(m.dv1_disc_eq) == 15)
        self.assertTrue(len(m.v1_t_cont_eq) == 5)
        self.assertTrue(len(m.v1) == 21)

        expected_tau_points = [
            0.0,
            0.11270166537925834,
            0.49999999999999989,
            0.88729833462074226,
        ]
        expected_disc_points = [
            0,
            0.225403,
            1.0,
            1.774597,
            2.0,
            2.225403,
            3.0,
            3.774597,
            4.0,
            4.225403,
            5.0,
            5.774597,
            6.0,
            6.225403,
            7.0,
            7.774597,
            8.0,
            8.225403,
            9.0,
            9.774597,
            10,
        ]
        disc_info = m.t.get_discretization_info()

        self.assertTrue(disc_info['scheme'] == 'LAGRANGE-LEGENDRE')

        for idx, val in enumerate(disc_info['tau_points']):
            self.assertAlmostEqual(val, expected_tau_points[idx])

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])
            self.assertTrue(type(val) in [float, int])

        self.assertTrue(hasattr(m, '_pyomo_dae_reclassified_derivativevars'))
        self.assertTrue(m.dv1 in m._pyomo_dae_reclassified_derivativevars)

        repn_baseline = {
            id(m.dv1[3.0]): 1,
            id(m.v1[2.0]): -1.5,
            id(m.v1[2.225403]): 2.86374,
            id(m.v1[3.0]): -1.0,
            id(m.v1[3.774597]): -0.36374,
        }

        repn = generate_standard_repn(m.dv1_disc_eq[3.0].body)
        repn_gen = repn_to_rounded_dict(repn, 5)
        self.assertEqual(repn_baseline, repn_gen)

        repn_baseline = {
            id(m.dv1[5.0]): 1,
            id(m.v1[4.0]): -1.5,
            id(m.v1[4.225403]): 2.86374,
            id(m.v1[5.0]): -1.0,
            id(m.v1[5.774597]): -0.36374,
        }

        repn = generate_standard_repn(m.dv1_disc_eq[5.0].body)
        repn_gen = repn_to_rounded_dict(repn, 5)
        self.assertEqual(repn_baseline, repn_gen)

    # test collocation discretization with legendre points
    # second order derivative indexed by single ContinuousSet
    def test_disc_second_order_legendre(self):
        m = self.m.clone()
        m.dv1dt2 = DerivativeVar(m.v1, wrt=(m.t, m.t))
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=2, ncp=2, scheme='LAGRANGE-LEGENDRE')

        self.assertTrue(hasattr(m, 'dv1dt2_disc_eq'))
        self.assertTrue(hasattr(m, 'v1_t_cont_eq'))
        self.assertTrue(len(m.dv1dt2_disc_eq) == 4)
        self.assertTrue(len(m.v1_t_cont_eq) == 2)
        self.assertTrue(len(m.v1) == 7)

        self.assertTrue(hasattr(m, '_pyomo_dae_reclassified_derivativevars'))
        self.assertTrue(m.dv1 in m._pyomo_dae_reclassified_derivativevars)
        self.assertTrue(m.dv1dt2 in m._pyomo_dae_reclassified_derivativevars)

        repn_baseline = {
            id(m.dv1dt2[1.056624]): 1,
            id(m.v1[0]): -0.48,
            id(m.v1[1.056624]): 0.65569,
            id(m.v1[3.943376]): -0.17569,
        }

        repn = generate_standard_repn(m.dv1dt2_disc_eq[1.056624].body)
        repn_gen = repn_to_rounded_dict(repn, 5)
        self.assertEqual(repn_baseline, repn_gen)

        repn_baseline = {
            id(m.dv1dt2[6.056624]): 1,
            id(m.v1[5.0]): -0.48,
            id(m.v1[6.056624]): 0.65569,
            id(m.v1[8.943376]): -0.17569,
        }

        repn = generate_standard_repn(m.dv1dt2_disc_eq[6.056624].body)
        repn_gen = repn_to_rounded_dict(repn, 5)
        self.assertEqual(repn_baseline, repn_gen)

    # test collocation discretization on var indexed by ContinuousSet and Set
    def test_disc_multi_index(self):
        m = self.m.clone()
        m.v2 = Var(m.t, m.s)
        m.dv2 = DerivativeVar(m.v2)

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3)

        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertTrue(hasattr(m, 'dv2_disc_eq'))
        self.assertTrue(len(m.dv2_disc_eq) == 45)
        self.assertTrue(len(m.v2) == 48)

        expected_tau_points = [0.0, 0.1550510257216822, 0.64494897427831788, 1.0]
        expected_disc_points = [
            0,
            0.310102,
            1.289898,
            2.0,
            2.310102,
            3.289898,
            4.0,
            4.310102,
            5.289898,
            6.0,
            6.310102,
            7.289898,
            8.0,
            8.310102,
            9.289898,
            10,
        ]
        disc_info = m.t.get_discretization_info()

        self.assertTrue(disc_info['scheme'] == 'LAGRANGE-RADAU')

        for idx, val in enumerate(disc_info['tau_points']):
            self.assertAlmostEqual(val, expected_tau_points[idx])

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])

        self.assertTrue(hasattr(m, '_pyomo_dae_reclassified_derivativevars'))
        self.assertTrue(m.dv1 in m._pyomo_dae_reclassified_derivativevars)
        self.assertTrue(m.dv2 in m._pyomo_dae_reclassified_derivativevars)

    # test collocation discretization on var indexed by multiple ContinuousSets
    def test_disc_multi_index2(self):
        m = self.m.clone()
        m.t2 = ContinuousSet(bounds=(0, 5))
        m.v2 = Var(m.t, m.t2)
        m.dv2dt = DerivativeVar(m.v2, wrt=m.t)
        m.dv2dt2 = DerivativeVar(m.v2, wrt=m.t2)

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=2, ncp=2)

        self.assertTrue(hasattr(m, 'dv2dt_disc_eq'))
        self.assertTrue(hasattr(m, 'dv2dt2_disc_eq'))
        self.assertTrue(len(m.dv2dt_disc_eq) == 20)
        self.assertTrue(len(m.dv2dt2_disc_eq) == 20)
        self.assertTrue(len(m.v2) == 25)

        expected_t_disc_points = [0, 1.666667, 5.0, 6.666667, 10]
        expected_t2_disc_points = [0, 0.833333, 2.5, 3.333333, 5]

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_t_disc_points[idx])

        for idx, val in enumerate(list(m.t2)):
            self.assertAlmostEqual(val, expected_t2_disc_points[idx])

        self.assertTrue(hasattr(m, '_pyomo_dae_reclassified_derivativevars'))
        self.assertTrue(m.dv1 in m._pyomo_dae_reclassified_derivativevars)
        self.assertTrue(m.dv2dt in m._pyomo_dae_reclassified_derivativevars)
        self.assertTrue(m.dv2dt2 in m._pyomo_dae_reclassified_derivativevars)

    # test collocation discretization on var indexed by ContinuousSet
    # and multi-dimensional Set
    def test_disc_multidimen_index(self):
        m = self.m.clone()
        m.s2 = Set(initialize=[('A', 'B'), ('C', 'D'), ('E', 'F')])
        m.v2 = Var(m.t, m.s2)
        m.dv2 = DerivativeVar(m.v2)
        m.v3 = Var(m.s2, m.t)
        m.dv3 = DerivativeVar(m.v3)

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3)

        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertTrue(hasattr(m, 'dv2_disc_eq'))
        self.assertTrue(hasattr(m, 'dv3_disc_eq'))
        self.assertTrue(len(m.dv2_disc_eq) == 45)
        self.assertTrue(len(m.v2) == 48)
        self.assertTrue(len(m.dv3_disc_eq) == 45)
        self.assertTrue(len(m.v3) == 48)

        expected_tau_points = [0.0, 0.1550510257216822, 0.64494897427831788, 1.0]
        expected_disc_points = [
            0,
            0.310102,
            1.289898,
            2.0,
            2.310102,
            3.289898,
            4.0,
            4.310102,
            5.289898,
            6.0,
            6.310102,
            7.289898,
            8.0,
            8.310102,
            9.289898,
            10,
        ]
        disc_info = m.t.get_discretization_info()

        self.assertTrue(disc_info['scheme'] == 'LAGRANGE-RADAU')

        for idx, val in enumerate(disc_info['tau_points']):
            self.assertAlmostEqual(val, expected_tau_points[idx])

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])

        self.assertTrue(hasattr(m, '_pyomo_dae_reclassified_derivativevars'))
        self.assertTrue(m.dv1 in m._pyomo_dae_reclassified_derivativevars)
        self.assertTrue(m.dv2 in m._pyomo_dae_reclassified_derivativevars)
        self.assertTrue(m.dv3 in m._pyomo_dae_reclassified_derivativevars)

    # test passing the discretization invalid options
    def test_disc_invalid_options(self):
        m = self.m.clone()

        with self.assertRaises(TypeError):
            TransformationFactory('dae.collocation').apply_to(m, wrt=m.s)

        with self.assertRaises(ValueError):
            TransformationFactory('dae.collocation').apply_to(m, nfe=-1)

        with self.assertRaises(ValueError):
            TransformationFactory('dae.collocation').apply_to(m, ncp=0)

        with self.assertRaises(ValueError):
            TransformationFactory('dae.collocation').apply_to(m, scheme='foo')

        with self.assertRaises(ValueError):
            TransformationFactory('dae.collocation').apply_to(m, foo=True)

        TransformationFactory('dae.collocation').apply_to(m, wrt=m.t)
        with self.assertRaises(ValueError):
            TransformationFactory('dae.collocation').apply_to(m, wrt=m.t)

        m = self.m.clone()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m)
        with self.assertRaises(ValueError):
            disc.apply_to(m)

    # test looking up radau collocation points
    def test_lookup_radau_collocation_points(self):
        # Save initial flag value
        colloc_numpy_avail = pyomo.dae.plugins.colloc.numpy_available

        # Numpy flag must be False to test lookup
        pyomo.dae.plugins.colloc.numpy_available = False

        m = self.m.clone()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3)

        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertTrue(len(m.dv1_disc_eq) == 15)
        self.assertTrue(len(m.v1) == 16)

        expected_tau_points = [0.0, 0.1550510257216822, 0.64494897427831788, 1.0]
        expected_disc_points = [
            0,
            0.310102,
            1.289898,
            2.0,
            2.310102,
            3.289898,
            4.0,
            4.310102,
            5.289898,
            6.0,
            6.310102,
            7.289898,
            8.0,
            8.310102,
            9.289898,
            10,
        ]
        disc_info = m.t.get_discretization_info()

        self.assertTrue(disc_info['scheme'] == 'LAGRANGE-RADAU')

        for idx, val in enumerate(disc_info['tau_points']):
            self.assertAlmostEqual(val, expected_tau_points[idx])

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])

        m = self.m.clone()
        with self.assertRaises(ValueError):
            disc = TransformationFactory('dae.collocation')
            disc.apply_to(m, ncp=15, scheme='LAGRANGE-RADAU')

        # Restore initial flag value
        pyomo.dae.plugins.colloc.numpy_available = colloc_numpy_avail

    # test looking up legendre collocation points
    def test_lookup_legendre_collocation_points(self):
        # Save initial flag value
        colloc_numpy_avail = pyomo.dae.plugins.colloc.numpy_available

        # Numpy flag must be False to test lookup
        pyomo.dae.plugins.colloc.numpy_available = False

        m = self.m.clone()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3, scheme='LAGRANGE-LEGENDRE')

        self.assertTrue(hasattr(m, 'dv1_disc_eq'))
        self.assertTrue(len(m.dv1_disc_eq) == 15)
        self.assertTrue(len(m.v1) == 21)

        expected_tau_points = [
            0.0,
            0.11270166537925834,
            0.49999999999999989,
            0.88729833462074226,
        ]
        expected_disc_points = [
            0,
            0.225403,
            1.0,
            1.774597,
            2.0,
            2.225403,
            3.0,
            3.774597,
            4.0,
            4.225403,
            5.0,
            5.774597,
            6.0,
            6.225403,
            7.0,
            7.774597,
            8.0,
            8.225403,
            9.0,
            9.774597,
            10,
        ]

        disc_info = m.t.get_discretization_info()

        self.assertTrue(disc_info['scheme'] == 'LAGRANGE-LEGENDRE')

        for idx, val in enumerate(disc_info['tau_points']):
            self.assertAlmostEqual(val, expected_tau_points[idx])

        for idx, val in enumerate(list(m.t)):
            self.assertAlmostEqual(val, expected_disc_points[idx])

        m = self.m.clone()
        with self.assertRaises(ValueError):
            disc = TransformationFactory('dae.collocation')
            disc.apply_to(m, ncp=15, scheme='LAGRANGE-LEGENDRE')

        # Restore initial flag value
        pyomo.dae.plugins.colloc.numpy_available = colloc_numpy_avail

    # test discretization using fewer points than ContinuousSet initialized
    # with
    def test_initialized_continuous_set(self):
        m = ConcreteModel()
        m.t = ContinuousSet(initialize=[0, 1, 2, 3, 4])
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v)

        log_out = StringIO()
        with LoggingIntercept(log_out, 'pyomo.dae'):
            TransformationFactory('dae.collocation').apply_to(m, nfe=2)
        self.assertIn('More finite elements', log_out.getvalue())

    # Test discretizing an invalid derivative
    def test_invalid_derivative(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v, wrt=(m.t, m.t, m.t))

        with self.assertRaises(DAE_Error):
            TransformationFactory('dae.collocation').apply_to(m)

    # test reduce_collocation_points invalid options
    def test_reduce_colloc_invalid(self):
        m = self.m.clone()
        m.u = Var(m.t)
        m2 = m.clone()

        disc = TransformationFactory('dae.collocation')
        disc2 = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3)

        # No ContinuousSet specified
        with self.assertRaises(TypeError):
            disc.reduce_collocation_points(m, contset=None)

        # Component passed in is not a ContinuousSet
        with self.assertRaises(TypeError):
            disc.reduce_collocation_points(m, contset=m.s)

        # Call reduce_collocation_points method before applying discretization
        with self.assertRaises(RuntimeError):
            disc2.reduce_collocation_points(m2, contset=m2.t)

        # Call reduce_collocation_points on a ContinuousSet that hasn't been
        #  discretized
        m2.tt = ContinuousSet(bounds=(0, 1))
        disc2.apply_to(m2, wrt=m2.t)
        with self.assertRaises(ValueError):
            disc2.reduce_collocation_points(m2, contset=m2.tt)

        # No Var specified
        with self.assertRaises(TypeError):
            disc.reduce_collocation_points(m, contset=m.t, var=None)

        # Component passed in is not a Var
        with self.assertRaises(TypeError):
            disc.reduce_collocation_points(m, contset=m.t, var=m.s)

        # New ncp not specified
        with self.assertRaises(TypeError):
            disc.reduce_collocation_points(m, contset=m.t, var=m.v1, ncp=None)

        # Negative ncp specified
        with self.assertRaises(ValueError):
            disc.reduce_collocation_points(m, contset=m.t, var=m.v1, ncp=-3)

        # Too large ncp specified
        with self.assertRaises(ValueError):
            disc.reduce_collocation_points(m, contset=m.t, var=m.v1, ncp=10)

        # Passing Vars not indexed by the ContinuousSet
        m.v2 = Var()
        m.v3 = Var(m.s)
        m.v4 = Var(m.s, m.s)

        with self.assertRaises(IndexError):
            disc.reduce_collocation_points(m, contset=m.t, var=m.v2, ncp=1)

        with self.assertRaises(IndexError):
            disc.reduce_collocation_points(m, contset=m.t, var=m.v3, ncp=1)

        with self.assertRaises(IndexError):
            disc.reduce_collocation_points(m, contset=m.t, var=m.v4, ncp=1)

        # Calling reduce_collocation_points more than once
        disc.reduce_collocation_points(m, contset=m.t, var=m.u, ncp=1)
        with self.assertRaises(RuntimeError):
            disc.reduce_collocation_points(m, contset=m.t, var=m.u, ncp=1)

    # test trying to discretize a ContinuousSet twice
    def test_discretize_twice(self):
        m = self.m.clone()

        disc1 = TransformationFactory('dae.collocation')
        disc1.apply_to(m, nfe=5, ncp=3)

        disc2 = TransformationFactory('dae.collocation')

        with self.assertRaises(DAE_Error):
            disc2.apply_to(m, nfe=5, ncp=3)

    # test reduce_collocation_points on var indexed by single ContinuousSet
    def test_reduce_colloc_single_index(self):
        m = self.m.clone()
        m.u = Var(m.t)

        m2 = m.clone()
        m3 = m.clone()

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3)
        disc.reduce_collocation_points(m, contset=m.t, var=m.u, ncp=1)

        self.assertTrue(hasattr(m, 'u_interpolation_constraints'))
        self.assertEqual(len(m.u_interpolation_constraints), 10)

        disc2 = TransformationFactory('dae.collocation')
        disc2.apply_to(m2, wrt=m2.t, nfe=5, ncp=3)
        disc2.reduce_collocation_points(m2, contset=m2.t, var=m2.u, ncp=3)

        self.assertFalse(hasattr(m2, 'u_interpolation_constraints'))

        disc3 = TransformationFactory('dae.collocation')
        disc3.apply_to(m3, wrt=m3.t, nfe=5, ncp=3)
        disc3.reduce_collocation_points(m3, contset=m3.t, var=m3.u, ncp=2)

        self.assertTrue(hasattr(m3, 'u_interpolation_constraints'))
        self.assertEqual(len(m3.u_interpolation_constraints), 5)

    # test reduce_collocation_points on var indexed by ContinuousSet and Set
    def test_reduce_colloc_multi_index(self):
        m = self.m.clone()
        m.u = Var(m.t, m.s)

        m2 = m.clone()
        m3 = m.clone()

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=5, ncp=3)
        disc.reduce_collocation_points(m, contset=m.t, var=m.u, ncp=1)

        self.assertTrue(hasattr(m, 'u_interpolation_constraints'))
        self.assertEqual(len(m.u_interpolation_constraints), 30)

        disc2 = TransformationFactory('dae.collocation')
        disc2.apply_to(m2, wrt=m2.t, nfe=5, ncp=3)
        disc2.reduce_collocation_points(m2, contset=m2.t, var=m2.u, ncp=3)

        self.assertFalse(hasattr(m2, 'u_interpolation_constraints'))

        disc3 = TransformationFactory('dae.collocation')
        disc3.apply_to(m3, wrt=m3.t, nfe=5, ncp=3)
        disc3.reduce_collocation_points(m3, contset=m3.t, var=m3.u, ncp=2)

        self.assertTrue(hasattr(m3, 'u_interpolation_constraints'))
        self.assertEqual(len(m3.u_interpolation_constraints), 15)


class TestCleanModel(unittest.TestCase):
    """
    Class for testing the clean_model option of the collocation
    discretization
    """

    def _make_model(self):
        # v1 is a state, dv1 its derivative, v2 an algebraic variable
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.v1 = Var(m.t)
        m.dv1 = DerivativeVar(m.v1)
        m.v2 = Var(m.t)
        m.ode = Constraint(m.t, rule=lambda m, t: m.dv1[t] == -m.v1[t])
        m.alg = Constraint(m.t, rule=lambda m, t: m.v2[t] == m.v1[t] ** 2)
        return m

    def test_default_is_no_cleanup(self):
        m = self._make_model()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=2, ncp=2)
        for t in m.t:
            self.assertIn(t, m.ode)
            self.assertIn(t, m.alg)
            self.assertIn(t, m.dv1)
            self.assertIn(t, m.v2)

    def test_clean_model(self):
        # For LAGRANGE-RADAU the only non-collocation point is the initial
        # point. For LAGRANGE-LEGENDRE every finite element boundary is one.
        for scheme, n_disc_eq, n_cont_eq in [
            ('LAGRANGE-RADAU', 4, 0),
            ('LAGRANGE-LEGENDRE', 4, 2),
        ]:
            with self.subTest(scheme=scheme):
                m = self._make_model()
                disc = TransformationFactory('dae.collocation')
                disc.apply_to(m, nfe=2, ncp=2, scheme=scheme, clean_model=True)

                if scheme == 'LAGRANGE-RADAU':
                    non_colloc = {m.t.first()}
                else:
                    non_colloc = set(m.t.get_finite_elements())
                self.assertEqual(set(m.t) - set(m.dv1), non_colloc)

                for t in non_colloc:
                    self.assertNotIn(t, m.ode)
                    self.assertNotIn(t, m.alg)
                    self.assertNotIn(t, m.dv1)
                    self.assertNotIn(t, m.v2)
                    # The state variable is preserved everywhere: the initial
                    # conditions and the continuity equations need it.
                    self.assertIn(t, m.v1)
                for t in set(m.t) - non_colloc:
                    self.assertIn(t, m.ode)
                    self.assertIn(t, m.alg)
                    self.assertIn(t, m.dv1)
                    self.assertIn(t, m.v2)

                # Discretization and continuity equations are untouched
                self.assertEqual(len(m.dv1_disc_eq), n_disc_eq)
                if n_cont_eq:
                    self.assertEqual(len(m.v1_t_cont_eq), n_cont_eq)

    def test_clean_model_multi_index(self):
        # The ContinuousSet is not the first index of u
        m = self._make_model()
        m.s = Set(initialize=[1, 2, 3])
        m.u = Var(m.s, m.t)
        m.alg2 = Constraint(m.s, m.t, rule=lambda m, s, t: m.u[s, t] == m.v1[t])

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=1, ncp=2, clean_model=True)

        t0 = m.t.first()
        for s in m.s:
            self.assertNotIn((s, t0), m.u)
            self.assertNotIn((s, t0), m.alg2)
        self.assertIn(t0, m.v1)

    def test_clean_model_preserves_scalar_constraint(self):
        m = self._make_model()
        m.ic = Constraint(expr=m.v1[0] == 1.0)
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=2, ncp=2, clean_model=True)
        self.assertTrue(m.ic.active)
        self.assertIn(m.t.first(), m.v1)

    def test_clean_model_multiple_contsets_error(self):
        m = self._make_model()
        m.z = ContinuousSet(bounds=(0, 1))
        m.v3 = Var(m.z)
        m.dv3 = DerivativeVar(m.v3, wrt=m.z)
        m.ode3 = Constraint(m.z, rule=lambda m, z: m.dv3[z] == -m.v3[z])

        disc = TransformationFactory('dae.collocation')
        with self.assertRaisesRegex(DAE_Error, 'more than one ContinuousSet'):
            disc.apply_to(m, wrt=m.t, nfe=2, ncp=2, clean_model=True)

    def test_clean_model_indexed_block_error(self):
        m = self._make_model()
        m.b = Block(m.t)
        disc = TransformationFactory('dae.collocation')
        with self.assertRaisesRegex(DAE_Error, 'Block indexed by a ContinuousSet'):
            disc.apply_to(m, nfe=2, ncp=2, clean_model=True)


class TestStructuralReduceCollocationPoints(unittest.TestCase):
    """
    Class for testing the structural option of reduce_collocation_points
    """

    def _make_model(self, nfe=4, ncp=3):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.v = Var(m.t, initialize=1.0)
        m.dv = DerivativeVar(m.v, wrt=m.t)
        m.u = Var(m.t, bounds=(-5, 5), initialize=0.0)
        m.ode = Constraint(m.t, rule=lambda m, t: m.dv[t] == -m.v[t] ** 2 + m.u[t])
        m.obj = Objective(
            expr=sum((m.v[t] - 2.0) ** 2 for t in m.t)
            + 0.1 * sum(m.u[t] ** 2 for t in m.t)
        )
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=nfe, ncp=ncp)
        m.v[0].fix(1.0)
        return m, disc

    def test_structural_writes_no_interpolation_constraints(self):
        nfe = 4
        m, disc = self._make_model(nfe=nfe, ncp=3)
        n_before = len(m.u)
        disc.reduce_collocation_points(m, var=m.u, ncp=1, contset=m.t, structural=True)
        # No interpolation constraints are generated at all
        self.assertEqual(len(m.u_interpolation_constraints), 0)
        # One entry per finite element, plus the initial point, rather than
        # one entry per collocation point
        self.assertEqual(len(m.u), nfe + 1)
        self.assertLess(len(m.u), n_before)
        # Every remaining reference to u resolves to a surviving entry
        live = set(id(m.u[i]) for i in m.u)
        for con in m.component_data_objects(Constraint, active=True):
            for v in identify_variables(con.body):
                if v.parent_component() is m.u:
                    self.assertIn(id(v), live)

    def test_structural_false_is_unchanged(self):
        # The default and an explicit structural=False must agree
        m1, disc1 = self._make_model()
        disc1.reduce_collocation_points(m1, var=m1.u, ncp=1, contset=m1.t)
        m2, disc2 = self._make_model()
        disc2.reduce_collocation_points(
            m2, var=m2.u, ncp=1, contset=m2.t, structural=False
        )
        self.assertEqual(len(m1.u), len(m2.u))
        self.assertEqual(
            len(m1.u_interpolation_constraints), len(m2.u_interpolation_constraints)
        )
        for i in m1.u_interpolation_constraints:
            self.assertEqual(
                str(m1.u_interpolation_constraints[i].expr),
                str(m2.u_interpolation_constraints[i].expr),
            )

    def test_structural_requires_ncp_1(self):
        m, disc = self._make_model()
        with self.assertRaisesRegex(NotImplementedError, 'only.*implemented for ncp=1'):
            disc.reduce_collocation_points(
                m, var=m.u, ncp=2, contset=m.t, structural=True
            )

    @unittest.skipIf(
        not SolverFactory('ipopt').available(False), "ipopt is not available"
    )
    def test_structural_matches_algebraic(self):
        # The structural reduction must be a pure symbol swap: same solution,
        # fewer rows and columns, and duals still available.
        #
        # Note that duals require the writer's linear presolve to be off. That
        # is not specific to the interpolation constraints: the discretization
        # equations of any collocation model define the derivative variables
        # linearly, so a presolve that eliminates variables will always
        # eliminate those. A consumer that needs duals therefore cannot
        # presolve the redundant interpolation rows away either, which is what
        # makes not writing them worthwhile.
        results = {}
        for structural in (False, True):
            m, disc = self._make_model()
            disc.reduce_collocation_points(
                m, var=m.u, ncp=1, contset=m.t, structural=structural
            )
            m.dual = Suffix(direction=Suffix.IMPORT)
            solver = SolverFactory('ipopt')
            solver.solve(m)
            results[structural] = dict(
                obj=value(m.obj),
                v=[m.v[t].value for t in m.t],
                n_var=len(list(m.component_data_objects(Var, active=True))),
                n_con=len(list(m.component_data_objects(Constraint, active=True))),
                n_dual=len(m.dual),
            )

        alg, struct = results[False], results[True]
        # The two formulations are different NLPs with the same solution, so
        # they agree to solver tolerance rather than exactly.
        self.assertAlmostEqual(alg['obj'], struct['obj'], delta=1e-6)
        for v_alg, v_struct in zip(alg['v'], struct['v']):
            self.assertAlmostEqual(v_alg, v_struct, delta=1e-4)
        # Fewer variables and fewer constraints, and duals are still returned
        self.assertLess(struct['n_var'], alg['n_var'])
        self.assertLess(struct['n_con'], alg['n_con'])
        self.assertEqual(struct['n_dual'], struct['n_con'])


if __name__ == '__main__':
    unittest.main()
