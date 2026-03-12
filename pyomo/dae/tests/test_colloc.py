# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________


import pyomo.common.unittest as unittest

from pyomo.environ import Var, Set, Constraint, ConcreteModel, TransformationFactory, pyomo
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
    """Tests for the clean_model flag on the collocation transformation."""

    def _make_ode_model(self):
        """Simple ODE model: dv1/dt = -v1, with algebraic variable v2."""
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.v1 = Var(m.t)
        m.dv1 = DerivativeVar(m.v1)
        m.v2 = Var(m.t)  # algebraic variable
        m.ode = Constraint(m.t, rule=lambda m, t: m.dv1[t] == -m.v1[t])
        m.alg = Constraint(m.t, rule=lambda m, t: m.v2[t] == m.v1[t] ** 2)
        return m

    def test_default_no_cleanup(self):
        """clean_model='none' (default) leaves the model untouched."""
        m = self._make_ode_model()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=2, ncp=2)
        t0 = m.t.first()
        # At t0, ode and alg should still be present and active
        self.assertIn(t0, m.ode)
        self.assertTrue(m.ode[t0].active)
        self.assertIn(t0, m.alg)
        self.assertTrue(m.alg[t0].active)
        # disc_eq should be untouched
        self.assertEqual(len(m.dv1_disc_eq), 4)

    def test_deactivate_radau(self):
        """clean_model='deactivate' deactivates constraints at t0 for RADAU."""
        m = self._make_ode_model()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=2, ncp=2, clean_model='deactivate')
        t0 = m.t.first()
        # User constraints at t0 should be inactive
        self.assertFalse(m.ode[t0].active)
        self.assertFalse(m.alg[t0].active)
        # Collocation point constraints should still be active
        fe = m.t.get_finite_elements()
        # For RADAU, right FE boundary (fe[1]) is a collocation point
        self.assertTrue(m.ode[fe[1]].active)
        self.assertTrue(m.alg[fe[1]].active)
        # disc_eq must be completely untouched
        self.assertEqual(len(m.dv1_disc_eq), 4)
        # State variable entry at t0 must still exist
        self.assertIn(t0, m.v1)

    def test_delete_radau(self):
        """clean_model='delete' removes constraint and variable entries at t0 for RADAU."""
        m = self._make_ode_model()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=2, ncp=2, clean_model='delete')
        t0 = m.t.first()
        # User constraints at t0 should be gone
        self.assertNotIn(t0, m.ode)
        self.assertNotIn(t0, m.alg)
        # Derivative and algebraic variable entries at t0 should be gone
        self.assertNotIn(t0, m.dv1)
        self.assertNotIn(t0, m.v2)
        # State variable entry at t0 must be preserved
        self.assertIn(t0, m.v1)
        # disc_eq must be completely untouched
        self.assertEqual(len(m.dv1_disc_eq), 4)
        # Collocation point entries should still exist
        fe = m.t.get_finite_elements()
        self.assertIn(fe[1], m.ode)
        self.assertIn(fe[1], m.dv1)

    def test_deactivate_legendre(self):
        """clean_model='deactivate' deactivates constraints at all FE boundaries for LEGENDRE."""
        m = self._make_ode_model()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=2, ncp=2, scheme='LAGRANGE-LEGENDRE', clean_model='deactivate')
        fe = m.t.get_finite_elements()
        # All FE boundaries are non-collocation points for Legendre
        for t_fe in fe:
            self.assertFalse(m.ode[t_fe].active, f"ode[{t_fe}] should be inactive")
            self.assertFalse(m.alg[t_fe].active, f"alg[{t_fe}] should be inactive")
        # Interior (collocation point) constraints should still be active
        non_fe = [t for t in m.t if t not in fe]
        for t_cp in non_fe:
            self.assertTrue(m.ode[t_cp].active, f"ode[{t_cp}] should be active")
        # cont_eq and disc_eq must be completely untouched
        self.assertEqual(len(m.dv1_disc_eq), 4)
        self.assertEqual(len(m.v1_t_cont_eq), 2)
        # State variable must exist at all FE boundaries
        for t_fe in fe:
            self.assertIn(t_fe, m.v1)

    def test_delete_legendre(self):
        """clean_model='delete' removes constraint and variable entries at FE boundaries for LEGENDRE."""
        m = self._make_ode_model()
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=2, ncp=2, scheme='LAGRANGE-LEGENDRE', clean_model='delete')
        fe = m.t.get_finite_elements()
        # All FE boundaries should be absent from user constraints and non-state vars
        for t_fe in fe:
            self.assertNotIn(t_fe, m.ode, f"ode[{t_fe}] should be deleted")
            self.assertNotIn(t_fe, m.alg, f"alg[{t_fe}] should be deleted")
            self.assertNotIn(t_fe, m.dv1, f"dv1[{t_fe}] should be deleted")
            self.assertNotIn(t_fe, m.v2, f"v2[{t_fe}] should be deleted")
            # State variable must be preserved at FE boundaries (needed for cont_eq)
            self.assertIn(t_fe, m.v1, f"v1[{t_fe}] should be preserved")
        # disc_eq and cont_eq must be completely untouched
        self.assertEqual(len(m.dv1_disc_eq), 4)
        self.assertEqual(len(m.v1_t_cont_eq), 2)

    def test_multi_index_var_delete(self):
        """clean_model='delete' handles Var(m.s, m.t) with correct index position."""
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.s = Set(initialize=[1, 2, 3])
        m.v1 = Var(m.t)
        m.dv1 = DerivativeVar(m.v1)
        m.u = Var(m.s, m.t)  # algebraic: s is first index, t is second
        m.ode = Constraint(m.t, rule=lambda m, t: m.dv1[t] == -m.v1[t])
        m.alg = Constraint(m.s, m.t, rule=lambda m, s, t: m.u[s, t] == m.v1[t])

        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=1, ncp=2, clean_model='delete')
        t0 = m.t.first()

        # All (s, t0) entries of the algebraic variable should be gone
        for si in [1, 2, 3]:
            self.assertNotIn((si, t0), m.u)
        # The constraint at t0 should be gone
        self.assertNotIn(t0, m.ode)
        for si in [1, 2, 3]:
            self.assertNotIn((si, t0), m.alg)
        # State variable must still have t0
        self.assertIn(t0, m.v1)

    def test_scalar_ic_preserved(self):
        """Scalar initial condition constraints are not affected by clean_model='delete'."""
        m = self._make_ode_model()
        m.ic = Constraint(expr=m.v1[0] == 1.0)
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, nfe=2, ncp=2, clean_model='delete')
        # The scalar IC constraint must still be active
        self.assertTrue(m.ic.active)

    def test_invalid_option(self):
        """An invalid clean_model value raises an error during apply_to."""
        m = self._make_ode_model()
        disc = TransformationFactory('dae.collocation')
        with self.assertRaises(Exception):
            disc.apply_to(m, nfe=2, ncp=2, clean_model='bad_value')


if __name__ == '__main__':
    unittest.main()
