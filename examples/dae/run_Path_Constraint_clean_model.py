# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from Path_Constraint import m

# Clone the model so we can compare baseline vs. clean_model='delete'
m2 = m.clone()

# Baseline: LAGRANGE-LEGENDRE without clean_model.
# All finite element boundary points are non-collocation points, so u is
# unconstrained there. IPOPT leaves these at their initialization value,
# producing jagged artifacts in the control profile plot.
disc = pyo.TransformationFactory('dae.collocation')
disc.apply_to(m, nfe=7, ncp=6, scheme='LAGRANGE-LEGENDRE')
disc.reduce_collocation_points(m, var=m.u, ncp=1, contset=m.t)
pyo.SolverFactory('ipopt').solve(m, tee=True)

# clean_model='delete': removes u at all non-collocation points so IPOPT never
# sees spurious degrees of freedom and the control profile plot is artifact-free.
disc2 = pyo.TransformationFactory('dae.collocation')
disc2.apply_to(m2, nfe=7, ncp=6, scheme='LAGRANGE-LEGENDRE', clean_model='delete')
disc2.reduce_collocation_points(m2, var=m2.u, ncp=1, contset=m2.t)
pyo.SolverFactory('ipopt').solve(m2, tee=True)


def plotter(subplot, x, *series, **kwds):
    plt.subplot(subplot)
    for i, y in enumerate(series):
        plt.plot(
            list(x),
            [pyo.value(y[t]) for t in x],
            'brgcmk'[i % 6] + kwds.get('points', ''),
        )
    plt.title(kwds.get('title', ''))
    plt.legend(tuple(y.name for y in series), frameon=True, edgecolor='k').draw_frame(
        True
    )
    plt.xlabel(x.name)
    plt.gca().set_xlim([0, 1])


import matplotlib.pyplot as plt

plt.figure(1)
plotter(121, m.t, m.x1, m.x2, m.x3, title='Differential Variables (baseline)')
plotter(122, m.t, m.u, title='Control Variables (baseline)', points='o-')

plt.figure(2)
plotter(121, m2.t, m2.x1, m2.x2, m2.x3, title="Differential Variables (clean_model='delete')")
# Pass m2.u as the index set so only collocation-point entries are plotted
plotter(122, m2.u, m2.u, title="Control Variables (clean_model='delete')", points='o-')

plt.show()
