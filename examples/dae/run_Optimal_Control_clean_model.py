# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.environ as pyo
from Optimal_Control import m

# Clone the model so we can compare baseline vs. clean_model='delete'
m2 = m.clone()

# Baseline: LAGRANGE-LEGENDRE without clean_model.
# All finite element boundary points are non-collocation points, so u is
# unconstrained there. IPOPT leaves these at their initialization value,
# producing jagged artifacts in the control profile plot.
discretizer = pyo.TransformationFactory('dae.collocation')
discretizer.apply_to(m, nfe=20, ncp=3, scheme='LAGRANGE-LEGENDRE')
discretizer.reduce_collocation_points(m, var=m.u, ncp=1, contset=m.t)
pyo.SolverFactory('ipopt').solve(m, tee=True)

# clean_model='delete': removes u at all non-collocation points so IPOPT never
# sees spurious degrees of freedom and the control profile plot is artifact-free.
discretizer2 = pyo.TransformationFactory('dae.collocation')
discretizer2.apply_to(m2, nfe=20, ncp=3, scheme='LAGRANGE-LEGENDRE',
                      clean_model='delete')
discretizer2.reduce_collocation_points(m2, var=m2.u, ncp=1, contset=m2.t)
pyo.SolverFactory('ipopt').solve(m2, tee=True)

import matplotlib.pyplot as plt

t = sorted(m.t)
t2 = sorted(m2.t)
t2_u = sorted(m2.u)

plt.figure(1)
plt.plot(t, [pyo.value(m.x1[i]) for i in t])
plt.plot(t, [pyo.value(m.x2[i]) for i in t])

plt.figure(2)
plt.plot(t, [pyo.value(m.u[i]) for i in t])

plt.figure(3)
plt.plot(t2, [pyo.value(m2.x1[i]) for i in t2])
plt.plot(t2, [pyo.value(m2.x2[i]) for i in t2])

# Iterate m2.u directly to avoid accessing deleted non-collocation entries
plt.figure(4)
plt.plot(t2_u, [pyo.value(m2.u[i]) for i in t2_u])

plt.show()