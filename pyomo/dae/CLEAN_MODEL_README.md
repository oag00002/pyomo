# `clean_model` Option for `dae.collocation`

## Background and Motivation

When `dae.collocation` discretizes a model using orthogonal collocation, the
user's ODE/DAE constraints are expanded to **all** time points in the
`ContinuousSet` via `expand_components()`. This includes points where
derivative and algebraic variables are mathematically undefined by the
discretization scheme — the **non-collocation points**:

- **LAGRANGE-RADAU**: only `t = t_0` (the initial time). Every finite element's
  right boundary is a collocation point, so `t_0` is the only exception.
- **LAGRANGE-LEGENDRE**: every finite element boundary (`t_0`, and all interior
  FE junction points). Only the points strictly inside each element are
  collocation points.

The discretization equations (`*_disc_eq`) and continuity equations
(`*_cont_eq`) already handle this correctly — they skip non-collocation points
via `Constraint.Skip`. The problem is the **user's model equations**, which
expand everywhere. At non-collocation points, derivative and algebraic variables
appear in these constraints but are never defined by any discretization equation,
creating redundant, ill-posed rows in the NLP.

In practice this is not always destructive to solver convergence, but it
produces a concrete artifact: after `reduce_collocation_points` makes a control
variable `u` piecewise-constant, the `u` entries at non-collocation points are
free variables with no effect on the objective. IPOPT leaves them at their
initialization value (typically 0), and plotting `u` over `sorted(m.t)` shows
jagged dips at every non-collocation point.

## The `clean_model` Flag

A new configuration option `clean_model` has been added to
`dae.collocation.apply_to`. It accepts three values:

| Value | Behavior |
|---|---|
| `'none'` | Default. No change from existing behavior. |
| `'deactivate'` | Deactivates user Constraints at non-collocation points. Variables at those points remain (as free, unconstrained entries). |
| `'delete'` | Permanently deletes variable and constraint entries at non-collocation points. Produces the cleanest model and the cleanest plots. |

### Usage

```python
discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(m, nfe=20, ncp=3, scheme='LAGRANGE-LEGENDRE',
                     clean_model='delete')
```

### Design Decisions

**Why `'none'` as default?** Full backward compatibility. Existing models and
tests are unaffected unless the user opts in.

**Why `'deactivate'` as a separate option?** Pyomo `VarData` has no
`deactivate()` interface, so deactivate mode only touches constraints. The
redundant variable entries remain as free variables, which is cleaner for
`pprint()` inspection but still presents them to the solver. `'delete'` is
strictly preferable for solver performance and plot correctness.

**State variables are always preserved.** The state variable (the `_sVar` of
each `DerivativeVar`) must retain entries at all time points — its values at
non-collocation FE boundaries are used by `*_cont_eq` continuity equations
(LEGENDRE) and by the Lagrange polynomial interpolation. `id(svar)` is
collected from `reclassified_list` and those Var components are skipped
entirely in the deletion pass.

**`*_disc_eq` and `*_cont_eq` are always preserved.** These are the
correctly-formed discretization and continuity equations. They are identified
by suffix and skipped.

**Scalar constraints are never touched.** Initial condition constraints such as
`m.ic = Constraint(expr=m.v1[0] == 1.0)` are not indexed, so `is_indexed()`
returns False and they are skipped automatically.

**Non-collocation points are identified from `*_disc_eq`.** Rather than
replicating the scheme-specific skip logic from `_lagrange_radau_transform` and
`_lagrange_legendre_transform`, the cleanup reads the already-built `disc_eq`
constraint, whose existing indices are exactly the collocation points. This
makes `get_non_collocation_indices` scheme-agnostic and robust.

**Multi-dimensional index handling.** Variables like `Var(m.s, m.t)` require
knowing the flat tuple position of the `ContinuousSet` index. The helper
`_get_ds_flat_positions` sums subset dimensions using the same approach as
`get_index_information` in `misc.py`, giving the correct offset for any
index-set structure.

**Cleanup timing.** `clean_model` is stored as `self._clean_model` alongside
`self._scheme_name` during `_apply`, and the cleanup is triggered at the end
of `_transformBlock` — after `expand_components` and after all `DerivativeVar`
reclassification is complete — so `reclassified_list` is fully populated before
the deletion pass runs.

## Files Changed

### `pyomo/dae/plugins/colloc.py`
- Added `clean_model` `ConfigValue` with `domain=In(['none', 'deactivate', 'delete'])` and default `'none'`.
- Added `self._clean_model = config.clean_model` alongside existing instance-variable assignments in `_apply`.
- Added cleanup dispatch block at the end of `_transformBlock` (replacing a pre-existing TODO comment).
- Added two new imports from `pyomo.dae.misc`.

### `pyomo/dae/misc.py`
Four new public/private functions appended at the end of the file:

- **`get_non_collocation_indices(d, ds)`** — Returns the set of ContinuousSet values that are not collocation points for a given (reclassified) DerivativeVar, by reading its `*_disc_eq` constraint.
- **`_get_ds_flat_positions(comp, ds_name_to_info)`** — Private helper that computes the flat tuple position of each ContinuousSet in a component's index set, for correct multi-dimensional index extraction.
- **`deactivate_model_at_non_colloc_points(block, reclassified_list)`** — Public utility that deactivates user Constraints at non-collocation points. Lives in `misc.py` for reuse by other discretization methods (e.g., implicit Euler).
- **`delete_model_at_non_colloc_points(block, reclassified_list)`** — Public utility that deletes variable and constraint entries at non-collocation points, preserving state variables and protected constraints.

Both public functions emit a `logger.warning` when the model contains more than
one `ContinuousSet` (PDE case; see Limitations below).

### `pyomo/dae/tests/test_colloc.py`
- Added `Constraint` to the `from pyomo.environ import ...` line.
- Added `TestCleanModel` test class with 8 tests covering:
  - Default `'none'` (backward compatibility)
  - RADAU `'deactivate'` and `'delete'`
  - LEGENDRE `'deactivate'` and `'delete'`
  - Multi-index variable (`Var(m.s, m.t)`)
  - Scalar IC constraint preservation
  - Invalid option rejection

### `pyomo/dae/tests/inspect_clean_model.py` *(new)*
Standalone visual inspection script. Runs all six combinations (RADAU and
LEGENDRE × `'none'`, `'deactivate'`, `'delete'`) and prints `pprint()` output
alongside an index summary showing which entries exist and which constraints are
active. Run from the repo root:
```
python pyomo/dae/tests/inspect_clean_model.py
```

### `examples/dae/run_Optimal_Control_clean_model.py` *(new)*
Demonstrates `clean_model='delete'` on the Optimal Control example (Ex 1 from
Dynopt Guide). Uses LAGRANGE-LEGENDRE to expose the jagged artifact, clones the
model, solves both baseline and clean versions, and plots four figures for
side-by-side comparison. Iterates `sorted(m2.u)` (not `sorted(m2.t)`) for the
clean control profile to avoid re-creating deleted entries.

### `examples/dae/run_Path_Constraint_clean_model.py` *(new)*
Same structure for the Path Constraint example (Ex 4 from Dynopt Guide). Uses
the `plotter` helper from the original `run_Path_Constraint.py`. Passes `m2.u`
as the index-set argument to `plotter` for the clean control subplot, so the
function iterates only the collocation-point keys that still exist after
deletion.

## Plotting Caveat

After `clean_model='delete'`, iterating `sorted(m.t)` to plot a control
variable will silently re-create deleted `VarData` entries with `None` values
(Pyomo's `__missing__` behavior), reproducing the jagged artifact. Always
iterate `sorted(m.u)` or `m.u.keys()` for any variable whose non-collocation
entries have been deleted.

## Limitations and Future Work

### PDE Systems
The functions in `misc.py` operate on all `ContinuousSet` dimensions present in
`reclassified_list`. For PDE models with both a time and a spatial
`ContinuousSet`, cleanup will also delete algebraic variable entries at spatial
non-collocation points. This is structurally correct for the collocation
equations but can inadvertently remove algebraic variable entries needed to
express spatial boundary conditions. State variables are always preserved. A
`logger.warning` is emitted when multiple `ContinuousSet`s are detected.
**This feature is validated for ODE/DAE systems only.**

### Implicit Euler Extension
`deactivate_model_at_non_colloc_points` and `delete_model_at_non_colloc_points`
live in `misc.py` and take only `block` and `reclassified_list` — they carry no
collocation-specific logic. The implicit Euler discretization in
`pyomo/dae/plugins/finitedifference.py` creates the same redundant structure at
non-collocation points and could call these same functions with an equivalent
`clean_model` flag added to its `CONFIG`.

### `pyomo.contrib.mpc` Integration
The `pyomo.contrib.mpc` package constructs collocation-based controller models
(e.g., CSTR examples using LAGRANGE-LEGENDRE) that exhibit the same redundant
variable structure. Specifically, multi-indexed state variables like
`conc[t, comp]` end up with derivative and algebraic entries at all FE boundary
time points. The `clean_model='delete'` option can be passed directly through
`dae.collocation.apply_to` in any `contrib.mpc` model construction workflow.
However, any downstream code in `contrib.mpc` that iterates over the full time
set to extract or plot variable values will need to be updated to iterate over
`var.keys()` instead of `m.time` for variables that have had non-collocation
entries deleted — identical to the plotting fix described above.
