# FND-1 Numerical Assessment

> Archive status: WORKING NOTE / QUARANTINED
> This note records the first reproducible benchmark batch produced by [run_fnd1_benchmarks.py](F:/Black%20Mesa%20Research%20Facility/Main%20Facility/Physics%20department/SCT%20Theory/.worktrees/speculative-wip/speculative/numerics/run_fnd1_benchmarks.py). It is a bounded numerical assessment, not a theorem-level statement.

## What Was Run

The first batch was written to [speculative/numerics/results/first_batch](F:/Black%20Mesa%20Research%20Facility/Main%20Facility/Physics%20department/SCT%20Theory/.worktrees/speculative-wip/speculative/numerics/results/first_batch).

Batch configuration:

- operator families:
  - `sj_inverse`
  - `sj_inverse_truncated`
- normalization labels:
  - `raw`
  - `density_scaled`
- sizes:
  - `N = 200, 500, 1000`
- seeds:
  - `0, 1, 2, 3, 4`

This produced:

- `60` benchmark runs
- `120` seed-level summary rows
- `30` aggregate rows

## What Failed

The clearest failure is the ultraviolet heat-kernel target.

- For `sj_inverse`, the fitted ultraviolet exponent stayed near zero across the full batch, roughly in the range `-0.0053` to `-0.0004`, far from the declared 2D target `p = -1`.
- For `sj_inverse_truncated`, the fitted ultraviolet exponent moved farther from zero but still stayed far from the target, roughly in the range `-0.1314` to `-0.1091`.
- As a result, every size-level `heat_kernel_size_pass` row is `fail`.

Because the family-level protocol requires both the spectral-dimension and heat-kernel conditions to pass together, both operator families ended the batch with:

- `family_outcome = fail`
- `normalization_sensitivity = fail`
- `ensemble_stability = fail`

## What Remained Ambiguous

The spectral-dimension side is more complicated.

- `sj_inverse` passed the size-level spectral-dimension check at all three sizes for both tested normalization labels.
- `sj_inverse_truncated` failed the spectral-dimension check at `N = 200` but passed at `N = 500` and `N = 1000`.

This means the benchmark does see a nontrivial spectral-dimension-like signal in part of the reviewed SJ family. But under the current protocol that is not enough, because the same operator families still fail the ultraviolet heat-kernel test everywhere.

One additional limitation is implementation-specific:

- in the current runner, `raw` and `density_scaled` are non-discriminating for the reported metrics because the normalization changes `lambda_sq` only by a constant factor, while the `t` grid is rebuilt from the same spectrum each time

So the present normalization comparison should not be read as physical evidence that the two labels behave the same in a deeper sense. It is mainly a sign that the current metric implementation does not separate them cleanly. For a stronger second batch, the protocol or runner may need either:

- a fixed external `t` grid rather than a spectrum-adaptive one
- an additional observable that does not collapse under simple rescaling

## Candidate Status After Batch One

Under the current benchmark rules, no reviewed candidate survives to a genuine second-round promotion.

The most honest current status is:

- `sj_inverse`: interesting negative result with a spectral-dimension signal but no heat-kernel support
- `sj_inverse_truncated`: still useful as a bounded rescue attempt, but it does not clear the combined benchmark gate

The family-level `normalization_sensitivity = fail` rows should also be read narrowly: in this protocol they mean that no normalization label cleared both benchmark gates together, not that normalization dependence has been fully mapped.

So the numerical program currently supports a stronger obstruction picture than a rescue picture for the reviewed SJ-family routes.
