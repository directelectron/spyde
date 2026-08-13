# Vendored from de_ground_crew

Source: `git@github.com:CSSFrancis/de_ground_crew.git`
Commit: `e9e21de25bc10fcdc6135f34bd8ca7e736c37c6f` (origin/main)
Copied: 2026-08-13

| vendored file | upstream path | lines | sha256 (upstream) |
|---|---|---|---|
| `_motion_correction_v2.py` | `_motion_correction_v2.py` | 472 | `d734d3e676b571a3` |
| `_motion_correction_v3.py` | `_motion_correction_v3.py` | 996 | `5d8def04fd6ee072` |
| `local_motion.py` | `workers/local_motion.py` | 651 | `7ebe902bc676e85a` |
| `dose_weighting.py` | `workers/dose_weighting.py` | 87 | `6c3c56578c55e900` |
| `_image_ops.py` | `workers/_image_ops.py` | 166 | `4966977a30ad2ed0` |
| `_gpu_memory.py` | `workers/_gpu_memory.py` | 37 | `d0a36efedb0a4102` |

## Permitted edits

Only import rewrites. Upstream resolves its modules as top-level names
(`_motion_correction_v2`, `workers.dose_weighting`, …); inside a package those
cannot resolve, so each is repointed at the vendored sibling. No logic, no
constants, no formatting.

| file | line | from | to |
|---|---|---|---|
| `_motion_correction_v3.py` | 25 | `from _motion_correction_v2 import` | vendored sibling |
| `_motion_correction_v3.py` | ~416 | `from workers.dose_weighting import` | vendored sibling |
| `local_motion.py` | ~114 | `from workers.motion_correction_worker import` | `_worker_extracts` |
| `local_motion.py` | ~470 | `from workers.dose_weighting import` | vendored sibling |

`test_motion.py::TestVendorIsPristine` normalises these back before hashing, so
every other byte is still checked against the table above and drift is a test
failure rather than a discovery.

### `_worker_extracts.py`

Not in the table: it is a partial copy, not a whole file. Upstream's
`workers/motion_correction_worker.py` imports PySide6 and so cannot be vendored,
but four of its functions are pure numerics the rest of this code calls —
`_bandpass_filter`, `_cross_correlate`, `_apply_shift_fourier`,
`_compute_log_fft`. They are copied verbatim with `_GPU`/`_cp` bound to the CPU
path. Re-copy them by line number when re-syncing.

## Re-syncing

Re-copy the files, re-apply the one import edit, update this table, run the tests.
Do not merge upstream changes by hand.
