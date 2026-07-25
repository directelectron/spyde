- [x] Remove the link which will stick two figures together. It sucks and is annoying. It's better to have the alignment where two figures will snap side/side or on resize.
  - Already gone: commit `9bfe6ae` removed the edge-snap grouping (groups, 🔗 badge,
    dwell-to-merge, `onStuckMove`/`groupNudge`, linked resize, shake-to-break).
    Snap-on-drag (`snapPosition`) and snap-on-resize (`snapSize`) both remain, fed
    live peer rects by `MDIArea`. Verified in the running app + pinned by
    `mdi_layout.spec.ts` ("snapped windows stay INDEPENDENT"): windows snap flush,
    and dragging or resizing one leaves its neighbour untouched.

- [x] The vector orientation map results are a little off
  - FIXED (real bug): the scipy path resolved the orientation quaternion from the
    bare `theta` pose parameter, but LM freely parks part of the total rotation in
    the free 2x2 `A` — so the reported in-plane orientation was short by that
    amount. `strain_from_pose` already polar-decomposed `M = A·Rot(theta)` and its
    docstring even claimed "the rotation R is absorbed into the orientation";
    nothing did. Now resolved via `pose_in_plane_angle`. On sped_ag vs the dense
    raw-OM reference, IPF X/Y/Z agreement: **25%/4%/67% → 72%/67%/67%** with its
    own seed, and **44%/2%/100% → 99%/98%/100%** when handed a good seed, i.e.
    exact parity with the production path. Pinned by
    `test_vector_orientation_seed.py::TestPoseInPlaneAngle`.
  - The PRODUCTION batched-torch path (what a map actually uses) was already at
    98%/98%/100% and is unaffected — there `M = S·Rot(theta)` with `S` SPD by
    construction, so its `theta` is the physical angle. So the bug bit the
    no-torch fallback and the live refine overlay, not the map.
  - Ruled out with real-data A/Bs, documented in code so they aren't re-attempted:
    cosine-normalising the coarse-seed correlation (a clear win on synthetic data,
    collapses real agreement to 0%/28%/0%), and refining more coarse candidates /
    adding a measured-coverage term to the candidate score (neutral, ~40% slower).
  - Remaining, if the fallback path matters: its pyxem rasterised-delta seed still
    picks the wrong template often enough to hold IPF-Z at 67%. Handing it the
    batched polar-histogram seed measured 99%/98%/100%.
