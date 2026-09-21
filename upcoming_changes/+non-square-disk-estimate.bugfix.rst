Spot detection no longer fails on a diffraction pattern that is not square.
Sizing the disks — the first thing the neural detector does to every frame —
combined the autocorrelation's horizontal and vertical profiles, which only
have the same length on a square detector, so any other shape raised instead
of degrading. A composed multi-angle pattern is non-square whenever its
members' reciprocal offsets differ between the axes.
