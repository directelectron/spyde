Vector Orientation Mapping's Compute Maps no longer fails on a scan with one
degenerate position ("The input matrix is singular"); that position's strain
is NaN like one with too few pairs. A multi-phase result puts the phase map
and one orientation map per phase on the orientation window as chips, so a
single phase can be looked at on its own; the phase map was painting every
position grey for the matcher's result. The matched pattern's circles now
fade with each reflection's intensity, so the spots the match rests on stand
out from the barely excited ones.
