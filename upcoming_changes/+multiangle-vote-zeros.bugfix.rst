Multi-angle real-space alignment: a member whose true offset on an axis is
zero no longer has its correct votes discarded. The patch vote dropped every
zero so an axis with no signal could not look unanimous, which left the few
patches that mis-registered a genuinely unshifted member to set its offset
unopposed, at an agreement of one. Zeros now count in the median and are left
out only of the agreement. A patch too small to score no longer hands the
choice to the raw-intensity default while reporting a gain of -inf; the
sum-over-angles node keeps the accumulator it chose instead of numpy's
uint64; and a saved 4-D sum that inherited the stack's signal type is no
longer mistaken for a stack on reopening.
