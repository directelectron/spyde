Vector Orientation Mapping's Compute Maps opens the orientation window at
once and fills it in a band of scan rows at a time as the match lands, with
a percentage that counts positions rather than sitting on "33%" until the
whole scan had matched. The result is the same as before: the match is per
position, and the refinement still runs over the whole map. The IPF Refine
heat map also moved off the navigator thread, so dragging the crosshair with
it open no longer waits on its correlation at every step.
