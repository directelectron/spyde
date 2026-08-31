The DPC field map now recomputes as the beam region is dragged, instead of
staying frozen until the pass before it had finished. The beam region is a
real selector, so a superseded measurement is cancelled rather than left to
run, and the centre of mass is ~37x faster.
