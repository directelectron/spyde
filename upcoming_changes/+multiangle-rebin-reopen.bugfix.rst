A multi-angle acquisition that was rebinned in the app reopens as its tree
again. Rebin summed into uint64 — four times the bytes asked for — and the
sum over angles then refused a 64-bit stack, so the saved file opened as a
plain 5-D array with no Summed node and no angle ring, and only the log said
why. Rebin now sums into the narrowest exact width (uint32 for a uint16 scan
binned 2x2x2x2), a 64-bit stack keeps its width, and a re-expansion that
fails is reported in the app.
