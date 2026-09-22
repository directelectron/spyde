Scrubbing the Summed node of a reopened multi-angle acquisition reads at the
store's decode floor. Each angle was read through a lazy slice of the stack,
55–80 ms per angle against a 25 ms chunk decode; the node now reads every
angle through the stack's one store reader. Measured on a real four-angle
stack with 51 MB chunks: a cold frame 282 → 99 ms, a chunk crossing 287 →
108 ms, a frame inside a decoded chunk 2 ms. Composed nodes also tell the
read-ahead where their next chunk boundary is, so a drag decodes the next
chunk before it arrives.
