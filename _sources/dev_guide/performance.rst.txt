Performance:
------------

One of the main goals of this project is to provide high performance plotting capabilities for
large datasets lazily.  Building on top of hyperspy the most important performance consideration
was how can we avoid loading large datasets into memory until absolutely necessary. Can we create
a workflow where users can interactively explore large datasets, transform them through things like
filtering, background subtractions, peak finding, and only interact with bits and pieces of the data
when absolutely necessary.


End-to-end lazy data pipeline
-----------------------------

The first part of this is achieved through an end-to-end lazy data pipeline. In SpyDE this is visualized
as a SignalTree. Each node represents a hyperspy signal object.  In hyperspy most lazy operations are
implemented through the `map` function.  These are functions that operate on the signal of the dataset.
Think 1 diffraction pattern in a 4D STEM dataset, 1 EDS or EELS spectrum, or 1 image in an in situ TEM
time series. The idea is that you can build a workflow. Something like:

1. Load 4D STEM dataset
2. Apply a background subtraction to each diffraction pattern
3. Find peaks in each diffraction pattern
4. Build a strain matrix from the peak positions

In SpyDE each of these steps would be represented as a node in the SignalTree and will share the
same navigation image.  You can toggle between the different nodes to visualize the results which will
be computed as needed for display.  This can cause extra delay for visualization, but most operations
are very fast.  This also allows us to easily play with different processing parameters, even compare
before computing/saving the entire strain matrix.

Visualization:
--------------
Visualization is often a difficult part of working with large datasets.  For large images pyqtgraph
does a really good job of handling/ rendering large images.  We avoid things like resetting the view
and the contrast levels as plots update to avoid some of the more expensive re-rendering operations. The
hardest part, however, is visualizing data stored on disk rather then memory.  Data needs to:

1. Be loaded from disk to CPU memory
2. Transferred to the GPU
3. Rendered

What actually happens is:
1. Load a small chunk of data from disk to CPU memory
  a. Optionally decompress if needed
  b. Perform any calculations on the chunk
  c. Transfer the data over TCP from the worker process to the main CPU process
2. Transfer the chunk to the GPU
3. Render the chunk

Of all of these steps 1c is often the slowest.

