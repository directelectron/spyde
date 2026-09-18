SpyDE
-----

SpyDE is built ontop of Hyper(Spy) with original support from Direct Electron (DE) to provide a powerful and flexible
platform for data analysis and visualization. Rather than create a completely new application from scratch, a core goal
of SpyDE is to continue to support HyperSpy while providing additional features and capabilities that enhance the
user experience using a GUI-based approach.


.. grid:: 2 2 2 2
  :gutter: 2

  .. grid-item-card::
    :link: user_guide/installing
    :link-type: doc

    :octicon:`rocket;2em;sd-text-info` Getting Started
    ^^^

    New to SpyDE? The getting started guide provides an
    introduction on basic usage and how to install it.

  .. grid-item-card::
    :link: user_guide/index
    :link-type: doc

    :octicon:`book;2em;sd-text-info` User Guide
    ^^^

    The user guide provides information on key concepts of SpyDE


  .. grid-item-card::
    :link: examples/index
    :link-type: doc

    :octicon:`zap;2em;sd-text-info` Examples
    ^^^

    Gallery of short examples illustrating simple tasks that can be performed with SpyDE.

  .. grid-item-card::
    :link: dev_guide/index
    :link-type: doc

    :octicon:`people;2em;sd-text-info` Developer Guide
    ^^^

    The developer guide provides information on how to contribute to SpyDE. We are always
    looking for new contributors and welcome any contributions!

Currently, HyperSpy is much more "feature complete" than SpyDE, but we are actively working to implement much of the
functionality from HyperSpy into SpyDE, that being said there is still a very real reason to learn how to use
the HyperSpy API directly as it provides a lot more flexibility and options for advanced users and understanding
how HyperSpy works under the hood can help you get the most out of SpyDE as well.

FAQS
----

"Why base things on HyperSpy?"

> There are many new python packages for TEM data analysis which are quite brilliant.  Many of them are based on pytorch
which provides excellent GPU acceleration and a modern API.  Machine learning, neural networks, and deep learning are
certainly the future of data analysis and these packages are leading the way.  This is something that HyperSpy lags
on.  That being said there is always a place for more traditional data analysis, a stable API and broad compatibility.
Most importantly HyperSpy has a "Lazy First" approach to data handling which is very important for handling
very large datasets. A core tennet of SpyDE is that "No data should be too big to analyze".  That being said, the
hope is that SpyDE can eventually integrate some of these new machine learning based packages into its workflows.

"Why use Dask?"

> This is a great question... In particular SpyDE uses the second generation Dask Scheduler (distributed).  This is
a pretty questionable choice for an application that tries to be "responsive" as Dask is not known for its low-latency
task scheduling. That being said Dask provides a lot of benefits that are very important for SpyDE.  Dask is
designed to handle very large datasets that don't fit into memory, it can effectively schedule tasks across multiple
cores and even multiple machines, and it has a very flexible API that can be extended to handle custom data types.
HyperSpy is also very integrated with Dask which makes it a natural choice for SpyDE.  The hope is that as Dask
improves its scheduling latency that SpyDE will benefit from these improvements.

"Why use Qt/pyqtgraph?"

> First and foremost what we'd like to use is [fastplotlib](fastplotlib.org) which is built on top of WGPU and is
blazingly fast. Unfortunately fastplotlib is not (quite) mature enough...  This would also allow us to port lots of
the interactivity back to HyperSpy and really modernize/improve the plotting in HyperSpy using cross platform GPU
acceleration.

"Why not use matplotlib?"

> Matplotlib is great for static plots and publication quality figures, but it is not designed for interactive
applications.  This is one of the reasons that HyperSpy is sometimes slow when plotting large images.  It struggles
with images that are larger than 1k x 1k pixels.  Qt/pyqtgraph provides a much more responsive and interactive
experience for the user.

