The angle ring works on a reopened multi-angle acquisition. It asked only
the runtime recipe whether one angle was on screen, and a reopened tree has
none, so the ring stayed on "all angles" with every member lit, no handle,
and a full node switch on every pick. A pick now stays on the node being
worked on when that is itself a stack (a binned copy, say) instead of
jumping back to the full-resolution root. A reopened tree also reports the
recorded member offsets rather than a sum's zeroed ones, names its shells
as the composition does, is brought up to the current signal type, reuses
the navigator planes the composition now keeps in the file instead of
reducing the whole stack to draw a thumbnail, and a failure while
rebuilding the tree no longer opens the dataset a second time.
