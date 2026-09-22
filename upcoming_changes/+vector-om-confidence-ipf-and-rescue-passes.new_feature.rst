Vector Orientation Mapping draws its IPF maps by confidence: a position's
colour is scaled by its correlation and a position nothing matched is grey,
so an amorphous region no longer paints as loudly as a grain and an
unmatched position no longer paints as the identity orientation (pure red
in IPF-X). The Run tab gained a "Rescue passes" setting (default 3, was one
pass): the neighbour rescue that re-fits a position from its neighbours'
orientations now repeats until a pass changes nothing, so a mis-indexed
patch wider than one position is cleaned up too.
