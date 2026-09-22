File → Export to NumPy writes the maps the focused window shows as one
``.npz`` (or the shown map alone as ``.npy``): every component of a committed
Strain, DPC or orientation tree under a typeable key (``exx``, ``Bx``,
``abs_B``), the chip views of an IPF window, the raw fields of the result
behind it (quaternions, the fractional strain tensor, the un-rotated field),
a live virtual image, and the live Strain window's field. The file carries
the calibrated axes and a ``meta_json`` record naming each key's label,
quantity and provenance. A window on a dataset exports only the frame it
shows; Save Signal remains the door for the data itself.
