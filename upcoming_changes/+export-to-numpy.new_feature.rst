File → Export to NumPy writes what the focused window shows as one
``.npz`` (or the shown array alone as ``.npy``, or as ``.csv`` columns):
every component of a committed Strain, DPC or orientation tree under a
typeable key (``exx``, ``Bx``, ``abs_B``), the chip views of an IPF window,
the raw fields of the result behind it (quaternions, the fractional strain
tensor, the un-rotated field), a live virtual image, a line profile, the
spectrum under the navigator, and the live Strain window's field. The file
carries the calibrated axes and a ``meta_json`` record naming each key's
label, quantity and provenance; a CSV puts each trace in a column against
its axis. A window on a dataset exports only the frame it shows; Save
Signal remains the door for the data itself.
