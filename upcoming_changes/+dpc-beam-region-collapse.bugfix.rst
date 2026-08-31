The DPC beam region no longer collapses the moment the pointer enters the
diffraction pattern. Its radius was being written to the widget alone, while
the figure's own state kept the radius the widget was created with — a fifth
as large — and reverted to it on the next redraw, so the region that was
actually measured was a few pixels wide and dragging it moved that.
