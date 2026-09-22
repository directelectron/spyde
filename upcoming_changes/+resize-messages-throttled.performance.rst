Resizing a window sends the figure at most ten size updates a second
instead of one per frame. Each update round-trips through the backend and
re-sends the figure's whole image, so an orientation, phase or strain map
window used to queue dozens of full-image pushes during a corner drag and
lag its cursor.
