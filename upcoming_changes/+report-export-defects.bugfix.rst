A movie cell in a report now exports: every HTML export carries its poster still
with a play badge and its caption, and an interactive export inlines the rendered
movie itself when the file is inside the 100 MB embed budget, saying the size and
the budget when it is not. A figure with no pixels keeps its caption instead of
vanishing, and an exported interactive figure is sized from its own shape the way
the sidebar cell is, rather than dropped into one 480 px box that letterboxed a
wide row of panels and stretched a single square pattern.

In the sidebar, a window dropped on the report body between two cells now becomes
a figure cell even when the drag arrives with an unreadable payload, which is how
a real drag out of the OS can look; and a split block's text side is the same
markdown editor a text cell has, with the formatting toolbar and Ctrl-B / Ctrl-I
it was missing. Emptying a figure cell back to a placeholder no longer leaves it
holding its baked image, so a later save does not write an asset the cell no
longer owns.
