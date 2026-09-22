In the multi-angle loader the arrow keys nudge the member being compared.
The pair view's nudge pad shared a React ref with the alignment grid's, so
whichever mounted last took the focus and the keys could move the member the
other pad was showing. The tableau also grouped members onto rings by a 0.05°
tolerance while the backend counted shells at 0.01°, so the picture could
show one ring where the status line said two shells; both now use 0.01°.
