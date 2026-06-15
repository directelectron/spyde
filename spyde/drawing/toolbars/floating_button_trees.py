from typing import Optional, Dict, Any, List
from PySide6 import QtWidgets, QtGui, QtCore

from spyde.qt.style import SmoothButton


class RoundedButton(SmoothButton):
    """
    Minimal styled push button with optional icon and text — borderless
    dark fill, painted with antialiased rounded corners by SmoothButton.
    """

    def __init__(
        self,
        icon_path: Optional[str] = None,
        text: Optional[str] = None,
        tooltip: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(
            text or "",
            parent,
            fill="rgba(30,30,30,230)",
            fill_hover="rgba(40,40,40,230)",
            fill_pressed="rgba(50,50,50,230)",
            border=None,
            font_px=None,  # keep the app font (matches the old QSS look)
            margin=2,
        )

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setIconSize(QtCore.QSize(18, 18))
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        if icon_path:
            self.setIcon(QtGui.QIcon(icon_path))
        if tooltip:
            self.setToolTip(tooltip)


class ButtonTree(QtWidgets.QWidget):
    """
    Layout a tree of buttons using the Reingold-Tilford tidy tree algorithm (Buchheim et al., 2002),
    oriented horizontally (root on the left, children to the right).
    Input 'tree' is a nested dict: { 'Root': { 'Child A': {...}, 'Child B': {...} }, ... }.
    """

    class _TreeNode:
        def __init__(
            self,
            label: str,
            button: Optional[QtWidgets.QPushButton],
            children: Optional[List["ButtonTree._TreeNode"]] = None,
            parent: Optional["ButtonTree._TreeNode"] = None,
            is_dummy: bool = False,
        ):
            self.label = label
            self.button = button
            self.children: List["ButtonTree._TreeNode"] = children or []
            self.parent = parent
            self.is_dummy = is_dummy

            # Tidy algorithm fields
            self.x = 0.0
            self.y = 0
            self.prelim = 0.0
            self.mod = 0.0
            self.change = 0.0
            self.shift = 0.0
            self.thread: Optional["ButtonTree._TreeNode"] = None
            self.ancestor: "ButtonTree._TreeNode" = self
            self.number = 0  # index among siblings

        def left(self) -> Optional["ButtonTree._TreeNode"]:
            return self.children[0] if self.children else self.thread

        def right(self) -> Optional["ButtonTree._TreeNode"]:
            return self.children[-1] if self.children else self.thread

        def left_sibling(self) -> Optional["ButtonTree._TreeNode"]:
            if not self.parent or self.number == 0:
                return None
            return self.parent.children[self.number - 1]

        def leftmost_sibling(self) -> Optional["ButtonTree._TreeNode"]:
            if not self.parent or not self.parent.children:
                return None
            return self.parent.children[0]

    def __init__(self, title: str, tree: Dict[str, Any]):
        super().__init__()
        self.setObjectName("ButtonTree")

        # Scene/view boilerplate
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        self.graphics_view = QtWidgets.QGraphicsView(self)
        self.graphics_view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.graphics_view.setMinimumSize(250, 150)
        self.graphics_view.setMaximumSize(300, 200)
        self.graphics_view.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.graphics_view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.graphics_view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        # Transparent background for the view and viewport (remove frame)
        self.graphics_view.setStyleSheet("background: transparent; border: none;")
        self.graphics_view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.graphics_view.viewport().setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TranslucentBackground
        )
        self.graphics_view.viewport().setAutoFillBackground(False)

        self.graphics_scene = QtWidgets.QGraphicsScene(self)
        self.graphics_scene.setBackgroundBrush(QtCore.Qt.GlobalColor.transparent)
        self.graphics_view.setScene(self.graphics_scene)
        self._layout.addWidget(self.graphics_view)

        # Node geometry and spacing (px)
        self.node_w = 80
        self.node_h = 25
        self.level_gap = 30  # gap between parent and child columns (x direction)
        self.sibling_gap = 20  # gap between siblings (y direction)

        # Build tidy layout and paint (defer to next event loop tick so view is ready)
        root = self._build_tree(tree)

        # update the size to accommodate the tree
        # Defer layout/render to ensure widget has been laid out and view sizes are valid
        QtCore.QTimer.singleShot(
            0, lambda r=root: (self._reingold_tilford_layout(r), self._render_tree(r))
        )

    # --- Build an internal tree from nested dict --------------------------------

    def _build_tree(self, d: Dict[str, Any]) -> "_TreeNode":
        roots: List[ButtonTree._TreeNode] = []
        for label, subtree in d.items():
            btn = label
            node = self._build_subtree(label, subtree, btn, None)
            roots.append(node)

        if not roots:
            # Empty tree; provide a dummy root for stability
            return ButtonTree._TreeNode("__root__", None, [], None, is_dummy=True)

        if len(roots) == 1:
            return roots[0]

        # Multiple roots -> create a dummy super-root (not rendered)
        super_root = ButtonTree._TreeNode("__root__", None, is_dummy=True, parent=None)
        for i, r in enumerate(roots):
            r.parent = super_root
            r.number = i
        super_root.children = roots
        return super_root

    def _build_subtree(
        self,
        label: str,
        subtree: Any,
        button: QtWidgets.QPushButton,
        parent: Optional["_TreeNode"],
    ) -> "_TreeNode":
        node = ButtonTree._TreeNode(label, button, [], parent, is_dummy=False)
        if isinstance(subtree, dict):
            for i, (child_label, child_subtree) in enumerate(subtree.items()):
                child_btn = child_label
                child_node = self._build_subtree(
                    child_label, child_subtree, child_btn, node
                )
                child_node.number = i
                node.children.append(child_node)
        # Treat None or non-dict as leaf
        return node

    # --- Reingold-Tilford (Buchheim et al.) ------------------------------------

    def _reingold_tilford_layout(self, root: "_TreeNode") -> None:
        # Initialize sibling numbers
        def assign_numbers(n: ButtonTree._TreeNode):
            for i, c in enumerate(n.children):
                c.number = i
                assign_numbers(c)

        assign_numbers(root)

        # First and second walks
        self._first_walk(root)
        min_x = self._second_walk(root, 0.0, 0)

        # Shift so that all x >= 0
        if min_x < 0:
            self._shift_tree(root, -min_x)

    def _first_walk(self, v: "_TreeNode") -> None:
        if not v.children:
            ls = v.left_sibling()
            v.prelim = (ls.prelim + 1.0) if ls else 0.0
        else:
            default_ancestor = v.children[0]
            for w in v.children:
                self._first_walk(w)
                default_ancestor = self._apportion(w, default_ancestor)
            self._execute_shifts(v)
            left = v.children[0]
            right = v.children[-1]
            midpoint = (left.prelim + right.prelim) / 2.0
            ls = v.left_sibling()
            if ls:
                v.prelim = ls.prelim + 1.0
                v.mod = v.prelim - midpoint
            else:
                v.prelim = midpoint

    def _apportion(self, v: "_TreeNode", default_ancestor: "_TreeNode") -> "_TreeNode":
        w = v.left_sibling()
        if w is None:
            return default_ancestor

        vir = v
        vor = v
        vil = w
        vol = v.leftmost_sibling()

        sir = vir.mod
        sor = vor.mod
        sil = vil.mod
        sol = vol.mod if vol else 0.0

        while vil.right() and vir.left():
            vil = vil.right()
            vir = vir.left()
            vol = vol.left() if vol else None
            vor = vor.right() if vor else None
            if vor:
                vor.ancestor = v
            shift = (vil.prelim + sil) - (vir.prelim + sir) + 1.0
            if shift > 0:
                a = self._ancestor(vil, v, default_ancestor)
                self._move_subtree(a, v, shift)
                sir += shift
                sor += shift
            sil += vil.mod
            sir += vir.mod
            sol += vol.mod if vol else 0.0
            sor += vor.mod if vor else 0.0

        if vil.right() and not (vor and vor.right()):
            if vor:
                vor.thread = vil.right()
                vor.mod += sil - sor
        if vir.left() and not (vol and vol.left()):
            if vol:
                vol.thread = vir.left()
                vol.mod += sir - sol

        return default_ancestor

    def _move_subtree(self, wl: "_TreeNode", wr: "_TreeNode", shift: float) -> None:
        subtrees = wr.number - wl.number
        if subtrees <= 0:
            return
        ratio = shift / subtrees
        wr.change += shift
        wr.shift += shift
        wl.change -= ratio

    def _execute_shifts(self, v: "_TreeNode") -> None:
        shift = 0.0
        change = 0.0
        for w in reversed(v.children):
            w.prelim += shift
            w.mod += shift
            change += w.change
            shift += w.shift + change

    def _ancestor(
        self, vil: "_TreeNode", v: "_TreeNode", default_ancestor: "_TreeNode"
    ) -> "_TreeNode":
        if vil.ancestor and vil.ancestor in (v.parent.children if v.parent else []):
            return vil.ancestor
        return default_ancestor

    def _second_walk(self, v: "_TreeNode", m: float, depth: int) -> float:
        v.x = v.prelim + m
        v.y = depth
        min_x = v.x
        for w in v.children:
            min_x = min(min_x, self._second_walk(w, m + v.mod, depth + 1))
        return min_x

    def _shift_tree(self, v: "_TreeNode", dx: float) -> None:
        v.x += dx
        for w in v.children:
            self._shift_tree(w, dx)

    # --- Render to QGraphicsScene ----------------------------------------------

    def _render_tree(self, root: "_TreeNode") -> None:
        self.graphics_scene.clear()

        # Collect all non-dummy nodes
        nodes: List[ButtonTree._TreeNode] = []

        def collect(n: ButtonTree._TreeNode):
            if not n.is_dummy:
                nodes.append(n)
            for c in n.children:
                collect(c)

        collect(root)

        # Precompute positions (horizontal = depth, vertical = breadth)
        pos: Dict[ButtonTree._TreeNode, QtCore.QPointF] = {}
        for node in nodes:
            px = node.y * (self.node_w + self.level_gap)
            py = node.x * (self.node_h + self.sibling_gap)
            pos[node] = QtCore.QPointF(px, py)

        # Draw connecting edges behind nodes
        pen = QtGui.QPen(QtGui.QColor(180, 180, 180, 200))
        pen.setWidthF(1.25)
        pen.setCosmetic(True)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)

        for parent in nodes:
            for child in parent.children:
                if child.is_dummy:
                    continue
                p0 = pos[parent] + QtCore.QPointF(
                    self.node_w, self.node_h / 2.0
                )  # right-center of parent
                p1 = pos[child] + QtCore.QPointF(
                    0.0, self.node_h / 2.0
                )  # left-center of child
                path = QtGui.QPainterPath(p0)
                dx = max(20.0, (self.level_gap + self.node_w) * 0.35)
                path.cubicTo(
                    p0 + QtCore.QPointF(dx, 0.0), p1 - QtCore.QPointF(dx, 0.0), p1
                )
                edge_item = self.graphics_scene.addPath(path, pen)
                edge_item.setZValue(-1.0)  # keep edges behind nodes

        # Add node widgets
        for node in nodes:
            if isinstance(node.button, QtWidgets.QWidget):
                node.button.setFixedSize(self.node_w, self.node_h)
                item = self.graphics_scene.addWidget(node.button)
                item.setPos(pos[node])
            else:
                # Fallback: render label if button widget is missing
                text_item = self.graphics_scene.addText(str(node.label))
                text_item.setDefaultTextColor(QtGui.QColor(220, 220, 220))
                text_item.setPos(pos[node])

        # Fit the view to content
        self.graphics_view.setSceneRect(self.graphics_scene.itemsBoundingRect())
