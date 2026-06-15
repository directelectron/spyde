"""
style.py — single source of truth for SpyDE's dark theme.

Every interactive widget the actions/carets build should come from the
factories here (or at least use the QSS constants), so colors, hover/press
feedback and fonts stay consistent app-wide.

Palette (matches the application stylesheet in __main__.py):

  surfaces   #0d0d0d MDI canvas · #141414 panels/docks/dialogs ·
             #1d1d1d menu/status bars · #1e1e1e raised buttons
  lines      rgba(255,255,255,60) standard border on dark surfaces
  fills      rgba(255,255,255,30) button · 40 input · 45 hover · 60 pressed
  accent     warm orange rgba(240,150,45) — checked/active states, menu
             selection, slider fill; same family as the histogram's amber
             level lines and gamma curve
  good/danger green commit · red stop/record (subwindow title bar buttons)

Conventions:
  - control text is 10px white; disabled text rgba(255,255,255,60)
  - never use "border: 1px solid black" on dark surfaces (invisible)
  - every clickable widget must have :hover and :pressed feedback
  - every button uses RADIUS; inputs/spin boxes use RADIUS_INPUT
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

# ── Tokens ────────────────────────────────────────────────────────────────────
SURFACE_CANVAS = "#0d0d0d"
SURFACE_PANEL = "#141414"
SURFACE_BAR = "#1d1d1d"
SURFACE_RAISED = "#1e1e1e"
# The main-window custom title bar — clearly lighter than the panels (#141414),
# menus (#1d1d1d) and the MDI canvas (#0d0d0d) so it reads as a distinct top
# strip rather than blending into any of them.
SURFACE_TITLEBAR = "#333333"

TEXT = "white"
TEXT_DIM = "rgba(255,255,255,160)"
TEXT_DISABLED = "rgba(255,255,255,60)"
FONT_SMALL = "10px"

BORDER = "rgba(255,255,255,60)"
BORDER_FAINT = "rgba(255,255,255,40)"

FILL_BTN = "rgba(255,255,255,30)"
FILL_HOVER = "rgba(255,255,255,45)"
FILL_PRESSED = "rgba(255,255,255,60)"
FILL_DISABLED = "rgba(255,255,255,10)"
FILL_INPUT = "rgba(255,255,255,40)"

RADIUS_F = 6.0                 # numeric radius for QPainter-drawn buttons
RADIUS = f"{int(RADIUS_F)}px"  # single rounding radius for every button app-wide
RADIUS_INPUT = "4px"           # line edits, combo boxes, spin boxes

ACCENT = "rgba(240,150,45,190)"
ACCENT_BORDER = "rgba(255,190,100,220)"
ACCENT_SOFT = "rgba(240,150,45,90)"   # checked tool icons, subtle highlights
GOOD = "rgba(80,160,80,180)"
GOOD_HOVER = "rgba(100,200,100,200)"
DANGER = "rgba(180,60,60,200)"
DANGER_HOVER = "rgba(220,80,80,220)"

# ── QSS building blocks ───────────────────────────────────────────────────────

def _button_rules(*selectors: str) -> str:
    """
    Standard themed button rules for the given base selectors.

    Needed at more than one specificity level: the plain ``QPushButton`` rule
    is the app-wide default, but descendant rules like ``QDialog QPushButton``
    must be re-emitted explicitly or higher-specificity dark-surface rules
    would strip the hover/pressed feedback.
    """
    def s(state: str = "") -> str:
        return ", ".join(f"{sel}{state}" for sel in selectors)

    return (
        f"{s()} {{ color: {TEXT}; background: {FILL_BTN}; "
        f"border: 1px solid {BORDER}; border-radius: {RADIUS}; "
        f"padding: 3px 6px; font-size: {FONT_SMALL}; }}"
        f"{s(':hover')} {{ background: {FILL_HOVER}; }}"
        f"{s(':pressed')} {{ background: {FILL_PRESSED}; }}"
        f"{s(':checked')} {{ background: {ACCENT}; "
        f"border: 1px solid {ACCENT_BORDER}; }}"
        f"{s(':disabled')} {{ color: {TEXT_DISABLED}; "
        f"background: {FILL_DISABLED}; }}"
    )


BUTTON_QSS = _button_rules("QPushButton")

# Spin-box step buttons need explicit dark-theme styling: the native Fusion
# arrows are drawn dark gray and disappear on our dark fills, so we draw the
# arrows from white SVGs shipped in qt/assets/icons.
_ICONS = (Path(__file__).resolve().parent / "assets" / "icons").as_posix()

SPIN_QSS = (
    f"QDoubleSpinBox, QSpinBox {{ color: {TEXT}; background: {FILL_INPUT}; "
    f"border: 1px solid {BORDER}; border-radius: {RADIUS_INPUT}; "
    f"font-size: {FONT_SMALL}; }}"
    f"QDoubleSpinBox:hover, QSpinBox:hover {{ border: 1px solid {TEXT_DIM}; }}"
    f"QDoubleSpinBox:disabled, QSpinBox:disabled "
    f"{{ color: {TEXT_DISABLED}; background: {FILL_DISABLED}; }}"
    # Steppers are flat rounded hover targets inset 1px from the field edge —
    # no resting fill or separator lines, so the field keeps its rounded
    # silhouette and the buttons match the app-wide soft-corner look.
    f"QDoubleSpinBox::up-button, QSpinBox::up-button {{ "
    f"subcontrol-origin: border; subcontrol-position: top right; width: 16px; "
    f"margin: 1px 1px 0 0; background: transparent; border: none; "
    f"border-radius: 3px; }}"
    f"QDoubleSpinBox::down-button, QSpinBox::down-button {{ "
    f"subcontrol-origin: border; subcontrol-position: bottom right; width: 16px; "
    f"margin: 0 1px 1px 0; background: transparent; border: none; "
    f"border-radius: 3px; }}"
    f"QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover, "
    f"QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover "
    f"{{ background: {FILL_HOVER}; }}"
    f"QDoubleSpinBox::up-button:pressed, QSpinBox::up-button:pressed, "
    f"QDoubleSpinBox::down-button:pressed, QSpinBox::down-button:pressed "
    f"{{ background: {FILL_PRESSED}; }}"
    f"QDoubleSpinBox::up-arrow, QSpinBox::up-arrow "
    f"{{ image: url({_ICONS}/spin_up.svg); width: 7px; height: 7px; }}"
    f"QDoubleSpinBox::down-arrow, QSpinBox::down-arrow "
    f"{{ image: url({_ICONS}/spin_down.svg); width: 7px; height: 7px; }}"
)

INPUT_QSS = (
    f"QLineEdit {{ color: {TEXT}; background: {FILL_INPUT}; "
    f"border: 1px solid {BORDER}; border-radius: {RADIUS_INPUT}; "
    f"padding: 1px 4px; font-size: {FONT_SMALL}; }}"
    f"QLineEdit:hover {{ border: 1px solid {TEXT_DIM}; }}"
    f"QLineEdit:disabled {{ color: {TEXT_DISABLED}; "
    f"background: {FILL_DISABLED}; }}"
    f"QComboBox {{ color: {TEXT}; background: {FILL_BTN}; "
    f"border: 1px solid {BORDER}; border-radius: {RADIUS_INPUT}; "
    f"padding: 1px 4px; font-size: {FONT_SMALL}; }}"
    f"QComboBox:hover {{ background: {FILL_HOVER}; }}"
    f"QComboBox:disabled {{ color: {TEXT_DISABLED}; "
    f"background: {FILL_DISABLED}; }}"
    f"QComboBox::drop-down {{ border: none; width: 16px; }}"
    f"QComboBox::down-arrow {{ image: url({_ICONS}/spin_down.svg); "
    f"width: 8px; height: 8px; }}"
    f"QComboBox QAbstractItemView {{ color: {TEXT}; "
    f"background: {SURFACE_RAISED}; border: 1px solid {BORDER}; "
    f"selection-background-color: #2a2a2a; }}"
)

CHECKBOX_QSS = (
    f"QCheckBox {{ color: {TEXT}; font-size: {FONT_SMALL}; spacing: 5px; }}"
    f"QCheckBox:disabled {{ color: {TEXT_DISABLED}; }}"
    f"QCheckBox::indicator {{ width: 12px; height: 12px; "
    f"border: 1px solid {BORDER}; border-radius: 3px; "
    f"background: {FILL_INPUT}; }}"
    f"QCheckBox::indicator:hover {{ border: 1px solid {TEXT_DIM}; "
    f"background: {FILL_HOVER}; }}"
    f"QCheckBox::indicator:checked {{ background: {ACCENT}; "
    f"border: 1px solid {ACCENT_BORDER}; "
    f"image: url({_ICONS}/check.svg); }}"
    f"QCheckBox::indicator:disabled {{ background: {FILL_DISABLED}; }}"
)

# Horizontal sliders: thin rounded groove, accent fill up to the handle,
# round handle that brightens on hover (every control needs hover feedback).
SLIDER_QSS = (
    f"QSlider::groove:horizontal {{ height: 4px; background: {FILL_INPUT}; "
    f"border-radius: 2px; }}"
    f"QSlider::sub-page:horizontal {{ background: {ACCENT}; "
    f"border-radius: 2px; }}"
    f"QSlider::handle:horizontal {{ width: 12px; margin: -5px 0; "
    f"border-radius: 6px; background: rgba(220,220,220,255); "
    f"border: 1px solid {BORDER}; }}"
    f"QSlider::handle:horizontal:hover {{ background: white; "
    f"border: 1px solid {ACCENT_BORDER}; }}"
    f"QSlider::handle:horizontal:pressed {{ background: {ACCENT_BORDER}; }}"
    f"QSlider::sub-page:horizontal:disabled {{ background: {FILL_DISABLED}; }}"
    f"QSlider::handle:horizontal:disabled {{ background: {TEXT_DISABLED}; }}"
)

LABEL_QSS = f"color: {TEXT}; font-size: {FONT_SMALL};"

# EditableLabel: the click-to-edit field used by the Plot Axes + Metadata
# panels. Themed to match the rest of the sidebar — accent-tinted hover on the
# label, the shared input look while editing.
EDITABLE_LABEL_QSS = (
    f"QLabel#editableLabelPart {{ color: {TEXT}; font-size: {FONT_SMALL}; "
    f"border-radius: {RADIUS_INPUT}; padding: 1px 3px; }}"
    f"QLabel#editableLabelPart:hover {{ background-color: {ACCENT_SOFT}; }}"
)

# Section header inside the sidebar panels (e.g. the Name/Scale/Offset/Units
# column titles, metadata subsection captions).
PANEL_HEADER_QSS = (
    f"color: {TEXT_DIM}; font-size: {FONT_SMALL}; font-weight: 600;"
)

# ── Application stylesheet ────────────────────────────────────────────────────
# Single global QSS applied by MainWindow at startup. Everything is built
# from the tokens above so the whole app — docks, dialogs, menus, live
# widgets — shares one look. Widgets with their own setStyleSheet (subwindow
# title bars, toolbars, carets) intentionally override parts of this.
APP_QSS = (
    f"QMdiArea {{ background: {SURFACE_CANVAS}; }}"
    f"QMainWindow {{ background-color: {SURFACE_CANVAS}; }}"
    f"QDockWidget, QDockWidget > QWidget "
    f"{{ background-color: {SURFACE_PANEL}; color: {TEXT}; }}"
    f"QDockWidget::title {{ background-color: {SURFACE_PANEL}; "
    f"color: {TEXT}; padding: 2px; }}"
    # Menus / status bar
    f"QMenuBar {{ background-color: {SURFACE_BAR}; color: {TEXT}; }}"
    f"QMenuBar::item {{ background-color: transparent; color: {TEXT}; "
    f"padding: 2px 8px; }}"
    f"QMenuBar::item:selected {{ background-color: {FILL_HOVER}; "
    f"border-radius: {RADIUS_INPUT}; }}"
    f"QMenu {{ background-color: {SURFACE_RAISED}; color: {TEXT}; "
    f"border: 1px solid {BORDER}; }}"
    f"QMenu::item {{ padding: 3px 18px; }}"
    f"QMenu::item:selected {{ background-color: {ACCENT}; }}"
    f"QMenu::item:disabled {{ color: {TEXT_DISABLED}; }}"
    f"QStatusBar {{ background-color: {SURFACE_BAR}; color: {TEXT}; }}"
    # Dialogs (children default to transparent, so the dialog fill shows
    # through; interactive widgets below keep their own styling)
    f"QDialog, QMessageBox {{ background-color: {SURFACE_PANEL}; "
    f"color: {TEXT}; }}"
    f"QFileDialog {{ background-color: {SURFACE_PANEL}; color: {TEXT}; }}"
    f"QFileDialog QWidget {{ background-color: {SURFACE_PANEL}; "
    f"color: {TEXT}; }}"
    # Buttons — re-emitted at file-dialog specificity so the dark-surface
    # rule above doesn't strip hover feedback there
    + BUTTON_QSS
    + _button_rules("QFileDialog QPushButton")
    # Inputs
    + INPUT_QSS
    + SPIN_QSS
    + CHECKBOX_QSS
    + SLIDER_QSS
    + (
        f"QPlainTextEdit, QTextEdit, QDateEdit, QTimeEdit, QDateTimeEdit "
        f"{{ color: {TEXT}; background-color: {FILL_INPUT}; "
        f"border: 1px solid {BORDER}; border-radius: {RADIUS_INPUT}; }}"
        # Item views (file lists, trees, tables)
        f"QListView, QTreeView, QTableView {{ background-color: #1a1a1a; "
        f"color: {TEXT}; alternate-background-color: #151515; "
        f"selection-background-color: #2a2a2a; selection-color: {TEXT}; }}"
        f"QHeaderView::section {{ background-color: {SURFACE_BAR}; "
        f"color: {TEXT}; border: 0px; padding: 4px; }}"
        # Slim flat scrollbars
        f"QScrollBar:vertical {{ background: transparent; width: 10px; "
        f"margin: 0; }}"
        f"QScrollBar::handle:vertical {{ background: {FILL_BTN}; "
        f"border-radius: 4px; min-height: 24px; }}"
        f"QScrollBar::handle:vertical:hover {{ background: {FILL_HOVER}; }}"
        f"QScrollBar:horizontal {{ background: transparent; height: 10px; "
        f"margin: 0; }}"
        f"QScrollBar::handle:horizontal {{ background: {FILL_BTN}; "
        f"border-radius: 4px; min-width: 24px; }}"
        f"QScrollBar::handle:horizontal:hover {{ background: {FILL_HOVER}; }}"
        f"QScrollBar::add-line, QScrollBar::sub-line "
        f"{{ width: 0px; height: 0px; }}"
        f"QScrollBar::add-page, QScrollBar::sub-page "
        f"{{ background: transparent; }}"
        f"QToolTip {{ color: {TEXT}; background-color: {SURFACE_RAISED}; "
        f"border: 1px solid {BORDER}; }}"
        # Generic text containers stay transparent over dark surfaces
        f"QLabel {{ color: {TEXT}; background-color: transparent; }}"
        # Group boxes: faint rounded frame + dim title, matching the panels.
        # (Previously unstyled, so they fell back to the heavy Fusion border.)
        f"QGroupBox {{ color: {TEXT}; background-color: transparent; "
        f"border: 1px solid {BORDER_FAINT}; border-radius: {RADIUS}; "
        f"margin-top: 8px; padding: 6px 4px 4px 4px; }}"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; "
        f"padding: 0 4px; color: {TEXT_DIM}; font-size: {FONT_SMALL}; "
        f"font-weight: 600; }}"
        # Progress bars (dialogs / long-op feedback) in the accent colour.
        f"QProgressBar {{ background: {FILL_INPUT}; border: 1px solid {BORDER}; "
        f"border-radius: {RADIUS_INPUT}; height: 6px; text-align: center; "
        f"color: {TEXT}; font-size: {FONT_SMALL}; }}"
        f"QProgressBar::chunk {{ background: {ACCENT}; "
        f"border-radius: {RADIUS_INPUT}; }}"
        # Tab widgets (carets, multi-panel dialogs).
        f"QTabWidget::pane {{ border: 1px solid {BORDER_FAINT}; "
        f"border-radius: {RADIUS_INPUT}; }}"
        f"QTabBar::tab {{ background: transparent; color: {TEXT_DIM}; "
        f"padding: 4px 10px; border-radius: {RADIUS_INPUT}; }}"
        f"QTabBar::tab:hover {{ background: {FILL_HOVER}; color: {TEXT}; }}"
        f"QTabBar::tab:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}"
    )
)


# ── Smooth-cornered button ────────────────────────────────────────────────────

def _qcolor(token: str | None) -> QtGui.QColor | None:
    """Parse the rgba()/named-color string tokens above into a QColor."""
    if token is None:
        return None
    token = token.strip()
    if token.startswith(("rgba(", "rgb(")):
        parts = [float(x) for x in
                 token[token.index("(") + 1:token.rindex(")")].split(",")]
        if len(parts) == 3:
            parts.append(255)
        return QtGui.QColor(int(parts[0]), int(parts[1]),
                            int(parts[2]), int(parts[3]))
    return QtGui.QColor(token)


class SmoothButton(QtWidgets.QPushButton):
    """
    Rounded push button painted with an antialiased QPainter.

    Qt's stylesheet engine rasterises ``border-radius`` without antialiasing,
    which leaves stair-stepped corners — clearly visible on 1px light borders
    over dark surfaces. Painting the rounded rect ourselves gives the same
    smooth corners as CaretGroup / RoundedToolBar.

    Colors default to the shared theme tokens; pass overrides for special
    buttons (e.g. the green Commit / red Stop title-bar buttons).
    ``border=None`` draws no outline; ``font_px=None`` keeps the app font.
    """

    def __init__(self, text: str = "", parent=None, *,
                 danger_when_checked: bool = False,
                 fill: str = FILL_BTN,
                 fill_hover: str = FILL_HOVER,
                 fill_pressed: str = FILL_PRESSED,
                 fill_disabled: str = FILL_DISABLED,
                 border: str | None = BORDER,
                 text_color: str = TEXT,
                 text_disabled: str = TEXT_DISABLED,
                 font_px: int | None = 10,
                 margin: int = 0):
        super().__init__(text, parent)
        self._radius = RADIUS_F
        self._margin = float(margin)
        if danger_when_checked:
            checked, checked_hover, checked_border = (
                DANGER, DANGER_HOVER, DANGER_HOVER)
        else:
            checked, checked_hover, checked_border = (
                ACCENT, ACCENT, ACCENT_BORDER)
        self._c = {
            "fill": _qcolor(fill),
            "hover": _qcolor(fill_hover),
            "pressed": _qcolor(fill_pressed),
            "disabled": _qcolor(fill_disabled),
            "border": _qcolor(border),
            "checked": _qcolor(checked),
            "checked_hover": _qcolor(checked_hover),
            "checked_border": _qcolor(checked_border),
            "text": _qcolor(text_color),
            "text_disabled": _qcolor(text_disabled),
        }
        if font_px is not None:
            f = self.font()
            f.setPixelSize(font_px)
            self.setFont(f)
        # repaint on mouse enter/leave so hover feedback works
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)

    def sizeHint(self) -> QtCore.QSize:
        fm = self.fontMetrics()
        w = fm.horizontalAdvance(self.text()) + 14  # 6px padding + 1px border
        h = fm.height() + 8
        if not self.icon().isNull():
            isz = self.iconSize()
            w += isz.width() + (4 if self.text() else 0)
            h = max(h, isz.height() + 8)
        m = int(2 * self._margin)
        return QtCore.QSize(w + m, h + m)

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    def _state_colors(self):
        c = self._c
        if not self.isEnabled():
            return c["disabled"], c["border"], c["text_disabled"]
        if self.isCheckable() and self.isChecked():
            fill = c["checked_hover"] if self.underMouse() else c["checked"]
            return fill, c["checked_border"], c["text"]
        if self.isDown():
            return c["pressed"], c["border"], c["text"]
        if self.underMouse():
            return c["hover"], c["border"], c["text"]
        return c["fill"], c["border"], c["text"]

    def paintEvent(self, event: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        fill, border, text_color = self._state_colors()

        inset = self._margin + 0.5  # half-pixel align for a crisp 1px pen
        rect = QtCore.QRectF(self.rect()).adjusted(inset, inset, -inset, -inset)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, self._radius, self._radius)
        p.fillPath(path, fill)
        if border is not None:
            pen = QtGui.QPen(border)
            pen.setWidthF(1.0)
            pen.setCosmetic(True)
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.drawPath(path)

        fm = self.fontMetrics()
        icon = self.icon()
        if not icon.isNull():
            isz = self.iconSize()
            tw = fm.horizontalAdvance(self.text()) if self.text() else 0
            gap = 4 if self.text() else 0
            x = rect.center().x() - (isz.width() + gap + tw) / 2.0
            iy = rect.center().y() - isz.height() / 2.0
            mode = (QtGui.QIcon.Mode.Normal if self.isEnabled()
                    else QtGui.QIcon.Mode.Disabled)
            icon.paint(p, QtCore.QRect(int(x), int(iy), isz.width(),
                                       isz.height()),
                       QtCore.Qt.AlignmentFlag.AlignCenter, mode)
            if self.text():
                p.setPen(text_color)
                p.drawText(QtCore.QRectF(x + isz.width() + gap, rect.top(),
                                         tw + 2, rect.height()),
                           QtCore.Qt.AlignmentFlag.AlignVCenter
                           | QtCore.Qt.AlignmentFlag.AlignLeft,
                           self.text())
        elif self.text():
            p.setPen(text_color)
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                       self.text())


# ── Widget factories ──────────────────────────────────────────────────────────

def make_label(text: str, parent=None, wrap: bool = True) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text, parent)
    lbl.setStyleSheet(LABEL_QSS)
    lbl.setWordWrap(wrap)
    return lbl


def make_button(text: str, parent=None, enabled: bool = True,
                danger_when_checked: bool = False) -> QtWidgets.QPushButton:
    btn = SmoothButton(text, parent, danger_when_checked=danger_when_checked)
    btn.setEnabled(enabled)
    return btn


def spin_min_width(spin: QtWidgets.QDoubleSpinBox) -> int:
    """Width that fits the widest possible value + suffix (no clipping)."""
    decimals = spin.decimals() if hasattr(spin, "decimals") else 0
    sample = f"-{spin.maximum():.{decimals}f}{spin.suffix()}"
    fm = spin.fontMetrics()
    # text + step buttons + frame padding
    return max(64, fm.horizontalAdvance(sample) + 34)


def make_double_spin(parent, lo: float, hi: float, value: float,
                     decimals: int, suffix: str = ""
                     ) -> QtWidgets.QDoubleSpinBox:
    spin = QtWidgets.QDoubleSpinBox(parent)
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    spin.setSingleStep(10 ** -decimals)
    spin.setValue(value)
    if suffix:
        spin.setSuffix(suffix)
    spin.setStyleSheet(SPIN_QSS)
    # Fixed-but-fitting width: ad-hoc 64/72 px clipped high-decimal values
    spin.setFixedWidth(spin_min_width(spin))
    return spin


def make_slider_row(parent, label_text: str, lo: float, hi: float,
                    value: float, decimals: int = 2, suffix: str = ""):
    """
    Labelled slider + spinbox row with two-way sync (single shared
    implementation for what find_vectors and pyxem each used to build).

    Returns (row_widget, spinbox, slider).
    """
    scale = 10 ** decimals
    row = QtWidgets.QWidget(parent)
    h = QtWidgets.QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)
    lbl = make_label(label_text, row, wrap=False)
    spin = make_double_spin(row, lo, hi, value, decimals, suffix)
    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, row)
    slider.setStyleSheet(SLIDER_QSS)
    slider.setRange(int(lo * scale), int(hi * scale))
    slider.setValue(int(value * scale))

    def _spin_to_slider(v, _s=slider, _sc=scale):
        _s.blockSignals(True)
        _s.setValue(int(v * _sc))
        _s.blockSignals(False)

    def _slider_to_spin(v, _sp=spin, _sc=scale):
        _sp.blockSignals(True)
        _sp.setValue(v / _sc)
        _sp.blockSignals(False)

    spin.valueChanged.connect(_spin_to_slider)
    slider.valueChanged.connect(_slider_to_spin)
    h.addWidget(lbl)
    h.addWidget(slider, 1)
    h.addWidget(spin)
    return row, spin, slider
