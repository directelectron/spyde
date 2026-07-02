"""
_template_action.py — copyable skeletons for new actions. NOT registered
anywhere; a smoke test imports this module so the examples can't rot.

Pick the shape (see spyde/actions/README.md for the decision guide):

1. **TransformAction** — signal + params → a new node in the SAME tree.
2. **RegionAction**   — an interactive ROI → a linked live output plot.
3. **Staged wizard**  — a caret with open/tune/run/commit/close stages, heavy
   compute on a worker, result promoted to a NEW SignalTree.

To wire a TOOLBAR action, add a `spyde/toolbars.yaml` entry:

    My New Action:
      description: One-line tooltip.
      icon: drawing/toolbars/icons/my_icon.svg
      function: spyde.actions.my_module.MyTransformAction   # dotted path
      signal_class: pyxem.signals.Diffraction2D             # isinstance gate
      # signal_types: [spyde_diffraction_vectors_image]     # _signal_type gate
      # exclude_signal_types: [...]                         # blacklist
      # requires_vectors: True          # hide until tree.diffraction_vectors
      plot_dim: [2]                     # 1-D / 2-D plots it applies to
      toolbar_side: bottom
      navigation: False                 # navigator vs signal windows
      # toggle: True / parameters: {...} / subfunctions: {...}

To wire STAGED (wizard) actions, register each stage in
``spyde.actions.registry.STAGED_HANDLERS``:

    "mywiz_open":   "spyde.actions.my_module.mywiz_open",
    "mywiz_tune":   "spyde.actions.my_module.mywiz_tune",
    "mywiz_run":    "spyde.actions.my_module.mywiz_run",
    "mywiz_commit": "spyde.actions.my_module.mywiz_commit",
    "mywiz_close":  "spyde.actions.my_module.mywiz_close",

and add the caret component on the renderer side (see
``electron/src/renderer/src/components/wizardHooks.ts``).
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions.action import RegionAction, TransformAction
from spyde.actions.commit import commit_result_tree
from spyde.actions.context import src_plot_tree as _src_plot_tree
from spyde.actions.lifecycle import wait_for_vectors
from spyde.actions.wizard import WizardController

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TransformAction — signal + params → new node in the SAME tree.
#    The template resolves params (YAML defaults < caret values < kwargs),
#    runs the hyperspy method / free function, adds the node + PlotState.
# ─────────────────────────────────────────────────────────────────────────────

class TemplateTransformAction(TransformAction):
    name = "Template Transform"
    method = "rebin"                    # hyperspy method on the signal…
    # function = my_free_function      # …or function(signal, **kwargs)
    node_name = "Transformed"
    parameters = {
        "scale_x": {"default": 2},
        "scale_y": {"default": 2},
    }

    def build_kwargs(self, signal, scale_x=2, scale_y=2, **_):
        nav = signal.axes_manager.navigation_dimension
        return {"scale": [1] * nav + [int(scale_x), int(scale_y)]}


# ─────────────────────────────────────────────────────────────────────────────
# 2. RegionAction — ROI on the source plot → live linked output plot.
#    Override reduce(); the base builds the output window, the selector, the
#    initial compute, and live param edits (update_live_params ← update_vi).
#    The dispatcher tracks {selector, out_wids, action} in
#    session._action_artifacts so toggling the action off closes everything.
# ─────────────────────────────────────────────────────────────────────────────

class TemplateRegionAction(RegionAction):
    name = "Template Region"
    output_dims = 2

    def reduce(self, signal, selector, indices, **params):
        # Slice the signal by the ROI (see masks.widget_to_mask for detector
        # geometry) and return the reduced array for the output plot.
        return np.zeros((16, 16), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Staged wizard — controller + `fn(session, plot, payload)` stage handlers.
#    THE RULES (each maps to a bug class this framework closed — README §7):
#    * open bumps the generation guard BEFORE spawning a worker; close bumps
#      it FIRST (React StrictMode fires open/close/open synchronously).
#    * a vector-dependent stage self-waits with wait_for_vectors instead of
#      erroring in the find-vectors attach gap.
#    * heavy compute runs via run_on_worker; UI applies happen in on_done
#      (marshalled to the asyncio main thread) — never on the worker.
#    * any bare-figure window the wizard opens calls own_window(wid) so ✕ and
#      teardown reach it; remove() must be idempotent (self._closed).
#    * Commit promotes the live result via commit_result_tree (provenance!).
# ─────────────────────────────────────────────────────────────────────────────

class TemplateWizard(WizardController):
    key = "mywiz"

    def __init__(self, session, tree):
        super().__init__(session, tree)
        self.result = None              # the live result to commit

    def remove(self) -> None:
        if self._closed:
            return
        self._closed = True
        # tear down overlays / extra windows here
        if getattr(self.tree, "_mywiz", None) is self:
            self.tree._mywiz = None

    def commit(self):
        if self.result is None or self.session is None:
            return None
        return commit_result_tree(
            self.session, title="My Result",
            primary=self.result, primary_label="value",
            provenance={"action": "My Wizard", "params": {}},
        )


def mywiz_open(session, plot, payload) -> None:
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        from spyde.backend.ipc import emit_error
        emit_error("My Wizard: no active dataset")
        return
    if getattr(tree, "diffraction_vectors", None) is None:
        # Only for vector-dependent wizards: wait out the attach gap.
        if wait_for_vectors(session, plot,
                            lambda: mywiz_open(session, plot, payload),
                            what="My Wizard", strict=True):
            return
    wiz = getattr(tree, "_mywiz", None)
    if wiz is not None and not wiz._closed:
        return                                       # idempotent re-open
    wiz = TemplateWizard(session, tree)
    gen = wiz.guard()                                # BEFORE the worker

    def _build(result):
        if not wiz.still(gen):
            return                                   # superseded by close/open
        wiz.result = result
        tree._mywiz = wiz
        # open result windows here; wiz.own_window(wid) for bare figures

    wiz.run_on_worker(lambda: np.zeros((8, 8), np.float32),
                      name="mywiz-open", on_done=_build)


def mywiz_close(session, plot, payload=None) -> None:
    src, tree = _src_plot_tree(session, plot)
    wiz = getattr(tree, "_mywiz", None) if tree is not None else None
    if wiz is not None:
        wiz.cancel_inflight()                        # FIRST — kills in-flight open
        wiz.remove()


def mywiz_commit(session, plot, payload) -> None:
    src, tree = _src_plot_tree(session, plot)
    wiz = getattr(tree, "_mywiz", None) if tree is not None else None
    if wiz is None:
        from spyde.backend.ipc import emit_error
        emit_error("My Wizard: nothing to commit")
        return
    wiz.commit()
