"""
vector_orientation_action.py — live Refine caret for vector orientation mapping.

Toolbar action on a diffraction-vectors result tree (gated `requires_vectors`).
A tabbed wizard mirroring the dense OM caret but driven by the sparse-vector
matcher in `vector_orientation.py`:

  Load     — drop CIF(s) → phases; accelerating voltage.
  Library  — angular resolution, min intensity → generate templates.
  Refine   — fits the pattern under the crosshair live; overlays the fitted
             template spots (green) on the measured vectors (red) and shows the
             recovered strain, residual and Friedel asymmetry. Updates on
             crosshair move and on slider changes — the "looks good immediately"
             view. Sliders: strain cap, match tolerance (sink bandwidth).

Batch (whole-field) compute is a later step; this caret is the interactive
single-pattern refine that lets the metric/affine be tuned by eye.
"""
from __future__ import annotations

import threading

import numpy as np
from pyqtgraph import ScatterPlotItem, mkPen

from spyde.signals.diffraction_vectors import COL_KX, COL_KY, COL_INTENSITY

# One caret per toolbar instance.
_VOM_BUILT_TOOLBARS: set = set()


def vector_orientation_mapping(toolbar, action_name: str = "Vector Orientation Mapping",
                               *args, **kwargs):
    """Build the vector-OM Refine caret on the signal plot of a vectors tree."""
    from PySide6 import QtCore as _QC, QtWidgets as _QW
    from spyde.drawing.toolbars.caret_group import CaretGroup, FileDropWidget
    from spyde.qt.style import make_label as _lbl, make_button as _btn
    from spyde.actions import vector_orientation as vo_core

    tid = id(toolbar)
    if tid in _VOM_BUILT_TOOLBARS:
        return
    _VOM_BUILT_TOOLBARS.add(tid)

    plot = toolbar.plot
    signal_tree = getattr(plot, "signal_tree", None)
    vecs = getattr(signal_tree, "diffraction_vectors", None)
    if vecs is None:
        print("Vector OM: no diffraction_vectors on tree.")
        return
    main_window = plot.main_window
    sig_ax = plot.plot_state.current_signal.axes_manager.signal_axes

    # ── Shared mutable state ─────────────────────────────────────────────────
    state = {
        "phases": [],
        "sim": [None],
        "lib": [None],            # TemplateLibrary
        "scatter": [None],        # green fitted-template overlay
        "vec_scatter": [None],    # red measured-vector overlay
        "refit_timer": [None],
        "relay": [None],
        "active": [False],
        "gen": [0],               # refit generation (drop stale threads)
        # seed from the core DEFAULTS so the caret and fit never drift apart
        "params": {k: vo_core.DEFAULTS[k] for k in ("strain_cap", "sink_bw")},
    }
    toolbar._vom_state = state

    caret = CaretGroup(title=action_name, toolbar=toolbar, action_name=action_name)
    toolbar.add_action_widget(action_name, caret, None)
    layout = caret.layout()
    W = 250

    # _on_tab/_schedule are defined further down; the tab-change callback may
    # fire during build (select_step(0)), so guard against forward references.
    def _on_tab_changed(i):
        fn = _callbacks.get("on_tab")
        if fn is not None:
            fn(i)

    _callbacks: dict = {}

    step_bar, stack, _select_step = CaretGroup.make_tab_stack(
        ["1 Load", "2 Library", "3 Refine", "4 Run"], parent=caret, width=W,
        on_tab_changed=_on_tab_changed,
    )

    # ── Page 0: Load ─────────────────────────────────────────────────────────
    p0 = _QW.QWidget(); v0 = _QW.QVBoxLayout(p0)
    v0.setContentsMargins(4, 4, 4, 4); v0.setSpacing(4)
    cif_drop = FileDropWidget(extensions=[".cif"], parent=p0)
    phase_lbl = _lbl("Phases: (none loaded)", p0)
    from spyde.qt.style import make_double_spin as _spin
    voltage_s = _spin(p0, 60, 300, 200, 0, " kV")
    v0.addWidget(_lbl("CIF file(s):", p0)); v0.addWidget(cif_drop)
    v0.addWidget(phase_lbl)
    vr = _QW.QWidget(); hr = _QW.QHBoxLayout(vr)
    hr.setContentsMargins(0, 0, 0, 0); hr.addWidget(_lbl("Voltage:", vr))
    hr.addWidget(voltage_s); v0.addWidget(vr)
    stack.addWidget(p0)

    # ── Page 1: Library ──────────────────────────────────────────────────────
    p1 = _QW.QWidget(); v1 = _QW.QVBoxLayout(p1)
    v1.setContentsMargins(4, 4, 4, 4); v1.setSpacing(4)
    res_s = _spin(p1, 0.3, 10.0, 1.0, 1, "°")
    min_int_s = _spin(p1, 0.0, 1.0, 0.01, 3)
    # max excitation error (relrod tolerance). Low-kV/SEM patterns are sparse —
    # the curved Ewald sphere excites fewer reflections — so a larger value
    # admits more spots. diffsims default 0.1; raise toward 0.02-0.05 Å⁻¹ for SEM.
    exc_s = _spin(p1, 0.005, 0.2, 0.1, 3, " Å⁻¹")
    gen_btn = _btn("Generate Library", p1, enabled=False)
    lib_lbl = _lbl("", p1)
    for lab, w in (("Angle density:", res_s), ("Min intensity:", min_int_s),
                   ("Max excit. err:", exc_s)):
        row = _QW.QWidget(); h = _QW.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0); h.addWidget(_lbl(lab, row)); h.addWidget(w)
        v1.addWidget(row)
    v1.addWidget(gen_btn); v1.addWidget(lib_lbl)
    stack.addWidget(p1)

    # ── Page 2: Refine ───────────────────────────────────────────────────────
    p2 = _QW.QWidget(); v2 = _QW.QVBoxLayout(p2)
    v2.setContentsMargins(4, 4, 4, 4); v2.setSpacing(4)
    from spyde.qt.style import make_slider_row
    cap_row, cap_s, cap_sl = make_slider_row(p2, "Strain cap", 0.5, 10.0, 5.0,
                                             decimals=1, suffix="%")
    sink_row, sink_s, sink_sl = make_slider_row(p2, "Tolerance", 0.5, 8.0, 4.0,
                                                decimals=1, suffix="%")
    refine_lbl = _lbl("Generate a library first.", p2)
    strain_lbl = _lbl("", p2)
    qc_lbl = _lbl("", p2)
    for r in (cap_row, sink_row):
        r.setEnabled(False)
    v2.addWidget(refine_lbl); v2.addWidget(cap_row); v2.addWidget(sink_row)
    v2.addWidget(strain_lbl); v2.addWidget(qc_lbl)
    stack.addWidget(p2)

    # ── Page 3: Run (whole-field batch → orientation + strain maps) ──────────
    p3 = _QW.QWidget(); v3 = _QW.QVBoxLayout(p3)
    v3.setContentsMargins(4, 4, 4, 4); v3.setSpacing(4)
    v3.addWidget(_lbl("Fit orientation + strain over the whole scan.", p3))
    warm_chk = _QW.QCheckBox("Warm-start from neighbours", p3)
    # off by default: independent fitting is faster + more accurate on real data
    # (sped_ag); warm-start can absorb a wrong branch as spurious strain.
    warm_chk.setChecked(False)
    from spyde.qt.style import CHECKBOX_QSS as _chk_qss
    smooth_chk = _QW.QCheckBox("Smooth strain field (edge-preserving)", p3)
    smooth_chk.setChecked(True)
    for c in (warm_chk, smooth_chk):
        c.setStyleSheet(_chk_qss)
    run_btn = _btn("Compute Map", p3, enabled=False)
    run_lbl = _lbl("", p3)
    v3.addWidget(warm_chk); v3.addWidget(smooth_chk)
    v3.addWidget(run_btn); v3.addWidget(run_lbl)
    stack.addWidget(p3)

    layout.addWidget(step_bar); layout.addWidget(stack)
    caret.finalize_layout()
    _select_step(0)

    om_action = toolbar._find_action(action_name)
    if om_action is not None:
        om_action.toggled.connect(
            lambda c: (_callbacks.get("on_toggle") or (lambda _c: None))(c))
        om_action.setChecked(True)
    pos_fn = toolbar.action_widgets.get(action_name, {}).get("position_fn")
    if pos_fn is not None:
        pos_fn()

    # ── Relay (worker → GUI) ─────────────────────────────────────────────────
    class _FitRelay(_QC.QObject):
        fit_ready = _QC.Signal(object, object, object)  # spots, strain3, qc
    relay = _FitRelay(toolbar)
    state["relay"][0] = relay

    def _scene_xy(kx, ky):
        # The rendered disk frame stores the array as [ky(row), kx(col)] and
        # pyqtgraph col-major maps array-axis-0 → scene-x, axis-1 → scene-y, so
        # a vector at (kx, ky) lands at scene-(x=ky, y=kx). Matches the
        # Find-Vectors overlay (_update_scatter), which uses pos=(ky, kx).
        # (Verified empirically against _render_disks_block, audit 2026-06-15.)
        return float(ky), float(kx)

    def _apply_fit(spots, strain3, qc):
        sc = state["scatter"][0]
        if sc is not None:
            sc.setData(spots or [])
        if strain3 is None:
            strain_lbl.setText("No fit (too few vectors).")
            qc_lbl.setText("")
            return
        exx, eyy, exy = strain3
        strain_lbl.setText(
            f"strain  εxx={exx*100:+.2f}%  εyy={eyy*100:+.2f}%  εxy={exy*100:+.2f}%")
        resid, friedel, nmatch = qc
        fa = "—" if (friedel is None or np.isnan(friedel)) else f"{friedel:.4f}"
        qc_lbl.setText(f"resid={resid:.4f}  friedel={fa}  matched={nmatch}")

    relay.fit_ready.connect(_apply_fit)

    # ── Overlay activation ───────────────────────────────────────────────────
    def _activate_overlay():
        if state["scatter"][0] is not None:
            return
        vec_sc = ScatterPlotItem(size=9, pen=mkPen("r", width=1.5), brush=None)
        vec_sc.setZValue(9)
        plot.addItem(vec_sc)
        state["vec_scatter"][0] = vec_sc
        sc = ScatterPlotItem(size=12, symbol="o",
                             pen=mkPen("g", width=1.8), brush=None)
        sc.setZValue(11)
        plot.addItem(sc)
        state["scatter"][0] = sc

        timer = _QC.QTimer()
        timer.setInterval(60)
        timer.setSingleShot(True)
        timer.timeout.connect(_do_refit)
        state["refit_timer"][0] = timer

        nav_sel = getattr(plot, "parent_selector", None)
        if nav_sel is None:
            nav_sel = getattr(getattr(plot, "plot_window", None),
                              "parent_selector", None)
        if nav_sel is not None and hasattr(nav_sel, "roi"):
            nav_sel.roi.sigRegionChanged.connect(_schedule)
            nav_sel.roi.sigRegionChangeFinished.connect(_schedule)

    def _current_nav(plot):
        from spyde.actions.pyxem import _get_current_nav_indices
        return _get_current_nav_indices(plot)

    def _do_refit():
        lib = state["lib"][0]
        if lib is None or not state["active"][0]:
            return
        # nav indices → (iy, ix); CrosshairSelector returns (col=x, row=y)
        nav = _current_nav(plot)
        ix, iy = (int(nav[0]), int(nav[1])) if len(nav) >= 2 else (0, 0)
        try:
            rows = vecs.at(iy, ix)
        except Exception:
            rows = np.zeros((0, 6), np.float32)
        # draw measured vectors immediately (red)
        vsc = state["vec_scatter"][0]
        if vsc is not None and len(rows):
            vsc.setData([{"pos": _scene_xy(r[COL_KX], r[COL_KY])} for r in rows])
        elif vsc is not None:
            vsc.setData([])

        state["params"]["strain_cap"] = cap_s.value() / 100.0
        state["params"]["sink_bw"] = sink_s.value() / 100.0
        state["gen"][0] += 1
        my_gen = state["gen"][0]
        mxy = rows[:, [COL_KX, COL_KY]].astype(np.float64) if len(rows) else None
        mI = rows[:, COL_INTENSITY].astype(np.float64) if len(rows) else None
        params = dict(state["params"])

        def _run():
            from spyde.actions.vector_orientation import fit_pattern
            if my_gen != state["gen"][0]:
                return
            if mxy is None or len(mxy) < 4:
                relay.fit_ready.emit([], None, None)
                return
            try:
                fit = fit_pattern(mxy, mI, lib, params)
            except Exception as e:
                print(f"Vector OM refit failed: {e}")
                return
            if my_gen != state["gen"][0]:
                return
            if fit is None:
                relay.fit_ready.emit([], None, None)
                return
            # project the fitted template spots for overlay
            from spyde.actions.vector_orientation import project_spots
            g = lib.spots_xy[fit.template_idx].astype(np.float64)
            pose = np.array([fit.theta, *fit.affine.ravel(),
                             *fit.translation], dtype=np.float64)
            proj = project_spots(pose, g)
            spots = [{"pos": _scene_xy(p[0], p[1])} for p in proj]
            strain3 = (float(fit.strain[0, 0]), float(fit.strain[1, 1]),
                       float(fit.strain[0, 1]))
            qc = (fit.residual, fit.friedel_asym, fit.n_matched)
            relay.fit_ready.emit(spots, strain3, qc)

        threading.Thread(target=_run, daemon=True).start()

    def _schedule(*_):
        t = state["refit_timer"][0]
        if t is not None and state["active"][0]:
            t.start()

    # ── Callbacks ────────────────────────────────────────────────────────────
    def _on_cif_loaded(files):
        from orix.crystal_map import Phase
        state["phases"].clear()
        for path in files:
            try:
                state["phases"].append(Phase.from_cif(path))
            except Exception as e:
                print(f"Failed to load CIF {path}: {e}")
        if state["phases"]:
            phase_lbl.setText("Phases: " + ", ".join(p.name for p in state["phases"]))
            gen_btn.setEnabled(True)
        else:
            phase_lbl.setText("Phases: (none loaded)")
            gen_btn.setEnabled(False)

    cif_drop.filesChanged.connect(_on_cif_loaded)

    def _compute_reciprocal_radius():
        half = [ax.scale * ax.size / 2.0 for ax in sig_ax]
        return min(half)

    def _on_generate():
        if not state["phases"]:
            return
        gen_btn.setEnabled(False)
        lib_lbl.setText("Generating library…")
        kv = voltage_s.value(); res = res_s.value()
        min_int = min_int_s.value(); recip = _compute_reciprocal_radius()
        exc = exc_s.value()

        class _GenRelay(_QC.QObject):
            done = _QC.Signal(bool, str)
        gr = _GenRelay(toolbar)
        state["_gen_relay"] = gr

        def _on_done(ok, msg):
            lib_lbl.setText(msg)
            gen_btn.setEnabled(True)
            if ok:
                gen_btn.setText("Regenerate")
                for r in (cap_row, sink_row):
                    r.setEnabled(True)
                refine_lbl.setText("Move the crosshair to refine each pattern.")
                run_btn.setEnabled(True)
                _activate_overlay()
                state["active"][0] = True
                _select_step(2)
                _schedule()

        gr.done.connect(_on_done)

        def _gen():
            try:
                from spyde.actions.orientation_compute import (
                    generate_library_from_phases)
                from spyde.actions.vector_orientation import build_template_library
                sim = generate_library_from_phases(
                    state["phases"], accelerating_voltage=kv, resolution=res,
                    minimum_intensity=min_int, reciprocal_radius=recip,
                    max_excitation_error=exc)
                state["sim"][0] = sim
                lib = build_template_library(
                    sim, plot.plot_state.current_signal, r_max=recip)
                state["lib"][0] = lib
                gr.done.emit(True, f"Library ready ({len(lib.spots_xy)} templates)")
            except Exception as e:
                import traceback; traceback.print_exc()
                gr.done.emit(False, f"Failed: {e}")

        threading.Thread(target=_gen, daemon=True).start()

    gen_btn.clicked.connect(_on_generate)

    for w in (cap_s, cap_sl, sink_s, sink_sl):
        w.valueChanged.connect(lambda *_: _schedule())

    # ── Whole-field batch compute → progressive orientation + strain maps ────
    # Mirrors the dense OM Run flow (pyxem.orientation_mapping): a chunked,
    # cluster-parallel compute writes a 12-channel live buffer (IPF X|Y|Z +
    # strain εxx,εyy,εxy) that the GUI polls and paints into PlotWindows as the
    # scan fills in. 12 float32 channels per nav position.
    _PREVIEW_CH = 12

    class _RunRelay(_QC.QObject):
        done = _QC.Signal(object)      # VectorOrientationResult or None
        failed = _QC.Signal(str)
        pct = _QC.Signal(float)        # coarse progress %
    run_relay = _RunRelay(toolbar)
    state["run_relay"] = run_relay
    relay_pct = run_relay.pct
    run_relay.pct.connect(
        lambda p: run_lbl.setText(f"Computing… {p:.0f}%"))

    def _strain_levels():
        return (-cap_s.value() / 100.0, cap_s.value() / 100.0)

    def _on_run_done(result):
        poll = state.get("_run_poll")
        if poll is not None:
            poll.stop()
        state["om_result"] = result
        if result is None:
            run_lbl.setText("Stopped.")
            run_btn.setEnabled(True)
            return
        signal_tree.vector_orientation = result
        try:
            # Final authoritative paint of the IPF (z) + strain panels.
            nav_plot = state.get("_run_nav_plot")
            if nav_plot is not None:
                nav_plot.image_item.setImage(
                    result.ipf_color_map("z"), autoLevels=False, levels=(0, 255))
            strain = (result.smoothed_strain() if smooth_chk.isChecked()
                      else result.strain)
            for di, comp in enumerate(("exx", "eyy", "exy")):
                p = state.get(f"_run_strain_{comp}")
                if p is not None:
                    m = strain[..., di]
                    lim = float(np.nanmax(np.abs(m))) or 1.0
                    p.image_item.setImage(m, autoLevels=False,
                                          levels=(-lim, lim))
            run_lbl.setText("✓ Map computed.")
        except Exception as e:
            import traceback; traceback.print_exc()
            run_lbl.setText(f"Display error: {e}")
        run_btn.setEnabled(True)

    def _on_run_failed(msg):
        poll = state.get("_run_poll")
        if poll is not None:
            poll.stop()
        run_lbl.setText(msg)
        run_btn.setEnabled(True)

    run_relay.done.connect(_on_run_done)
    run_relay.failed.connect(_on_run_failed)

    def _make_map_window(title, levels):
        pw = main_window.add_plot_window(is_navigator=False,
                                         signal_tree=signal_tree)
        pw.owner_plot_window = plot.plot_window
        pw.setWindowTitle(title)
        main_window._auto_position_near_owner(pw)
        mp = pw.add_new_plot()
        if mp.image_item not in mp.items:
            mp.addItem(mp.image_item)
        ny, nx = vecs.nav_shape
        if levels == "rgb":
            mp.image_item.setImage(np.zeros((ny, nx, 3), np.uint8),
                                   autoLevels=False, levels=(0, 255))
        else:
            mp.image_item.setImage(np.full((ny, nx), np.nan, np.float32),
                                   autoLevels=False, levels=levels)
        return pw, mp

    def _on_run():
        from spyde.drawing.update_functions import (
            ensure_live_buffer, read_live_buffer)
        lib = state["lib"][0]
        if lib is None:
            return
        run_btn.setEnabled(False)
        run_lbl.setText("Computing… 0%")
        params = dict(state["params"])
        params["strain_cap"] = cap_s.value() / 100.0
        params["sink_bw"] = sink_s.value() / 100.0
        warm = warm_chk.isChecked()
        state["run_stopped"] = [False]
        ny, nx = vecs.nav_shape
        t_cur = None
        if vecs.n_time > 0:
            try:
                t_cur = int(signal_tree.root.axes_manager.indices[0])
            except Exception:
                t_cur = None

        # Live 12-channel preview buffer.
        shm_name = f"spyde_vom_{id(plot)}"
        shm = ensure_live_buffer((ny, nx, _PREVIEW_CH), shm_name)
        state["_run_shm"] = shm

        # Result windows: an orientation IPF-Z map + three strain panels, all
        # painted progressively as chunks land.
        sl = _strain_levels()
        nav_pw, nav_plot = _make_map_window("Orientation (IPF-Z)", "rgb")
        state["_run_nav_plot"] = nav_plot
        for comp, ttl in (("exx", "Strain εxx"), ("eyy", "Strain εyy"),
                          ("exy", "Strain εxy")):
            _pw, _mp = _make_map_window(ttl, sl)
            state[f"_run_strain_{comp}"] = _mp

        # Poll the shm buffer and paint partial results.
        old = state.get("_run_poll")
        if old is not None:
            old.stop(); old.deleteLater()
        poll = _QC.QTimer(toolbar)
        poll.setInterval(150)
        state["_run_poll"] = poll

        def _poll():
            # Paint the partial maps from the shm buffer. The progress *label* is
            # owned by _prog (driven by the compute's own progress callback), so
            # the % advances smoothly even before the first live preview lands.
            arr = read_live_buffer((ny, nx, _PREVIEW_CH), shm_name)
            finite = np.isfinite(arr[..., 0])
            if not finite.any():
                return
            rgb = np.clip(np.nan_to_num(arr[..., 6:9]), 0, 255).astype(np.uint8)
            if state.get("_run_nav_plot") is not None:
                state["_run_nav_plot"].image_item.setImage(
                    rgb, autoLevels=False, levels=(0, 255))
            for di, comp in enumerate(("exx", "eyy", "exy")):
                p = state.get(f"_run_strain_{comp}")
                if p is not None:
                    p.image_item.setImage(arr[..., 9 + di], autoLevels=False,
                                          levels=sl)

        poll.timeout.connect(_poll)
        poll.start()

        stop = state["run_stopped"]

        def _prog(done, total):
            if total:
                run_lbl.setText(f"Computing… {100.0 * done / total:.0f}%")

        from spyde.actions.vector_orientation_gpu import (
            select_device, torch_available, gpu_available,
            gpu_unavailable_reason)
        dev = select_device()
        use_torch = torch_available() and dev is not None
        dev_name = dev.type if dev is not None else "none"
        print(f"[vector-OM] Run clicked — torch path={use_torch} device={dev_name} "
              f"accelerated={gpu_available()} "
              f"({vecs.nav_shape[0]}x{vecs.nav_shape[1]} patterns, "
              f"{len(lib.spots_xy)} templates)", flush=True)
        if use_torch:
            # Batched torch fit on the best device (CUDA → Apple-MPS → CPU).
            # backward() must run on the MAIN thread under CUDA on Windows
            # (off-thread segfaults), so we run inline and pump the Qt event
            # loop after each anneal stage (on_yield) — the fit is seconds, and
            # the poll timer paints the live shm preview meanwhile.
            from spyde.actions.vector_orientation_gpu import (
                compute_vector_orientation_gpu)
            if dev.type == "cpu":
                run_lbl.setText("No GPU — running on CPU (slower)…")

            def _yield():
                _QW.QApplication.processEvents(
                    _QC.QEventLoop.ExcludeUserInputEvents, 5)

            try:
                result = compute_vector_orientation_gpu(
                    vecs, lib, params, t=t_cur, progress=_prog,
                    stopped_flag=stop, shm_name=shm_name, on_yield=_yield)
                run_relay.done.emit(result)
            except Exception as e:
                import traceback; traceback.print_exc()
                run_relay.failed.emit(f"Failed: {e}")
            return

        # torch entirely absent → last-resort serial scipy fit (slow).
        print(f"[vector-OM] torch unavailable: {gpu_unavailable_reason()} "
              f"— falling back to the SLOW serial CPU fit.", flush=True)
        run_lbl.setText("No torch — using slow serial CPU fit…")

        # No GPU → fall back to the serial CPU matcher on a worker thread
        # (numpy/scipy, thread-safe). It paints only at the end.
        def _run():
            def _prog_t(done, total):
                if total:
                    relay_pct.emit(100.0 * done / total)
            try:
                from spyde.actions.vector_orientation import (
                    compute_vector_orientation)
                result = compute_vector_orientation(
                    vecs, lib, params, t=t_cur, warm_start=warm,
                    progress=_prog_t, stopped_flag=stop)
                run_relay.done.emit(result)
            except Exception as e:
                import traceback; traceback.print_exc()
                run_relay.failed.emit(f"Failed: {e}")

        threading.Thread(target=_run, daemon=True).start()

    run_btn.clicked.connect(_on_run)

    def _on_tab(i):
        on_refine = (i == 2)
        for it in (state["scatter"][0], state["vec_scatter"][0]):
            if it is not None:
                it.setVisible(on_refine)
        if on_refine and state["lib"][0] is not None:
            state["active"][0] = True
            _schedule()
        elif not on_refine:
            state["active"][0] = False

    _callbacks["on_tab"] = _on_tab

    def _on_action_toggled(checked):
        for it in (state["scatter"][0], state["vec_scatter"][0]):
            if it is not None:
                it.setVisible(bool(checked))
        state["active"][0] = bool(checked) and state["lib"][0] is not None
        if state["active"][0]:
            _schedule()

    _callbacks["on_toggle"] = _on_action_toggled
