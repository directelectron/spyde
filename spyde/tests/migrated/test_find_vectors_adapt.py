"""Find Vectors adapt: teach the neural detector on the user's marks.

The generic pieces (``spyde.teach``: the marks, the debounced fit, the taught
model store) are tested on their own; the detector fine-tune
(``spyde.models.adapt``) on the CPU; the wiring (``find_vectors_adapt``) through
a real Session.

Everything runs on the CPU: torch-CUDA can segfault under pytest on Windows,
and these test the wiring and the maths, not the GPU.
"""
from __future__ import annotations

import os
import threading
import time
import types

import numpy as np
import pytest

from spyde import teach
from spyde.tests.migrated._async import quiesce, wait_until, why_busy
from spyde.tests.migrated.conftest import close_session, make_session


@pytest.fixture
def cpu_models(monkeypatch, tmp_path):
    """Models load on the CPU, the model caches start empty, and the user's model
    directory is a temporary one (nothing is written to the real home)."""
    import torch
    from spyde.models import infer, registry
    monkeypatch.setattr(infer, "_default_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(registry, "_MODEL_CACHE", {})
    monkeypatch.setattr(registry, "_CPU_MODEL_CACHE", {})
    monkeypatch.setattr(registry, "user_models_dir", lambda: str(tmp_path))
    import spyde.torch_device
    monkeypatch.setattr(spyde.torch_device, "warmup_autograd", lambda: None)
    registry.reload_manifest()
    yield tmp_path
    registry.reload_manifest()


def _disk_pattern(shape=(96, 96), centres=((30, 30), (30, 66), (66, 30), (66, 66)), radius=5.0,
                  junk=((48, 14),), seed=0):
    """A small lattice of disks plus a hot-pixel cluster (the kind of junk a user
    double-clicks away), Poisson noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    image = np.full(shape, 2.0)
    for cy, cx in centres:
        image += 60.0 * (np.hypot(yy - cy, xx - cx) <= radius)
    for jy, jx in junk:
        image[jy:jy + 2, jx:jx + 2] += 400.0
    return rng.poisson(image).astype(np.float32)


# ── the marks store ───────────────────────────────────────────────────────────

class TestPointMarks:
    def test_add_remove_and_count(self):
        marks = teach.PointMarks()
        marks.add((0, 1), 10, 20, 1)
        marks.add((0, 1), 30, 40, 0)
        marks.add([2, 3], 5, 5, 0)
        assert len(marks) == 3 and marks.count(0) == 2 and marks.count(1) == 1
        assert sorted(marks.fields()) == [(0, 1), (2, 3)]
        assert marks.at((0, 1)).shape == (2, 3)
        assert not marks.remove_near((0, 1), 0, 0, radius=5)       # nothing that close
        assert marks.remove_near((0, 1), 31, 41, radius=5)
        assert marks.at((0, 1)).tolist() == [[10.0, 20.0, 1.0]]
        assert marks.remove_near((2, 3), 5, 5, radius=1)
        assert (2, 3) not in marks.fields()

    def test_round_trip(self):
        marks = teach.PointMarks()
        marks.add((4, 5), 1.5, 2.5, 1)
        marks.add((4, 5), 3.0, 4.0, 0)
        again = teach.PointMarks.from_dict(marks.to_dict())
        assert again.to_dict() == marks.to_dict()
        assert again.copy() is not again and len(again.copy()) == 2


# ── the debounced fit ─────────────────────────────────────────────────────────

class _Session:
    """No ``_dispatch_to_main``: run_on_worker runs inline, so the fit runs on
    the timer thread and the test only has to wait for it."""


class TestDebouncedFit:
    def _fitter(self, fit=None, **kw):
        done, started = [], []
        counter = {"snap": 0}

        def snapshot():
            counter["snap"] += 1
            return counter["snap"]

        fitter = teach.DebouncedFit(_Session(), snapshot=snapshot, fit=fit or (lambda s: s * 10),
                                    on_done=done.append, on_start=lambda: started.append(1),
                                    delay=0.15, **kw)
        return fitter, done, started

    def test_requests_in_quick_succession_fit_once(self):
        fitter, done, started = self._fitter()
        for _ in range(5):
            fitter.request()
            time.sleep(0.02)
        assert wait_until(lambda: done, 5)
        time.sleep(0.4)
        assert done == [10] and len(started) == 1

    def test_cancel_drops_the_result(self):
        release = threading.Event()

        def slow(snapshot):
            release.wait(5)
            return snapshot

        fitter, done, _ = self._fitter(fit=slow)
        thread = threading.Thread(target=fitter.run_now)
        thread.start()
        assert wait_until(lambda: fitter.busy, 5)
        fitter.cancel()
        release.set()
        thread.join(5)
        assert done == [] and not fitter.busy

    def test_a_request_during_a_fit_runs_once_more_after_it(self):
        release = threading.Event()
        calls = []

        def slow(snapshot):
            calls.append(snapshot)
            if len(calls) == 1:
                release.wait(5)
            return snapshot

        fitter, done, _ = self._fitter(fit=slow)
        thread = threading.Thread(target=fitter.run_now)
        thread.start()
        assert wait_until(lambda: fitter.busy, 5)
        fitter.run_now()                         # arrives mid-fit: queued, not concurrent
        assert len(calls) == 1
        release.set()
        thread.join(5)
        assert calls == [1, 2] and done == [1, 2]


# ── the taught-model store and the registry ──────────────────────────────────

class TestUserModelStore:
    def _save(self, directory, scope="file-a", payload=b"weights-1"):
        def write(path):
            with open(path, "wb") as handle:
                handle.write(payload)
        return teach.save_user_model(str(directory), kind="spotunet", scope=scope,
                                     write_weights=write, entry={"label": "taught"})

    def test_save_replaces_the_previous_model_of_a_scope(self, tmp_path):
        first = self._save(tmp_path, payload=b"one")
        second = self._save(tmp_path, payload=b"two")
        other = self._save(tmp_path, scope="file-b", payload=b"three")
        assert first["id"] != second["id"]           # content-addressed: a refit is a new id
        ids = [e["id"] for e in teach.user_models(str(tmp_path))]
        assert ids == [second["id"], other["id"]]
        assert not os.path.exists(teach.user_model_path(str(tmp_path), first))
        assert os.path.exists(teach.user_model_path(str(tmp_path), second))
        assert teach.remove_user_model(str(tmp_path), second["id"])
        assert [e["id"] for e in teach.user_models(str(tmp_path))] == [other["id"]]
        assert not teach.remove_user_model(str(tmp_path), "nope")

    def test_a_broken_manifest_is_ignored(self, tmp_path):
        (tmp_path / teach.USER_MANIFEST).write_text("{not json")
        assert teach.user_models(str(tmp_path)) == []

    def test_taught_model_is_listed_for_its_scope_only_and_never_default(self, cpu_models):
        import torch
        from spyde import models
        from spyde.models import registry
        base, _ = models.get_model(None)
        default = registry.default_model_id()
        entry = teach.save_user_model(
            str(cpu_models), kind=registry.TAUGHT_KIND, scope="file-a",
            write_weights=lambda p: torch.save({"state_dict": base.state_dict(),
                                                "base": 8, "levels": 3, "in_ch": 1}, p),
            entry={"label": "taught", "arch": registry._entry(default).get("arch"),
                   "parent": {"id": default}})
        registry.reload_manifest()
        assert registry.default_model_id() == default
        listed = [m["id"] for m in registry.available_models(scope="file-a")["models"]]
        assert entry["id"] in listed
        assert entry["id"] not in [m["id"] for m in registry.available_models(scope="file-b")["models"]]
        assert entry["id"] not in [m["id"] for m in registry.available_models()["models"]]
        model, device = models.get_model(entry["id"])        # loads, sha256 verified
        assert device.type == "cpu"
        assert torch.equal(model.head_hm.weight, base.head_hm.weight.cpu())

    def test_a_remote_refresh_does_not_drop_taught_models(self, cpu_models, monkeypatch):
        from spyde.models import registry
        entry = self._save(cpu_models)
        (cpu_models / registry.REMOTE_REGISTRY_FILE).write_text('{"default": null, "models": []}')
        registry.reload_manifest()
        assert entry["id"] in [m["id"] for m in registry.list_models()]


# ── the detector fine-tune ───────────────────────────────────────────────────

PARAMS = dict(spot_radius=5, min_distance=3, bg_sigma=12.0, threshold=0.3)


class TestAdapt:
    def test_marks_become_targets(self, cpu_models):
        from spyde import models
        from spyde.models.adapt import DISK, NOT_DISK, Working, mark_targets
        base, device = models.get_model(None)
        frame = _disk_pattern()
        working = Working([frame], PARAMS, int(base.levels))
        marks = [np.array([[48.5, 14.5, NOT_DISK], [30.0, 30.0, DISK]])]
        labels = mark_targets(base, working, marks, 5.0, device)
        f = working.factor
        target, weight = labels["target"][0, 0], labels["weight"][0, 0]
        jy, jx = int(round(48.5 * f)), int(round(14.5 * f))
        assert float(weight[jy, jx]) > 0 and float(target[jy, jx]) == 0.0
        # the disk mark snapped to a pixel within reach and became a positive
        ys, xs = np.nonzero(target.numpy() == 1.0)
        assert len(ys) == 1 and np.hypot(ys[0] - 30 * f, xs[0] - 30 * f) <= 2.5 * f + 1
        # every mark weighs the same: two marks, total weight 2
        assert abs(float(weight.sum()) - 2.0) < 1e-4

    def test_not_a_disk_marks_remove_the_junk_and_keep_the_disks(self, cpu_models):
        import torch
        from spyde import models
        from spyde.models.adapt import NOT_DISK, Working, adapt
        base, device = models.get_model(None)
        frames = [_disk_pattern(seed=s) for s in range(2)]
        working = Working(frames, PARAMS, int(base.levels))
        with torch.no_grad():
            before = working.peaks(*base(working.x))
        junk = [p for p in before[0] if np.hypot(p[0] - 48.5, p[1] - 14.5) < 5]
        if not junk:
            pytest.skip("the base model does not fire on the synthetic junk")
        marks = [np.array([[junk[0][0], junk[0][1], NOT_DISK]]), np.zeros((0, 3))]
        untouched = {k: v.clone() for k, v in base.state_dict().items()}
        model, report = adapt(base, device, frames, marks, PARAMS, steps=20)
        held_out = Working([_disk_pattern(seed=9)], PARAMS, int(base.levels))
        with torch.no_grad():
            after = held_out.peaks(*model(held_out.x))[0]
        near = lambda p, y, x: np.hypot(p[:, 0] - y, p[:, 1] - x).min() if len(p) else np.inf
        assert near(after, 48.5, 14.5) > 4                       # the junk is gone on a NEW pattern
        for cy, cx in ((30, 30), (30, 66), (66, 30), (66, 66)):   # and every disk is still found
            assert near(after, cy, cx) < 2
        assert report["not_disk_marks"] == 1 and report["disk_marks"] == 0
        assert 0.0 <= report["original_f1"] <= 1.0 and report["original_f1_base"] > 0.7
        assert next(model.parameters()).device.type == "cpu"
        assert all(torch.equal(v, untouched[k]) for k, v in base.state_dict().items())  # base untouched

    def test_a_stopped_fit_raises_cancelled(self, cpu_models):
        from spyde import models
        from spyde.models.adapt import NOT_DISK, Cancelled, adapt
        base, device = models.get_model(None)
        with pytest.raises(Cancelled):
            adapt(base, device, [_disk_pattern()], [np.array([[48.5, 14.5, NOT_DISK]])], PARAMS,
                  stop=[True])

    def test_original_f1_of_the_bundled_model(self, cpu_models):
        from spyde import models
        from spyde.models.adapt import original_f1
        base, device = models.get_model(None)
        assert original_f1(base, device) > 0.75


# ── the wiring ───────────────────────────────────────────────────────────────

def _scan(nav=(3, 4)):
    import hyperspy.api as hs
    data = np.stack([np.stack([_disk_pattern(seed=i * 10 + j) for j in range(nav[1])])
                     for i in range(nav[0])])
    signal = hs.signals.Signal2D(data)
    signal.set_signal_type("electron_diffraction")
    signal.metadata.General.title = "adapt test scan"
    return signal


def _signal_plot(session):
    return next((p for p in session._plots if not p.is_navigator and p.plot_state is not None), None)


class TestWiring:
    def test_toggle_marks_and_revert(self, cpu_models):
        from spyde.actions.find_vectors_adapt import FindVectorsAdapt
        from spyde.models.adapt import DISK, NOT_DISK
        tree = types.SimpleNamespace(source_path=None, root=_scan())
        adapt = FindVectorsAdapt(_Session(), tree)
        adapt.fitter.delay = 60                          # never fires in this test
        peaks_xy = np.array([[14.5, 48.5], [30.0, 30.0]])  # (x, y), as the preview draws
        adapt.toggle((0, 1), 48.0, 15.0, peaks_xy, 5.0)  # on a circle: not a disk, AT the circle
        adapt.toggle((0, 1), 80.0, 80.0, peaks_xy, 5.0)  # empty pattern: a disk here
        rows = adapt.marks.at((0, 1)).tolist()
        assert rows == [[48.5, 14.5, NOT_DISK], [80.0, 80.0, DISK]]
        adapt.toggle((0, 1), 81.0, 79.0, peaks_xy, 5.0)  # on a mark: removed
        assert adapt.marks.at((0, 1)).tolist() == [[48.5, 14.5, NOT_DISK]]
        adapt.fitter.cancel()
        adapt.revert()
        assert len(adapt.marks) == 0 and adapt.model_id is None

    def test_removing_the_last_mark_cancels_the_pending_fit(self, cpu_models):
        from spyde.actions.find_vectors_adapt import FindVectorsAdapt
        tree = types.SimpleNamespace(source_path=None, root=_scan())
        adapt = FindVectorsAdapt(_Session(), tree)
        adapt.fitter.delay = 0.2
        fits = []
        adapt.fitter.fit = lambda snapshot: fits.append(snapshot)
        adapt.toggle((0, 0), 80.0, 80.0, np.zeros((0, 2)), 5.0)      # a mark: fit requested
        adapt.toggle((0, 0), 80.0, 80.0, np.zeros((0, 2)), 5.0)      # and gone again
        time.sleep(0.6)
        assert fits == [] and len(adapt.marks) == 0

    def test_double_click_adapt_compute_provenance_revert(self, cpu_models):
        """Open the caret, mark the pattern under the crosshair, adapt: the taught
        model is saved for this dataset, Compute records it, Revert forgets it."""
        from spyde.actions.find_vectors_action import fv_close, fv_models, fv_open, fv_run
        from spyde.actions.find_vectors_adapt import dataset_scope, fv_adapt, fv_adapt_revert
        from spyde.models import registry
        from spyde.models.adapt import NOT_DISK
        session = make_session()
        messages = []
        import spyde.actions.find_vectors_adapt as module
        original_emit = module.emit
        module.emit = lambda m: (messages.append(m), original_emit(m))
        try:
            session._add_signal(_scan())
            assert quiesce(session), why_busy(session)
            src = _signal_plot(session)
            tree = src.signal_tree
            params = dict(method="neural", **PARAMS, kernel_radius=5)
            fv_open(session, src, params)
            assert wait_until(lambda: getattr(tree, "_fv_adapt", None) is not None, 60)
            adapt = tree._fv_adapt
            adapt.fitter.delay = 60
            preview = tree._fv_preview
            assert wait_until(lambda: (src.last_overlay_value(preview) or {}).get("index") is not None, 60), \
                "the preview never reported the position it drew"
            drawn = src.last_overlay_value(preview)
            index = drawn["index"]
            peaks = np.asarray(drawn["peaks"]["data"]).reshape(-1, 2)
            assert len(peaks), "the preview found nothing to mark"
            adapt.toggle(tuple(index), float(peaks[0][1]), float(peaks[0][0]), peaks, 5.0)
            assert adapt.marks.count(NOT_DISK) == 1
            fv_adapt(session, src, {})
            assert wait_until(lambda: adapt.model_id is not None, 120), "the fit never landed"
            entry = adapt.entry
            assert entry["scope"] == dataset_scope(tree)
            assert entry["parent"]["id"] == registry.default_model_id()
            assert entry["marks"]["fields"][0]["field"] == list(index)
            assert set(entry["calibration"]) == {"spot_radius", "min_distance", "bg_sigma", "threshold"}
            assert 0 <= entry["report"]["original_f1"] <= 1
            assert registry.default_model_id() != entry["id"]
            state = [m for m in messages if m.get("type") == "fv_adapt_state"][-1]
            assert state["model_id"] == entry["id"] and "original_f1" in state

            messages.clear()
            fv_models(session, src, {})          # the dropdown lists it for this dataset

            before = len(session.signal_trees)
            fv_run(session, src, {**params, "model_id": entry["id"]})
            assert wait_until(lambda: len(session.signal_trees) == before + 1, 60)
            result = session.signal_trees[-1]
            record = result._commit_provenance["taught_model"]
            assert record["model_id"] == entry["id"] and record["parent"]["id"] == entry["parent"]["id"]
            assert wait_until(lambda: getattr(result, "diffraction_vectors", None) is not None, 120)

            fv_adapt_revert(session, src, {})
            assert adapt.model_id is None and len(adapt.marks) == 0
            assert entry["id"] not in [m["id"] for m in registry.list_models()]
            assert not os.path.exists(teach.user_model_path(str(cpu_models), entry))
            fv_close(session, src, {})
            assert not adapt.active
        finally:
            module.emit = original_emit
            close_session(session)
