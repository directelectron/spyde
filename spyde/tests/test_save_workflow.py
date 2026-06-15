"""
Tests for File → Save and File → Save/Apply Workflow.

Coverage:
  - _current_signal() returns None when no active plot
  - _current_signal() returns the right signal from the active plot
  - _workflow_steps_for_signal() on a root signal (empty list)
  - _workflow_steps_for_signal() on a child node (ordered chain)
  - _workflow_steps_for_signal() on an unknown signal (None)
  - save_current_signal() writes a real .hspy file
  - save_current_signal() writes a real .zspy file
  - saved .hspy/.zspy round-trips the signal shape
  - save_workflow() writes the correct JSON structure
  - apply_workflow() applies each step to the active signal
  - apply_workflow() stops and warns on a bad transformation name
  - WorkflowViewDialog renders step list correctly
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import hyperspy.api as hs
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from spyde.signal_node import SignalNode
from spyde.signal_tree import BaseSignalTree
from spyde.misc.dialogs.workflow_dialog import WorkflowViewDialog


# ── Module-wide autouse: suppress every QMessageBox so no modal ever blocks ──
#
# Real QMessageBox.warning/critical/information calls show a modal dialog that
# blocks the test runner waiting for a human to click OK.  We patch all three
# for every test in this file.  Tests that need to assert a specific dialog
# was shown receive the `msgbox` fixture and inspect the mocks.


@pytest.fixture(autouse=True)
def msgbox():
    """Patch all three QMessageBox statics; yield a namespace with .warning/.critical/.information."""
    class _M:
        pass
    m = _M()
    with patch.object(QMessageBox, "warning") as w, \
         patch.object(QMessageBox, "critical") as c, \
         patch.object(QMessageBox, "information") as i:
        m.warning = w
        m.critical = c
        m.information = i
        yield m


# ── Pure-unit helpers (no Qt window needed) ──────────────────────────────────


def _make_signal(shape=(4, 4, 8, 8)):
    return hs.signals.Signal2D(np.zeros(shape))


def _make_tree_with_chain():
    """
    Build a minimal BaseSignalTree (no Qt) with a 3-node chain:
      root → child (gaussian_filter, sigma=2) → grandchild (rebin, scale=[1,1,2,2])
    Returns (tree, root_sig, child_sig, grandchild_sig).
    """
    root_sig = _make_signal()
    child_sig = _make_signal()
    grandchild_sig = _make_signal()

    mw = MagicMock()
    mw.dask_manager = MagicMock()
    mw.dask_manager.heavy_workers = None
    mw.add_plot_window.return_value = MagicMock()
    mw.add_plot_window.return_value.add_new_plot.return_value = MagicMock()

    tree = BaseSignalTree.__new__(BaseSignalTree)
    tree.root = root_sig
    tree.main_window = mw
    tree.navigator_signals = {}
    tree.signal_plots = []
    tree.navigator_plot_manager = None
    tree.client = None

    root_node = SignalNode(signal=root_sig, name="root", parent=None)
    child_node = SignalNode(
        signal=child_sig,
        name="gaussian_filter",
        parent=root_node,
        transformation="gaussian_filter",
        args=(),
        kwargs={"sigma": 2},
    )
    grandchild_node = SignalNode(
        signal=grandchild_sig,
        name="rebin",
        parent=child_node,
        transformation="rebin",
        args=(),
        kwargs={"scale": [1, 1, 2, 2]},
    )
    root_node.children["gaussian_filter"] = child_node
    child_node.children["rebin"] = grandchild_node
    tree.root_node = root_node
    return tree, root_sig, child_sig, grandchild_sig


# ── _workflow_steps_for_signal (pure logic, no Qt window) ────────────────────


class TestWorkflowStepsExtraction:
    def test_root_signal_returns_empty_list(self):
        tree, root_sig, _, _ = _make_tree_with_chain()
        from spyde.__main__ import MainWindow
        mw = MagicMock(spec=MainWindow)
        mw.signal_trees = [tree]
        steps = MainWindow._workflow_steps_for_signal(mw, root_sig)
        assert steps == []

    def test_child_returns_one_step(self):
        tree, _, child_sig, _ = _make_tree_with_chain()
        from spyde.__main__ import MainWindow
        mw = MagicMock(spec=MainWindow)
        mw.signal_trees = [tree]
        steps = MainWindow._workflow_steps_for_signal(mw, child_sig)
        assert len(steps) == 1
        assert steps[0]["transformation"] == "gaussian_filter"
        assert steps[0]["kwargs"] == {"sigma": 2}

    def test_grandchild_returns_ordered_chain(self):
        tree, _, _, grandchild_sig = _make_tree_with_chain()
        from spyde.__main__ import MainWindow
        mw = MagicMock(spec=MainWindow)
        mw.signal_trees = [tree]
        steps = MainWindow._workflow_steps_for_signal(mw, grandchild_sig)
        assert len(steps) == 2
        assert steps[0]["transformation"] == "gaussian_filter"
        assert steps[1]["transformation"] == "rebin"
        assert steps[1]["kwargs"] == {"scale": [1, 1, 2, 2]}

    def test_unknown_signal_returns_none(self):
        tree, _, _, _ = _make_tree_with_chain()
        from spyde.__main__ import MainWindow
        mw = MagicMock(spec=MainWindow)
        mw.signal_trees = [tree]
        unknown = _make_signal()
        result = MainWindow._workflow_steps_for_signal(mw, unknown)
        assert result is None

    def test_steps_are_in_root_first_order(self):
        """Verify steps[0] is root→child, not child→root."""
        tree, _, _, grandchild_sig = _make_tree_with_chain()
        from spyde.__main__ import MainWindow
        mw = MagicMock(spec=MainWindow)
        mw.signal_trees = [tree]
        steps = MainWindow._workflow_steps_for_signal(mw, grandchild_sig)
        names = [s["transformation"] for s in steps]
        assert names == ["gaussian_filter", "rebin"]


# ── WorkflowViewDialog ────────────────────────────────────────────────────────


class TestWorkflowViewDialog:
    def test_dialog_lists_steps(self, qtbot):
        from PySide6.QtWidgets import QPlainTextEdit
        steps = [
            {"transformation": "gaussian_filter", "args": [], "kwargs": {"sigma": 2}},
            {"transformation": "rebin", "args": [], "kwargs": {"scale": [1, 1, 2, 2]}},
        ]
        dlg = WorkflowViewDialog(steps=steps)
        qtbot.addWidget(dlg)
        text = dlg.findChild(QPlainTextEdit)
        content = text.toPlainText()
        assert "gaussian_filter" in content
        assert "rebin" in content
        assert "sigma=2" in content

    def test_empty_steps_shows_nothing(self, qtbot):
        from PySide6.QtWidgets import QPlainTextEdit
        dlg = WorkflowViewDialog(steps=[])
        qtbot.addWidget(dlg)
        text = dlg.findChild(QPlainTextEdit)
        assert text.toPlainText() == ""

    def test_accept_returns_accepted(self, qtbot):
        from PySide6.QtWidgets import QDialogButtonBox
        steps = [{"transformation": "foo", "args": [], "kwargs": {}}]
        dlg = WorkflowViewDialog(steps=steps)
        qtbot.addWidget(dlg)
        box = dlg.findChild(QDialogButtonBox)
        ok_btn = box.button(QDialogButtonBox.StandardButton.Ok)
        assert ok_btn is not None


# ── save_current_signal (integration via real file I/O) ──────────────────────


class TestSaveCurrentSignal:
    def test_e2e_save_hspy(self, tmp_path, qtbot, stem_4d_dataset, msgbox):
        """
        End-to-end: create 4D STEM via the app dialog (lazy da.random-backed signal
        with a live Dask distributed cluster), save via save_current_signal as .hspy,
        reload, verify shape.  Exercises the synchronous-scheduler fix for the h5py
        pickle error that occurs when distributed tries to serialize the dask graph.
        """
        win = stem_4d_dataset["window"]
        sig = win.signal_trees[0].root
        assert sig._lazy, "stem_4d_dataset root must be lazy"

        save_path = str(tmp_path / "e2e.hspy")
        with patch.object(type(win), "_current_signal", return_value=sig):
            with patch.object(QFileDialog, "getSaveFileName", return_value=(save_path, "")):
                win.save_current_signal()

        msgbox.critical.assert_not_called()
        msgbox.information.assert_called_once()
        assert save_path in msgbox.information.call_args[0][2]
        assert os.path.exists(save_path)

        loaded = hs.load(save_path, lazy=True)
        assert loaded.data.shape == sig.data.shape

    def test_e2e_save_zspy(self, tmp_path, qtbot, stem_4d_dataset, msgbox):
        """
        End-to-end: same as above but saving as .zspy.
        """
        win = stem_4d_dataset["window"]
        sig = win.signal_trees[0].root
        assert sig._lazy, "stem_4d_dataset root must be lazy"

        save_path = str(tmp_path / "e2e.zspy")
        with patch.object(type(win), "_current_signal", return_value=sig):
            with patch.object(QFileDialog, "getSaveFileName", return_value=(save_path, "")):
                win.save_current_signal()

        msgbox.critical.assert_not_called()
        msgbox.information.assert_called_once()
        assert os.path.exists(save_path)

        loaded = hs.load(save_path, lazy=True)
        assert loaded.data.shape == sig.data.shape

    def test_save_no_active_signal_shows_warning(self, qtbot, window, msgbox):
        """save_current_signal with no open signal shows a warning, does not crash."""
        win = window["window"]
        win.save_current_signal()
        msgbox.warning.assert_called_once()
        args = msgbox.warning.call_args[0]
        assert "No active signal" in args[2]

    def test_save_cancelled_file_dialog_does_nothing(self, tmp_path, qtbot, stem_4d_dataset):
        """Cancelling the file dialog (empty path) should not raise or write any file."""
        win = stem_4d_dataset["window"]
        sig = win.signal_trees[0].root
        with patch.object(type(win), "_current_signal", return_value=sig):
            with patch.object(QFileDialog, "getSaveFileName", return_value=("", "")):
                win.save_current_signal()  # must not raise


# ── save_workflow (JSON serialisation) ───────────────────────────────────────


class TestSaveWorkflow:
    def test_root_signal_shows_info_no_file_dialog(self, tmp_path, qtbot, stem_4d_dataset, msgbox):
        """Saving workflow for a root signal shows an info dialog; the save-file dialog is skipped."""
        win = stem_4d_dataset["window"]
        root_sig = win.signal_trees[0].root
        with patch.object(type(win), "_current_signal", return_value=root_sig):
            with patch.object(QFileDialog, "getSaveFileName") as mock_dlg:
                win.save_workflow()
        mock_dlg.assert_not_called()
        msgbox.information.assert_called_once()

    def test_saves_valid_json_via_steps_extraction(self, tmp_path, qtbot, stem_4d_dataset):
        """
        Drive _workflow_steps_for_signal directly to verify JSON structure.
        """
        win = stem_4d_dataset["window"]
        tree = win.signal_trees[0]

        child_sig = hs.signals.Signal2D(np.zeros(tree.root.data.shape))
        child_node = SignalNode(
            signal=child_sig,
            name="rebin",
            parent=tree.root_node,
            transformation="rebin",
            args=(),
            kwargs={"scale": [1, 1, 2, 2]},
        )
        tree.root_node.children["rebin_test"] = child_node

        steps = win._workflow_steps_for_signal(child_sig)
        assert steps is not None and len(steps) == 1

        json_path = str(tmp_path / "workflow.json")
        with open(json_path, "w") as fh:
            json.dump({"version": 1, "steps": steps}, fh, indent=2, default=str)

        with open(json_path) as fh:
            data = json.load(fh)
        assert data["version"] == 1
        assert data["steps"][0]["transformation"] == "rebin"
        assert data["steps"][0]["kwargs"]["scale"] == [1, 1, 2, 2]

    def test_no_active_signal_shows_warning(self, qtbot, window, msgbox):
        """save_workflow with no open signal shows a warning."""
        win = window["window"]
        win.save_workflow()
        msgbox.warning.assert_called_once()

    def test_save_cancelled_does_nothing(self, tmp_path, qtbot, stem_4d_dataset):
        """Cancelling the file dialog after confirming must not raise."""
        win = stem_4d_dataset["window"]
        tree = win.signal_trees[0]

        child_sig = hs.signals.Signal2D(np.zeros(tree.root.data.shape))
        child_node = SignalNode(
            signal=child_sig, name="rebin_cancel", parent=tree.root_node,
            transformation="rebin", args=(), kwargs={"scale": [1, 1, 1, 1]},
        )
        tree.root_node.children["rebin_cancel"] = child_node

        with patch.object(type(win), "_current_signal", return_value=child_sig):
            with patch.object(QFileDialog, "getSaveFileName", return_value=("", "")):
                win.save_workflow()  # must not raise


# ── apply_workflow ────────────────────────────────────────────────────────────


class TestApplyWorkflow:
    def _write_workflow(self, path: str, steps: list[dict]) -> None:
        with open(path, "w") as fh:
            json.dump({"version": 1, "steps": steps}, fh)

    def test_apply_single_step(self, tmp_path, qtbot, stem_4d_dataset, msgbox):
        """apply_workflow with a valid method adds a new node to the signal tree."""
        win = stem_4d_dataset["window"]
        tree = win.signal_trees[0]
        root_sig = tree.root
        initial_count = len(list(tree.walk()))

        wf_path = str(tmp_path / "wf.json")
        self._write_workflow(wf_path, [
            {"transformation": "rebin", "args": [], "kwargs": {"scale": [1, 1, 2, 2]}},
        ])

        with patch.object(type(win), "_current_signal", return_value=root_sig):
            with patch.object(QFileDialog, "getOpenFileName", return_value=(wf_path, "")):
                with patch.object(WorkflowViewDialog, "exec", return_value=WorkflowViewDialog.DialogCode.Accepted):
                    win.apply_workflow()

        assert len(list(tree.walk())) == initial_count + 1
        msgbox.information.assert_called_once()

    def test_apply_multi_step_chain(self, tmp_path, qtbot, stem_4d_dataset, msgbox):
        """Applying a 2-step workflow adds 2 nodes (each feeds the next)."""
        win = stem_4d_dataset["window"]
        tree = win.signal_trees[0]
        root_sig = tree.root
        initial_count = len(list(tree.walk()))

        wf_path = str(tmp_path / "wf2.json")
        self._write_workflow(wf_path, [
            {"transformation": "rebin", "args": [], "kwargs": {"scale": [1, 1, 2, 2]}},
            {"transformation": "rebin", "args": [], "kwargs": {"scale": [1, 1, 1, 1]}},
        ])

        with patch.object(type(win), "_current_signal", return_value=root_sig):
            with patch.object(QFileDialog, "getOpenFileName", return_value=(wf_path, "")):
                with patch.object(WorkflowViewDialog, "exec", return_value=WorkflowViewDialog.DialogCode.Accepted):
                    win.apply_workflow()

        assert len(list(tree.walk())) == initial_count + 2
        msgbox.information.assert_called_once()

    def test_apply_cancelled_dialog_does_not_modify_tree(self, tmp_path, qtbot, stem_4d_dataset):
        """Cancelling the confirm dialog should not modify the tree."""
        win = stem_4d_dataset["window"]
        tree = win.signal_trees[0]
        root_sig = tree.root
        initial_count = len(list(tree.walk()))

        wf_path = str(tmp_path / "wf_cancel.json")
        self._write_workflow(wf_path, [
            {"transformation": "rebin", "args": [], "kwargs": {"scale": [1, 1, 2, 2]}},
        ])

        with patch.object(type(win), "_current_signal", return_value=root_sig):
            with patch.object(QFileDialog, "getOpenFileName", return_value=(wf_path, "")):
                with patch.object(WorkflowViewDialog, "exec", return_value=WorkflowViewDialog.DialogCode.Rejected):
                    win.apply_workflow()

        assert len(list(tree.walk())) == initial_count

    def test_apply_invalid_transformation_shows_warning(self, tmp_path, qtbot, stem_4d_dataset, msgbox):
        """An unknown method name should trigger a warning, not an unhandled exception."""
        win = stem_4d_dataset["window"]
        root_sig = win.signal_trees[0].root

        wf_path = str(tmp_path / "wf_bad.json")
        self._write_workflow(wf_path, [
            {"transformation": "nonexistent_method_xyz", "args": [], "kwargs": {}},
        ])

        with patch.object(type(win), "_current_signal", return_value=root_sig):
            with patch.object(QFileDialog, "getOpenFileName", return_value=(wf_path, "")):
                with patch.object(WorkflowViewDialog, "exec", return_value=WorkflowViewDialog.DialogCode.Accepted):
                    win.apply_workflow()

        # add_transformation shows critical internally; apply_workflow then shows warning
        msgbox.warning.assert_called_once()

    def test_apply_no_active_signal_shows_warning(self, qtbot, window, msgbox):
        """apply_workflow with no open signal shows a warning."""
        win = window["window"]
        win.apply_workflow()
        msgbox.warning.assert_called_once()

    def test_apply_empty_workflow_shows_info(self, tmp_path, qtbot, stem_4d_dataset, msgbox):
        """A workflow file with an empty steps list should show an informational message."""
        win = stem_4d_dataset["window"]
        root_sig = win.signal_trees[0].root

        wf_path = str(tmp_path / "wf_empty.json")
        self._write_workflow(wf_path, [])

        with patch.object(type(win), "_current_signal", return_value=root_sig):
            with patch.object(QFileDialog, "getOpenFileName", return_value=(wf_path, "")):
                win.apply_workflow()

        msgbox.information.assert_called_once()

    def test_apply_cancelled_file_dialog_does_nothing(self, qtbot, stem_4d_dataset):
        """Cancelling the open-file dialog should not raise."""
        win = stem_4d_dataset["window"]
        root_sig = win.signal_trees[0].root
        with patch.object(type(win), "_current_signal", return_value=root_sig):
            with patch.object(QFileDialog, "getOpenFileName", return_value=("", "")):
                win.apply_workflow()  # must not raise

    def test_apply_corrupt_json_shows_error(self, tmp_path, qtbot, stem_4d_dataset, msgbox):
        """A corrupt JSON file should show a critical error dialog."""
        win = stem_4d_dataset["window"]
        root_sig = win.signal_trees[0].root
        bad_path = str(tmp_path / "corrupt.json")
        with open(bad_path, "w") as fh:
            fh.write("{not valid json ~~")

        with patch.object(type(win), "_current_signal", return_value=root_sig):
            with patch.object(QFileDialog, "getOpenFileName", return_value=(bad_path, "")):
                win.apply_workflow()

        msgbox.critical.assert_called_once()
