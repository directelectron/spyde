"""
test_boundary.py — the shell must not drag the science stack in behind it.

This is the constraint the whole split is built on: de-groundcrew and
de-autopilot are live, in-memory applications, and they install `de-shell`. If
importing it also pulled in hyperspy, dask, rosettasciio or pyxem, those apps
would inherit a multi-second import cost and a large dependency tree they have
no use for — and the boundary would have quietly stopped meaning anything.

Nothing else enforces this. A `from spyde...` or a `import dask` added to a
de_shell module is invisible to every other test in the repo, because SpyDE's
own suite has all of it imported already.

Run in a SUBPROCESS: this test is meaningless in-process once pytest has
collected SpyDE's suite, since those modules are in sys.modules regardless.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

#: Import these and nothing heavy may appear.
SHELL_MODULES = [
    "de_shell",
    "de_shell.ipc",
    "de_shell.log_stream",
    "de_shell.process_guard",
    "de_shell.debug_flags",
    "de_shell.compute",
    "de_shell.session",
]

#: Absent from sys.modules after the imports above.
FORBIDDEN = [
    "hyperspy", "dask", "distributed", "rsciio", "pyxem",
    "orix", "diffsims", "kikuchipy", "exspy", "spyde",
]


def _probe(source: str) -> dict:
    """Run `source` in a clean interpreter and parse its single JSON line."""
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestShellStaysThin:
    def test_importing_the_shell_pulls_in_no_science_stack(self):
        result = _probe(f"""
            import json, sys
            for m in {SHELL_MODULES!r}:
                __import__(m)
            leaked = sorted(
                f for f in {FORBIDDEN!r}
                if f in sys.modules or any(
                    k == f or k.startswith(f + ".") for k in sys.modules)
            )
            print(json.dumps({{"leaked": leaked}}))
        """)
        assert result["leaked"] == [], (
            "de_shell pulled in " + ", ".join(result["leaked"]) +
            " — see ARCHITECTURE-SPLIT.md; the shell must stay installable "
            "without the science stack."
        )

    def test_no_de_shell_module_imports_the_app(self):
        # Belt-and-braces on the direction of the dependency: the shell may be
        # imported BY an app, never the other way round. A lazy `import spyde`
        # inside a function would pass the sys.modules probe above, so grep the
        # source too.
        result = _probe("""
            import json, pathlib, re
            import de_shell
            root = pathlib.Path(de_shell.__file__).parent
            bad = []
            for p in sorted(root.rglob("*.py")):
                for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                    if re.search(r"^\\s*(from|import)\\s+spyde\\b", line):
                        bad.append(f"{p.name}:{i}: {line.strip()}")
            print(json.dumps({"bad": bad}))
        """)
        assert result["bad"] == [], "shell imports the app:\n" + "\n".join(result["bad"])


class TestComputeIsDaskFree:
    def test_thread_compute_works_with_no_dask_installed(self):
        # ThreadCompute exists precisely so an app can submit work without a
        # cluster. If it ever starts importing dask at module scope, the two
        # live apps stop being installable.
        result = _probe("""
            import json, sys
            from de_shell.compute import ThreadCompute, SyncFuture
            from concurrent.futures import ThreadPoolExecutor

            c = ThreadCompute(ThreadPoolExecutor(max_workers=2))
            doubled = c.submit(lambda x: x * 2, 21).result(timeout=30)
            nav = c.submit_nav_read(lambda: "frame").result(timeout=30)
            c.shutdown_nav_pool()

            seen = []
            SyncFuture("done").add_done_callback(lambda f: seen.append(f.result()))

            print(json.dumps({
                "doubled": doubled,
                "nav": nav,
                "distributed": c.is_distributed,
                "sync_cb": seen,
                "dask_imported": "dask" in sys.modules,
            }))
        """)
        assert result == {
            "doubled": 42, "nav": "frame", "distributed": False,
            "sync_cb": ["done"], "dask_imported": False,
        }

    def test_submit_without_an_executor_fails_loudly(self):
        from de_shell.compute import ThreadCompute
        with pytest.raises(RuntimeError, match="no executor"):
            ThreadCompute().submit(lambda: None)
