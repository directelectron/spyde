"""Find functions that BIND a name (`log`/`logger`) as a parameter or local while
the module defines that same name as a logging.Logger — i.e. a shadow that turns
`log.debug(...)` into an AttributeError when the except fires. Scans live source."""
import ast
import pathlib

NAMES = {"log", "logger"}
ROOT = pathlib.Path("spyde")
SKIP = {"tests", "external"}


def module_logger_names(tree):
    out = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Name) and t.id in NAMES
                        and isinstance(node.value, ast.Call)):
                    fn = node.value.func
                    name = getattr(fn, "attr", None) or getattr(fn, "id", None)
                    if name == "getLogger":
                        out.add(t.id)
    return out


def bound_names_in_func(fn):
    bound = set()
    for a in (fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs):
        bound.add(a.arg)
    if fn.args.vararg:
        bound.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        bound.add(fn.args.kwarg.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    bound.add(t.id)
        elif isinstance(node, (ast.For, ast.comprehension)):
            tgt = getattr(node, "target", None)
            if isinstance(tgt, ast.Name):
                bound.add(tgt.id)
    return bound


def calls_logger(fn, name):
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == name
                and node.func.attr in ("debug", "info", "warning", "error", "exception")):
            return True
    return False


problems = []
for path in ROOT.rglob("*.py"):
    if any(p in SKIP for p in path.parts):
        continue
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        continue
    mod_loggers = module_logger_names(tree)
    if not mod_loggers:
        continue
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound = bound_names_in_func(node)
            for name in mod_loggers & bound:
                if calls_logger(node, name):
                    problems.append(f"{path}:{node.lineno}  {node.name}() shadows `{name}`")

if problems:
    print("SHADOW PROBLEMS:")
    for p in problems:
        print("  " + p)
else:
    print("OK: no logger-shadow collisions in live source")
