"""The package version lives in pyproject.toml; the docs that name a
wheel or a versioned figure must follow it."""

import os
import re
import tomllib

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _version():
    with open(os.path.join(_REPO, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_version_is_0_2_0():
    assert _version() == "0.2.0"


def test_docs_wheel_names_follow_the_version():
    text = open(os.path.join(_REPO, "docs", "integration.md")).read()
    wheels = re.findall(r"cupti_profiler-(\d[\w.]*?)-cp", text)
    assert wheels, "docs/integration.md names no wheel"
    assert set(wheels) == {_version()}, wheels


def test_viz_extra_has_both_renderers():
    """tools/visualize_all.py needs matplotlib, tools/visualize_interactive.py
    bokeh (tested with 3.9.2): both in the viz extra, bokeh with a floor."""
    with open(os.path.join(_REPO, "pyproject.toml"), "rb") as f:
        viz = tomllib.load(f)["project"]["optional-dependencies"]["viz"]
    names = {re.split(r"[<>=!~ ]", d, maxsplit=1)[0] for d in viz}
    assert {"numpy", "matplotlib", "bokeh"} <= names, viz
    [bk] = [d for d in viz if d.startswith("bokeh")]
    floor = tuple(int(x) for x in bk.split(">=")[1].split("."))
    assert (3, 0) <= floor <= (3, 9, 2), bk
