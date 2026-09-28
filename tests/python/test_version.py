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
