"""Images in the docs: every image a Markdown file shows exists, and
every image directly in docs/images/ is shown somewhere (old ones live in
docs/images/archive/)."""

import glob
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _shown():
    shown = set()
    for md in glob.glob(os.path.join(REPO, "**", "*.md"), recursive=True):
        if "/build/" in md or "/.git/" in md:
            continue
        for ref in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", open(md, encoding="utf-8").read()):
            if "://" in ref:
                continue
            path = os.path.normpath(os.path.join(os.path.dirname(md), ref))
            assert os.path.exists(path), f"{md} shows {ref}, which does not exist"
            shown.add(path)
    return shown


def test_images_shown_exist_and_top_level_images_are_shown():
    shown = _shown()
    top = {os.path.normpath(p) for p in glob.glob(os.path.join(REPO, "docs", "images", "*.png"))}
    assert top, "no images in docs/images"
    assert top <= shown, f"not shown anywhere (move to docs/images/archive/): {sorted(top - shown)}"
