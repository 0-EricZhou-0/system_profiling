"""tools/label_spread.js (the Bokeh page's label placement) gives the
same results as tools/label_spread.py on the same inputs: random bars,
views and row limits, run under node. Skipped without node."""

import json
import os
import random
import shutil
import subprocess

import pytest

import viz_trace  # noqa: F401  (puts tools/ on sys.path)
import label_spread  # noqa: E402

NODE = shutil.which("node")
JS = os.path.join(viz_trace.TOOLS, "label_spread.js")
CHAR = 5.3


def _cases(n=300):
    rng = random.Random(7)
    cases = []
    for k in range(n):
        span = rng.choice([10.0, 60.0, 250.0])
        bars = []
        for _ in range(rng.randint(0, 90)):
            left = rng.uniform(-0.1 * span, span)
            right = left if rng.random() < 0.15 else left + rng.expovariate(1 / (0.05 * span))
            name = "".join(rng.choice("abcdefgh") for _ in range(rng.randint(1, 12)))
            bars.append([left, right, [f"{name} ({rng.randint(10, 99999)})", name]])
        lo = rng.uniform(0, 0.6 * span) if k % 2 else 0.0
        hi = lo + rng.uniform(0.05, 1.0) * (span - lo)
        cases.append(dict(bars=bars, lo=lo, hi=hi, width=rng.choice([300.0, 860.0]),
                          pad=rng.choice([0.0, 6.0, 10.0]), max_rows=rng.choice([0, 1, 2, 7, 16]),
                          max_shift=rng.choice([None, 0.08 * 860.0])))
    return cases


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_matches_python(tmp_path):
    cases = _cases()
    want = [label_spread.place_bar_labels(
        [tuple(b[:2]) + (b[2],) for b in c["bars"]], c["lo"], c["hi"], c["width"],
        lambda t: len(t) * CHAR, c["pad"], c["max_rows"], c["max_shift"]) for c in cases]
    inp = tmp_path / "cases.json"
    inp.write_text(json.dumps(cases))
    script = (f"const L = require({json.dumps(JS)});"
              f"const cs = JSON.parse(require('fs').readFileSync({json.dumps(str(inp))}, 'utf8'));"
              f"console.log(JSON.stringify(cs.map(c => L.placeBarLabels(c.bars, c.lo, c.hi, c.width,"
              f" t => t.length * {CHAR}, c.pad, c.max_rows, c.max_shift))));")
    got = json.loads(subprocess.run([NODE, "-e", script], capture_output=True, text=True,
                                    check=True, timeout=120).stdout)
    assert len(got) == len(want)
    n_out = 0
    for k, (g, w) in enumerate(zip(got, want)):
        assert len(g) == len(w), k
        for gi, wi in zip(g, w):
            assert (gi is None) == (wi is None), (k, gi, wi)
            if wi is None:
                continue
            assert gi[0] == wi[0] and gi[1] == wi[1] and gi[4] == wi[4], (k, gi, wi)
            assert gi[2] == pytest.approx(wi[2], abs=1e-9) and gi[3] == pytest.approx(wi[3], abs=1e-9)
            n_out += wi[0] == "out"
    assert n_out > 500                              # the off-bar path is exercised


def test_off_bar_labels_never_overlap():
    for c in _cases(120):
        placed = label_spread.place_bar_labels(
            [tuple(b[:2]) + (b[2],) for b in c["bars"]], c["lo"], c["hi"], c["width"],
            lambda t: len(t) * CHAR, c["pad"], c["max_rows"], c["max_shift"])
        per = c["width"] / (c["hi"] - c["lo"])
        rows: dict = {}
        for p in placed:
            if p and p[0] == "out":
                assert p[4] < c["max_rows"]
                w = len(p[1]) * CHAR
                x = (p[2] - c["lo"]) * per
                rows.setdefault(p[4], []).append((x - w / 2, x + w / 2))
        for spans in rows.values():
            spans.sort()
            for (a0, a1), (b0, _b1) in zip(spans, spans[1:]):
                assert a1 + c["pad"] <= b0 + 1e-6
            assert spans[0][0] >= -1e-6 and spans[-1][1] <= c["width"] + 1e-6
