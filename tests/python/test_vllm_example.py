"""The vLLM example's own options that do not need a vLLM server."""

import ast
import fnmatch
import os
import subprocess
import sys

from google.protobuf import text_format

import panels_pb2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_vllm_example_pins_the_launcher(tmp_path):
    """--launcher-cpus pins the launcher before the suite starts (so the
    GPU probe's threads inherit it). vllm is /bin/false: the example
    stops right after starting the suite."""
    cpu = sorted(os.sched_getaffinity(0))[-1]
    p = subprocess.run([sys.executable, os.path.join(REPO, "examples", "vllm_serving_profiling.py"),
                        "--vllm", "false", "--launcher-cpus", str(cpu), "--no-render",
                        "--output-dir", str(tmp_path), "--ready-timeout", "30"],
                       capture_output=True, text=True, timeout=120)
    assert f"launcher pinned to CPUs [{cpu}]" in p.stdout, p.stdout + p.stderr


def test_vllm_example_adopts_orphans(tmp_path):
    """The launcher is a child subreaper before vLLM starts (reap chains
    resolvable, orphans kept): the situation report says so."""
    p = subprocess.run([sys.executable, os.path.join(REPO, "examples", "vllm_serving_profiling.py"),
                        "--vllm", "false", "--no-render", "--output-dir", str(tmp_path),
                        "--ready-timeout", "30"], capture_output=True, text=True, timeout=120)
    assert "launcher is a child subreaper (adopt_orphans)" in p.stdout, p.stdout + p.stderr
    sub = [l for l in p.stderr.splitlines() if "child subreaper (this process)" in l]
    assert sub and "not set" not in sub[0], sub


def _gpu_metrics():
    tree = ast.parse(open(os.path.join(REPO, "examples", "vllm_serving_profiling.py")).read())
    [node] = [n for n in tree.body if isinstance(n, ast.Assign)
              and any(getattr(t, "id", "") == "GPU_METRICS" for t in n.targets)]
    return ast.literal_eval(node.value)


def test_vllm_example_collects_sm_activity_avg_and_max():
    """--gpu collects SM activity as the mean over SMs and as the busiest
    SM, and the example's SM panel shows both."""
    metrics = _gpu_metrics()
    sm = ["sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
          "sm__cycles_active.max.pct_of_peak_sustained_elapsed"]
    assert all(m in metrics for m in sm), metrics
    layout = text_format.Parse(
        open(os.path.join(REPO, "configs", "vllm_serving_panels.pbtxt")).read(),
        panels_pb2.PanelLayout())
    [panel] = [p for p in layout.panels if p.title.startswith("SM Utilization")]
    assert all(fnmatch.fnmatchcase(m, panel.series_glob) for m in sm)
