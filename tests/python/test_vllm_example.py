"""The vLLM example's own options that do not need a vLLM server."""

import os
import subprocess
import sys

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
