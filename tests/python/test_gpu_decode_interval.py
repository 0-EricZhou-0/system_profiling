"""gpu.decode_interval_ms: the host collects the GPU's buffered samples
once per interval (default 1 s), waits interruptibly (Stop() does not
wait out the interval), and a flush interval below it is rejected at
Configure(). Needs a GPU with PM sampling.
"""

import json

from gpu_helpers import decode_stats, gpu_config, gpu_frames, run_child, sample_times

# Report the process's own CPU over 3.5 s of sampling (so stop() lands
# mid-interval), then how long stop() took.
TIMED = """
c0 = time.process_time(); time.sleep(3.5); cpu = time.process_time() - c0
t = time.monotonic(); suite.stop(); stop_s = time.monotonic() - t
print(json.dumps({"cpu_s": cpu, "stop_s": stop_s}), flush=True)
"""


def _result(out):
    return json.loads([l for l in out.splitlines() if l.startswith("{")][-1])


def test_decode_interval_honored(tmp_path):
    rc, out, err = run_child(gpu_config(tmp_path, decode_interval_ms=500), TIMED)
    assert rc == 0, err
    st = decode_stats(gpu_frames(tmp_path))
    # 3.5 s / 500 ms = 7 passes, plus the final drain at stop.
    assert 6 <= st.decode_calls <= 10, st.decode_calls
    assert st.samples_lost == 0 and st.counter_data_full == 0


def test_default_interval_is_cheap_and_stop_is_prompt(tmp_path):
    """Default 1 s: the launcher's decode costs little CPU (a pass per
    second, not a busy loop), and stop() interrupts the wait."""
    # A small image, so its per-pass re-initialization is not what is measured.
    rc, out, err = run_child(gpu_config(tmp_path, max_samples=2000), TIMED)
    assert rc == 0, err
    r = _result(out)
    frames = gpu_frames(tmp_path)
    st = decode_stats(frames)
    assert 3 <= st.decode_calls <= 5, st.decode_calls
    assert r["cpu_s"] < 0.15, f"launcher used {r['cpu_s']:.3f} s CPU in 3.5 s"
    assert r["stop_s"] < 0.5, f"stop() took {r['stop_s']:.3f} s"
    assert len(sample_times(frames)) > 3000


def test_flush_below_decode_interval_rejected(tmp_path):
    cfg = gpu_config(tmp_path, decode_interval_ms=1000, flush_interval_ms=500)
    rc, out, err = run_child(cfg, "suite.stop()")
    assert rc != 0
    assert "ProfilerSuite::Configure failed: InvalidConfig" in err, err
    assert "flush_interval_ms (500) is less than decode_interval_ms (1000)" in err, err
    assert "started" not in out
