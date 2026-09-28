"""gpu.max_samples = 0 (default) sizes the counter-data image for one
decode pass: ceil(rate x decode_interval x 1.25) + 64 samples. An
explicit value is honored. hw_buffer_size must hold two decode intervals
of samples. Needs a GPU with PM sampling.
"""

import re

import pytest

from gpu_helpers import decode_stats, gpu_config, gpu_frames, run_child, sample_times, warnings

IMAGE = re.compile(r"Counter-data image: (\d+) samples \((auto|set)\), ([\d.]+) MB")


def _image(out):
    m = IMAGE.search(out)
    assert m, out
    return int(m[1]), m[2], float(m[3])


@pytest.mark.parametrize("hz,interval_ms,expected", [
    (100, 1000, 189), (500, 1000, 689), (1000, 1000, 1314), (1000, 200, 314)])
def test_auto_size(tmp_path, hz, interval_ms, expected):
    cfg = gpu_config(tmp_path, sampling_frequency_hz=hz, decode_interval_ms=interval_ms)
    rc, out, err = run_child(cfg, "suite.stop()")
    assert rc == 0, err
    n, how, mb = _image(out)
    assert (n, how) == (expected, "auto")
    assert mb < n * 0.02, f"{mb} MB for {n} samples"   # ~16 KB per slot


def test_explicit_max_samples_honored(tmp_path):
    rc, out, err = run_child(gpu_config(tmp_path, max_samples=3000), "suite.stop()")
    assert rc == 0, err
    assert _image(out)[:2] == (3000, "set")


def test_complete_at_500hz_with_auto_image(tmp_path):
    cfg = gpu_config(tmp_path, sampling_frequency_hz=500)
    rc, out, err = run_child(cfg, "time.sleep(3.5); suite.stop()")
    assert rc == 0, err
    assert _image(out)[:2] == (689, "auto")
    frames = gpu_frames(tmp_path)
    ts = sample_times(frames)
    diffs = [b - a for a, b in zip(ts, ts[1:])]
    assert len(ts) > 1600 and min(diffs) > 0 and max(diffs) <= 3_000_000
    st = decode_stats(frames)
    assert (st.samples_lost, st.invalid_samples, st.counter_data_full, st.hw_buffer_overflows) == (0, 0, 0, 0)
    assert not warnings(err), warnings(err)


def test_hw_buffer_too_small_rejected(tmp_path):
    cfg = gpu_config(tmp_path, sampling_frequency_hz=1000, hw_buffer_size=4 << 20)
    rc, out, err = run_child(cfg, "suite.stop()")
    assert rc != 0
    assert "ProfilerSuite::Configure failed: InvalidConfig" in err, err
    assert "hw_buffer_size (4 MiB) cannot hold two decode intervals" in err, err
