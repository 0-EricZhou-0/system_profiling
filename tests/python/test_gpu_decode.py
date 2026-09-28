"""GPU probe decode: every sample the PM sampler takes reaches the trace,
and when it cannot (counter-data image too small for a decode pass,
hardware-buffer overflow) the loss is counted in the trace and reported
on stderr, never silent. Needs a GPU with PM sampling.
"""

from gpu_helpers import decode_stats, gpu_config, gpu_frames, run_child, sample_times, warnings

RUN = "time.sleep(3.0); suite.stop()"


def _check_contiguous(ts, period_ns):
    assert ts, "no GPU samples"
    diffs = [b - a for a, b in zip(ts, ts[1:])]
    assert min(diffs) > 0, "timestamps not strictly increasing (duplicates or going back)"
    assert max(diffs) <= period_ns * 3 // 2, f"hole: max gap {max(diffs) / 1e6:.3f} ms"
    assert 0 not in ts


def test_complete_at_1khz(tmp_path):
    rc, out, err = run_child(gpu_config(tmp_path, sampling_frequency_hz=1000), RUN)
    assert rc == 0, err
    frames = gpu_frames(tmp_path)
    ts = sample_times(frames)
    _check_contiguous(ts, 1_000_000)
    span_s = (ts[-1] - ts[0]) / 1e9
    assert span_s > 2.5
    assert abs(len(ts) - (span_s * 1000 + 1)) <= 2, (len(ts), span_s)
    st = decode_stats(frames)
    assert st is not None and st.samples == len(ts)
    assert (st.samples_lost, st.invalid_samples, st.counter_data_full, st.hw_buffer_overflows) == (0, 0, 0, 0)
    assert st.empty_samples <= 2   # CUPTI's zero-length start marker(s), not data
    assert st.stretched_samples <= 1   # the run's first sample, covering sampling start
    assert not warnings(err), warnings(err)


def test_undersized_image_is_detected_and_reported(tmp_path):
    """max_samples=2 at 1 kHz: every decode pass overfills the image.
    CUPTI then returns invalid samples and loses real ones; they must not
    reach the trace as data, and the loss must be counted and reported."""
    rc, out, err = run_child(gpu_config(tmp_path, sampling_frequency_hz=1000, max_samples=2), RUN)
    assert rc == 0, err
    frames = gpu_frames(tmp_path)
    ts = sample_times(frames)
    assert ts, "no GPU samples at all"
    diffs = [b - a for a, b in zip(ts, ts[1:])]
    assert 0 not in ts and min(diffs) > 0, "invalid/duplicate samples reached the trace"
    st = decode_stats(frames)
    assert st is not None, "no decode stats in the trace"
    assert st.counter_data_full > 0
    assert st.samples_lost + st.invalid_samples > 0
    assert st.samples == len(ts)
    w = warnings(err)
    assert any("counter-data image (max_samples) filled up" in l for l in w), w
    assert any("decode summary" in l for l in w), w
    for kind in ("filled up", "falling behind", "missing between", "decode summary"):
        assert sum(kind in l for l in w) <= 1, f"warnings must not repeat per pass: {w}"
