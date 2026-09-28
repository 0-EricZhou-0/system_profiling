"""The GPU probe survives its decode thread falling behind. A stall long
enough to overflow the hardware buffer (CUPTI then returns nothing, for
good) makes the decoder re-enable the sampler; a stall that only
overfills the counter-data image costs that pass's excess samples. Either
way sampling goes on, the loss is counted and each event is reported.
Needs a GPU with PM sampling.
"""

import json

from gpu_helpers import decode_stats, gpu_config, gpu_frames, last_times_steady, run_child, warnings

STALL = """
time.sleep(2.5)                                  # two normal passes first
cp._native._testing_stall_next_decode_ms(%d)
time.sleep(%f)
print(json.dumps({"t_stop": time.monotonic_ns()}), flush=True)
suite.stop()
"""


def _run(tmp_path, stall_ms, run_s, **gpu):
    cfg = gpu_config(tmp_path, sampling_frequency_hz=1000, **gpu)
    rc, out, err = run_child(cfg, STALL % (stall_ms, run_s))
    assert rc == 0, err[-3000:]
    t_stop = json.loads([l for l in out.splitlines() if l.startswith("{")][-1])["t_stop"]
    frames = gpu_frames(tmp_path)
    return frames, decode_stats(frames), last_times_steady(tmp_path)["gpu"], t_stop, warnings(err)


def test_overflow_restarts_the_sampler(tmp_path):
    # 16 MiB (the least a 500 ms interval allows) holds ~3.8 s at 1 kHz
    # with these 2 metrics (~4.4 KB per sample); a pass comes 8 s late.
    frames, st, last, t_stop, w = _run(tmp_path, 8000, 12.0, decode_interval_ms=500,
                                       hw_buffer_size=16 << 20)
    assert st.hw_buffer_overflows >= 1 and st.sampler_restarts >= 1, st
    assert last > t_stop - 1_500_000_000, "no samples after the restart"
    assert 6000 < st.samples_lost < 11000, st          # the stall's samples, counted
    assert st.samples > 3000, st                          # sampling went on
    assert any("hardware buffer overflowed" in l and "re-enabled" in l for l in w), w


def test_full_image_recovers_by_itself(tmp_path):
    # The third pass comes 6 s late: more samples than the image holds.
    frames, st, last, t_stop, w = _run(tmp_path, 6000, 10.0)
    assert st.counter_data_full >= 1 and st.hw_buffer_overflows == 0, st
    assert last > t_stop - 1_500_000_000, "no samples after the full pass"
    assert st.samples_lost + st.invalid_samples > 0, st
    ts = [s.timestamp_ns for f in frames for s in f.samples]
    assert all(b > a for a, b in zip(ts, ts[1:])), "out-of-order samples reached the trace"
    assert sum(1 for l in w if "filled up" in l) == 1, w


def test_falling_behind_is_warned_before_loss(tmp_path):
    """A pass 2.5 s late (3.5 s of samples waiting, 86% of the 1 kHz
    image) loses nothing but is reported."""
    frames, st, last, t_stop, w = _run(tmp_path, 2500, 6.0)
    assert st.late_passes >= 1, st
    assert (st.counter_data_full, st.samples_lost, st.hw_buffer_overflows) == (0, 0, 0), st
    assert sum(1 for l in w if "falling behind" in l) == 1, w
