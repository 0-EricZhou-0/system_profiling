"""Double-buffered GPU decode: the decode thread decodes into one
counter-data image while a worker evaluates and re-initializes the other.
Every sample still arrives, in order, and the evaluation runs on the
worker (cupti-eval<N>), not on the decode thread (cupti-decode<N>).
Needs a GPU with PM sampling.
"""

import json

import pytest

from gpu_helpers import decode_stats, gpu_config, gpu_frames, run_child, sample_times, warnings

# Per-thread CPU (schedstat, ns) of the decode and eval threads, then stop.
THREAD_CPU = """
time.sleep(3.5)
cpu = {}
for tid in os.listdir("/proc/self/task"):
    comm = open(f"/proc/self/task/{tid}/comm").read().strip()
    if comm.startswith(("cupti-decode", "cupti-eval")):
        cpu[comm] = int(open(f"/proc/self/task/{tid}/schedstat").read().split()[0])
print(json.dumps(cpu), flush=True)
suite.stop()
"""


@pytest.mark.parametrize("hz", [500, 1000])
def test_lossless_and_evaluated_on_worker(tmp_path, hz):
    rc, out, err = run_child(gpu_config(tmp_path, sampling_frequency_hz=hz), THREAD_CPU)
    assert rc == 0, err
    cpu = json.loads([l for l in out.splitlines() if l.startswith("{")][-1])
    print("thread CPU ns:", cpu)
    assert set(cpu) == {"cupti-decode0", "cupti-eval0"}, cpu
    # 3 passes of hz samples each were evaluated on the worker.
    assert cpu["cupti-eval0"] > 1_000_000, cpu

    frames = gpu_frames(tmp_path)
    ts = sample_times(frames)
    period = 1_000_000_000 // hz
    diffs = [b - a for a, b in zip(ts, ts[1:])]
    assert min(diffs) > 0 and max(diffs) <= period * 3 // 2
    span_s = (ts[-1] - ts[0]) / 1e9
    assert span_s > 3.0 and abs(len(ts) - (span_s * hz + 1)) <= 2, (len(ts), span_s)
    st = decode_stats(frames)
    assert (st.samples_lost, st.invalid_samples, st.counter_data_full, st.hw_buffer_overflows) == (0, 0, 0, 0)
    assert st.samples == len(ts)
    assert not warnings(err), warnings(err)
