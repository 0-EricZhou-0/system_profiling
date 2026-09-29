---
title: "Full-System Profiler — Implementation Internals"
tags:
  - profiling
  - implementation
  - proc
  - linux
  - threading
---

# Full-system profiler internals

Detailed implementation documentation covering /proc parsing, threading model, protobuf streaming format, timestamp alignment, permission requirements, and design decisions.

## Project structure

```text ln:false
lib/
├── include/cupti_profiler/
│   ├── gpu_profiler.h          Public API — GPU PM Sampling (existing)
│   ├── system_profiler.h       Public API — CPU + memory
│   ├── disk_profiler.h         Public API — disk I/O
│   └── profiler_suite.h        Public API — orchestrator
└── src/
    ├── gpu_profiler.cpp         GpuProfiler::Impl (existing)
    ├── proc_readers.h/cpp       /proc/stat, /proc/meminfo, /proc/[PID]/*
    ├── disk_readers.h/cpp       /proc/diskstats, /sys/block/*, /proc/[PID]/io
    ├── system_profiler.cpp      SystemProfiler::Impl + sample thread
    ├── system_flush_thread.h/cpp  Flush for SystemMetricsTrace
    ├── disk_profiler.cpp        DiskProfiler::Impl + sample thread
    ├── disk_flush_thread.h/cpp  Flush for DiskMetricsTrace
    └── profiler_suite.cpp       Config loading + lifecycle orchestration
```

---

## /proc parsing details

### CPU utilization — `/proc/stat`

**Source:** First line of `/proc/stat`.

```text ln:false title:"/proc/stat format"
cpu  user nice system idle iowait irq softirq steal guest guest_nice
```

All values are cumulative **jiffies** (ticks of `CLK_TCK`, typically 100 Hz). To compute utilization between two samples:

```text ln:false
total = user + nice + system + idle + iowait + irq + softirq + steal
busy  = total - idle - iowait

CPU utilization % = delta(busy) / delta(total) × 100
User %            = delta(user + nice) / delta(total) × 100
System %          = delta(system) / delta(total) × 100
IOWait %          = delta(iowait) / delta(total) × 100
```

> [!NOTE]
> `guest` and `guest_nice` values are already included in `user` and `nice` respectively. They must not be added again.

**Implementation:** `ReadCPUStat()` in `lib/src/proc_readers.cpp` reads the first line, parses 8 fields into a `CPUStatSnapshot`. Delta computation happens in `SystemProfiler::Impl`'s sample thread.

### Per-process CPU — process CPU clock (`clock_getcpuclockid`)

**Source:** `clock_getcpuclockid(pid, &clk)` then
`clock_gettime(clk, &ts)` — one syscall per PID per tick. The kernel
returns `sum_exec_runtime` (nanoseconds on a CPU) summed over every
live thread of the process **plus the time already folded into the
thread group by threads that have exited**. The value is therefore
monotonic for the life of the process, whatever its thread churn.
It is readable for other users' processes without any capability.

**Granularity:** reading *another* process's clock returns each
thread's `sum_exec_runtime` as last updated by the scheduler — at a
context switch or at the scheduler tick (`CONFIG_HZ`; 4 ms at
250 Hz). A thread that runs without being descheduled is therefore
credited in tick-sized steps, so a single 10 ms sample can be off by
up to one tick per running thread (at HZ=250 and 100 Hz sampling, a
spinning single thread is credited 2 or 3 ticks per ~10.3 ms sample
and reads ~78% or ~116%). The
error does not accumulate: the time lands in the next sample, and
the sum over samples is exact.

**Why not `/proc/[PID]/schedstat`:** the TGID-level inode reports
the thread-group leader's `task_struct` only — not aggregated
across the thread group — so it caps a multi-threaded workload at
100% of one core.

**Why not walk `/proc/[PID]/task/*/schedstat`:** summing each live
thread's `sum_exec_runtime` and diffing per TID loses every thread
that exits between ticks — the slice from its last sighting to its
exit, and *all* of the CPU of a thread that lives less than one
tick. Measured on a compute node at 100 Hz sampling, against a
process that ran 300 threads one after another, each burning 2 ms
and exiting: the walk traced 0.049–0.125 s of a 0.62–0.64 s ground
truth; the CPU clock traced 0.635–0.642 s (six runs each;
`tests/python/test_cpu_clock.py`). The walk also costs one `opendir`
plus one file read per thread per tick — measured in C++ (`-O2`) at
11.9 / 204 / 780 µs per read for 1 / 31 / 120 threads, against
0.69 / 0.81 / 1.43 µs for the CPU clock (545× at vLLM's ~120
threads); the CPU clock is one syscall regardless of the thread
count.

**Per-process CPU %:**

```text ln:false
            cur − prev              ← cur, prev: process CPU clock (ns)
Active % = ──────────── × 100          at this tick and the previous one
               dt_ns
```

`dt_ns` is the **actual** wall-clock elapsed between the previous
sample tick and this one — *not* the nominal sample period.
`sleep_for` + per-PID `/proc` reads + scheduler jitter make the
real period strictly ≥ nominal; using the nominal value as the
denominator would systematically inflate the result (a tick that
takes 15 ms with the sampler configured at 100 Hz would report a
fully-busy thread as 150% instead of 100%).

The first tick a PID is seen only sets its baseline; its first
sample is emitted one tick later. If the clock cannot be read
(`ESRCH`/`EINVAL`: the process is gone), the PID is skipped for
that tick and its baseline is left untouched, so no negative or
garbage delta is emitted.

> [!IMPORTANT]
> Per-process CPU % can exceed 100% on multi-core systems (e.g., a
> process using 4 cores reports ~400%). The panel `peak_expr`
> caps the y-axis at `ncpus × 100`.

> [!NOTE]
> `sum_exec_runtime` is always tracked by the kernel scheduler
> regardless of the `kernel.sched_schedstats` sysctl, so per-PID
> CPU works out-of-the-box on every mainstream distro. This is the
> reason we do not use `/proc/[PID]/stat`'s `utime/stime` (10 ms
> `CLK_TCK` quantization → 0/100/200% staircase at 100 Hz sampling).
> The cost is that we do not split per-PID time into user / kernel /
> iowait — `sum_exec_runtime` is total on-CPU time only.

**Implementation:** `ReadPIDCpuTimeNs()` in
`lib/src/proc_readers.cpp`; per-PID baselines (CPU clock + actual
`dt`) live on `SystemProfiler::Impl::prevPID`.

### System memory — `/proc/meminfo`

**Source:** Key-value pairs in kB.

```text ln:false
MemTotal:       131072000 kB
MemFree:         12345678 kB
MemAvailable:    98765432 kB
Buffers:          1234567 kB
Cached:          45678901 kB
```

**Used memory** (matches `free` command):

```text ln:false
Used = MemTotal - MemFree - Buffers - Cached
```

`MemAvailable` (kernel 3.14+) is the best estimate of memory available to applications without swapping.

**Implementation:** `ReadMemInfo()` in `lib/src/proc_readers.cpp`. Values are converted from kB to bytes (× 1024) when stored in protobuf.

### Per-process memory — `/proc/[PID]/statm`

**Source:** 7 space-separated integers in **pages** (multiply by `sysconf(_SC_PAGESIZE)`, typically 4096).

```text ln:false
Fields: size resident shared text lib data dt
  [0] size     = VMS (total virtual memory)
  [1] resident = RSS (resident set size)
  [2] shared   = shared pages
```

**Implementation:** `ReadPIDStatm()` in `lib/src/proc_readers.cpp`.

### Disk throughput — `/proc/diskstats`

**Source:** One line per block device.

```text ln:false
major minor name rd_ios rd_merges rd_sectors rd_ticks wr_ios wr_merges wr_sectors wr_ticks ios_inflight io_ticks weighted_io_ticks
```

Fields of interest (0-indexed after `name`):

| Index | Field            | Type        | Notes                          |
| ----- | ---------------- | ----------- | ------------------------------ |
| 2     | `rd_sectors`     | cumulative  | Sectors read (× 512 = bytes)  |
| 6     | `wr_sectors`     | cumulative  | Sectors written (× 512 = bytes) |
| 8     | `ios_inflight`   | instantaneous | Currently in-flight IOs       |

**Throughput:**

```text ln:false
Read MB/s  = delta(rd_sectors) × 512 / dt_seconds / 1e6
Write MB/s = delta(wr_sectors) × 512 / dt_seconds / 1e6
```

> [!NOTE]
> Sectors are **always** 512 bytes regardless of the disk's physical sector size. This is a kernel convention.

**Implementation:** `ReadDiskStats()` in `lib/src/disk_readers.cpp`. Filters by the device list from config.

### Disk queue depth — `/sys/block/<dev>/inflight`

**Source:** Single line with two integers.

```text ln:false
<read_inflight> <write_inflight>
```

This gives the instantaneous number of in-flight read and write requests, which is the **queue depth** at the moment of sampling.

**Implementation:** `ReadDiskInflight()` in `lib/src/disk_readers.cpp`.

### Per-process disk I/O — `/proc/[PID]/io`

**Source:** Key-value pairs.

```text ln:false
rchar: 12345678                ← syscall-layer reads (any fd; includes page cache hits; never mmap)
wchar: 87654321                ← syscall-layer writes (any fd)
syscr: 1234
syscw: 5678
read_bytes: 4096000            ← fetched from storage (read() misses and mmap faults that miss)
write_bytes: 2048000           ← file pages dirtied
cancelled_write_bytes: 0       ← dirtied pages discarded before writeback
```

We emit all five byte counters, each as its own metric (`proc__io_rchar`, `proc__io_wchar`, `proc__io_read_bytes`, `proc__io_write_bytes`, `proc__io_cancelled_write_bytes`, all `.sum.per_second`). What each sees is in [metric-model.md, "Per-PID I/O counters"](metric-model.md#per-pid-io-counters-who-records-what).

> [!WARNING]
> This file requires **same-UID** ownership or `CAP_SYS_PTRACE`. If access is denied, the profiler logs a warning once per PID and skips per-process disk data rather than crashing.

**Implementation:** `ReadPIDIO()` in `lib/src/disk_readers.cpp`. Returns `accessible = false` on `EACCES`.

---

## Threading model

Each profiler follows the same 2-thread pattern:

```text ln:false
┌───────────────────┐
│  Profiler::Start() │
└────┬──────────┬───┘
     │          │
     ▼          ▼
┌─────────┐  ┌──────────┐
│ Sample   │  │ Flush    │
│ Thread   │  │ Thread   │
│          │  │          │
│ Reads    │  │ Drains   │
│ /proc at │  │ samples  │
│ interval │  │ at flush │
│          │  │ interval │
│ Computes │  │          │
│ deltas   │  │ Writes   │
│          │  │ length-  │
│ Pushes   │  │ delimited│
│ to batch │  │ protobuf │
│ (mutex)  │  │ to file  │
└─────────┘  └──────────┘
```

### GPU profiler threads

The GPU profiler has the same conceptual structure but uses CUPTI-specific APIs:

- **Decode thread** (equivalent to sample thread): once per `decode_interval_ms` (1 s), calls `cuptiPmSamplingDecodeData()` until the hardware buffer is drained, into one of two counter-data images
- **Eval worker**: evaluates each decoded image's samples via `cuptiProfilerHostEvaluateToGpuValues()` and re-initializes it, while the next decode uses the other image
- **Flush thread**: drains evaluated `SamplerRange` samples, writes length-delimited `GpuMetricsTrace`

### Synchronization

- **Sample batch**: `std::vector` of protobuf sample messages, protected by `std::mutex batchMutex`
- **Output file**: `std::ofstream` protected by `std::mutex outMutex`
- **Stop signals**: `std::atomic<bool>` per thread (`stopSample`, `stopFlush`)

### Shutdown sequence

1. `Stop()` sets `stopSample = true`, joins sample thread
2. Sets `stopFlush = true`, joins flush thread
3. Drains any remaining samples from the batch
4. Writes final length-delimited message (with regions for GPU)
5. Closes output file

---

## Protobuf streaming format

All three profilers use the same **length-delimited** streaming format:

```text ln:false
File layout:
  [varint: msg_size][serialized TraceMessage]
  [varint: msg_size][serialized TraceMessage]
  ...
  [varint: msg_size][serialized TraceMessage]  ← final (may contain regions)
```

- **Varint encoding**: standard protobuf variable-length integer (1–5 bytes for uint32)
- **Each message is self-contained**: includes metadata (hostname, interval, tracked PIDs/devices) plus a batch of samples
- **Crash safety**: if the process dies, all previously flushed messages are intact. Only the in-progress batch is lost.

The Python visualization reads these with a manual varint decoder, then merges all messages into a single trace by concatenating sample arrays.

---

## Timestamp alignment

### Clock domains

| Profiler | Clock source | Resolution |
| -------- | ------------ | ---------- |
| GPU      | `cuptiGetTimestamp()` (CUPTI internal clock) | Nanoseconds |
| System   | `std::chrono::steady_clock` | Nanoseconds |
| Disk     | `std::chrono::steady_clock` | Nanoseconds |

GPU and CPU/Disk use **different clock domains**. The visualization aligns them by normalizing each trace to **"time from first sample"** — each trace's first timestamp becomes t=0. Since `ProfilerSuite::Start()` starts all profilers within microseconds of each other, this provides adequate alignment for the millisecond-scale phenomena being measured.

> [!TIP]
> For tighter alignment in future work, `ProfilerSuite::Start()` could record both `steady_clock` and `cuptiGetTimestamp()` at the same moment and embed the offset in each trace.

---

## Config loading

The config is a **protobuf text format** file parsed via `google::protobuf::TextFormat::ParseFromString()`. This is included in `libprotobuf` which is already a dependency — no new libraries needed.

Features:
- `#` line comments
- Human-readable field names matching the `.proto` schema
- Type checking at parse time

The `ProfilerSuite::LoadConfig()` method:
1. Reads the file into a string
2. Parses into `ProfilerSuiteConfig` protobuf message
3. Converts proto fields to C++ config structs (`ProfilerConfig`, `SystemProfilerConfig`, `DiskProfilerConfig`)
4. Resolves PID `0` → `getpid()` for both system and disk profilers

---

## Permission requirements

| Resource | Required permission | Fallback |
| -------- | ------------------- | -------- |
| `/proc/stat` | World-readable | Always works |
| `/proc/meminfo` | World-readable | Always works |
| `/proc/[PID]/stat` | World-readable | Always works |
| `/proc/[PID]/statm` | World-readable | Always works |
| `/proc/diskstats` | World-readable | Always works |
| `/sys/block/*/inflight` | World-readable | Always works |
| `/proc/[PID]/io` | Same UID or `CAP_SYS_PTRACE` | Warns once, skips |
| CUPTI PM Sampling | GPU access + compute capability ≥ 7.5 | Fails at `Configure()` |

---

## Design decisions

| Decision | Rationale |
| -------- | --------- |
| Separate `.pb` files per component | Different sampling rates produce different trace sizes. Independent files allow partial collection (GPU-only, system-only, etc.) |
| `.pbtxt` config via `TextFormat::Parse` | Zero new C++ dependencies. Human-readable. Supports comments. Type-checked at parse time. |
| PID `0` sentinel resolved at runtime | User doesn't need to know their PID. Config files are reusable across runs. |
| `steady_clock` for CPU/Disk timestamps | Monotonic (no NTP jumps). Nanosecond resolution. Standard C++17. |
| `/proc` files opened and closed each read | Standard practice for `/proc` virtual filesystem. No stale file descriptors. Negligible overhead at 10–100 Hz. |
| Concrete flush threads per trace type | The three protobuf message types have different field structures. Concrete implementations are clearer than templates in a shared library. |
| Graceful `EACCES` for `/proc/[PID]/io` | Warns once per PID, skips per-process disk data. Avoids crashing when profiling other users' processes. |
| All profilers in one `libcupti_profiler.so` | Single library simplifies linking. CPU/Disk code has no CUDA runtime calls but co-locating is harmless. |
| Sample thread + flush thread per profiler | Mirrors GPU's decode + flush pattern. Decouples high-frequency collection from lower-frequency serialization. |
| Pimpl on all public classes | Public headers have zero internal/CUDA/CUPTI/protobuf includes. Users compile with any C++17 compiler. |

---

## Metrics collected

### CPU (system-wide)

| Metric | Unit | Source |
| ------ | ---- | ------ |
| `total_utilization_pct` | % | `delta(busy) / delta(total) × 100` from `/proc/stat` |
| `user_pct` | % | `delta(user+nice) / delta(total) × 100` |
| `system_pct` | % | `delta(system) / delta(total) × 100` |
| `iowait_pct` | % | `delta(iowait) / delta(total) × 100` |

### CPU (per-process)

| Metric | Unit | Source |
| ------ | ---- | ------ |
| `cpu_pct` | % of one core | `delta(process CPU clock ns) / actual_dt_ns × 100` — `clock_getcpuclockid(PID)`, the whole thread group including exited threads |

### Memory (system-wide)

| Metric | Unit | Source |
| ------ | ---- | ------ |
| `total_bytes` | bytes | `MemTotal` from `/proc/meminfo` |
| `used_bytes` | bytes | `MemTotal - MemFree - Buffers - Cached` |
| `available_bytes` | bytes | `MemAvailable` |
| `buffers_bytes` | bytes | `Buffers` |
| `cached_bytes` | bytes | `Cached` |

### Memory (per-process)

| Metric | Unit | Source |
| ------ | ---- | ------ |
| `rss_bytes` | bytes | Field 1 × PAGE_SIZE from `/proc/[PID]/statm` |
| `vms_bytes` | bytes | Field 0 × PAGE_SIZE |
| `shared_bytes` | bytes | Field 2 × PAGE_SIZE |

### Disk (per-device)

| Metric | Unit | Source |
| ------ | ---- | ------ |
| `read_bytes_per_sec` | bytes/s | `delta(sectors_read) × 512 / dt` from `/proc/diskstats` |
| `write_bytes_per_sec` | bytes/s | `delta(sectors_written) × 512 / dt` |
| `read_queue_depth` | count | Field 0 from `/sys/block/<dev>/inflight` |
| `write_queue_depth` | count | Field 1 from `/sys/block/<dev>/inflight` |

### Disk (per-process)

| Metric | Unit | Source |
| ------ | ---- | ------ |
| `proc__io_rchar.sum.per_second` | bytes/s | `delta(rchar) / dt` from `/proc/[PID]/io` |
| `proc__io_wchar.sum.per_second` | bytes/s | `delta(wchar) / dt` |
| `proc__io_read_bytes.sum.per_second` | bytes/s | `delta(read_bytes) / dt` |
| `proc__io_write_bytes.sum.per_second` | bytes/s | `delta(write_bytes) / dt` |
| `proc__io_cancelled_write_bytes.sum.per_second` | bytes/s | `delta(cancelled_write_bytes) / dt` |

---

## References

- [[full-system-overview|Overview and quick-start guide]]
- [[system-guide|GPU profiler system guide]]
- [Linux /proc/stat documentation](https://www.kernel.org/doc/html/latest/filesystems/proc.html)
- [Linux I/O statistics (iostats.rst)](https://www.kernel.org/doc/html/latest/admin-guide/iostats.html)
- [CUPTI PM Sampling API](https://docs.nvidia.com/cupti/main/main.html)
