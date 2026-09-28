---
title: "CUPTI hardware findings"
tags:
  - cupti
  - nvidia
  - gpu
---

# CUPTI hardware findings

Behaviours of CUPTI and the hardware under it. Sections 1–3 were found
while trying to add per-SM kernel tracing next to PM Sampling. That attempt was abandoned,
and the code it used (a kernel-activity capture path on the branch
`feature/sass-exploration`, commit `c1319c6`) is **not part of this
repository**. They are kept here because they are properties of the
hardware and the API, not of that code, and they bear on choices this
repository makes. For the overhead of each CUPTI subsystem, see
[cupti-overhead-analysis.md](cupti-overhead-analysis.md).

> [!IMPORTANT]
> **Sections 1–3: recorded on CUDA 12.8 / CUPTI 12.8, H100.** This
> repository now builds and runs on **CUDA 13.3** (with forward
> compatibility on a 12.8 driver). They have not been re-tested on 13.3;
> treat them as true for 12.8 and unverified for 13.3. **Section 4 was
> measured on CUDA 13.3** (`libcupti.so.2026.2.1`, H100 NVL, driver 570.124
> with forward compatibility, 2026-09-28).

| # | Finding | Hardware | CUDA / CUPTI | Recorded |
|---|---|---|---|---|
| 1 | PM Sampling silences Activity-API kernel records | H100 | 12.8 | 2026-05-16 |
| 2 | Short captures need `CUPTI_ACTIVITY_FLAG_FLUSH_FORCED` | H100 | 12.8 | same attempt (`c1319c6`) |
| 3 | The legacy Event/Metric API is gated off on Turing and newer | H100 | 12.8 | same attempt; written up 2026-08-27 |

## 1. PM Sampling and Activity-API kernel tracing cannot run concurrently

While PM Sampling is active, `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` records
are never delivered: `BufferRequested` fires once and `BufferCompleted`
never fires at all. Both subsystems contend for the same GPU
instrumentation channel.

The degradation is **one-directional**. PM Sampling keeps producing counter
samples; it is the Activity side that goes silent. So the failure presents
as "no kernels were captured", not as an error — an empty capture is the
only symptom. (It was first described the other way round, as PM Sampling
being disturbed; it is not.)

Consequence: correlating PM Sampling samples with per-kernel Activity
records needs two separate passes over the workload. This repository's GPU
probe uses PM Sampling only.

## 2. Short Activity-API captures need a forced flush

`cuptiActivityFlushAll(0)` only delivers buffers that CUPTI considers full
enough, which for a sub-second capture can be never: the records stay in
CUPTI's internal pipeline until process exit. Pass
`CUPTI_ACTIVITY_FLAG_FLUSH_FORCED` when flushing short runs.

## 3. The legacy Event/Metric API is gated off on Turing and newer

NVIDIA's `cupti-tutorial` NVLink-bandwidth sample sets
`metricSupport = false` for any device of compute capability above 7.0 and
then waives with "Legacy CUPTI metrics not supported from Turing+ devices."
On the H100 that check fired, and driving the sample as far as
`cuptiMetricGetIdFromName` required forcing `metricSupport = true` past it.

This is the practical reason this repository uses PM Sampling rather than
the legacy `cuptiMetricGetIdFromName` / `cuptiMetricCreateEventGroupSets`
route: on current GPUs the legacy route is closed before its overhead is
even the question.

## 4. PM Sampling decode: what `cuptiPmSamplingDecodeData` does (CUDA 13.3)

Measured with a standalone program driving the PM Sampling API directly
(4 metrics unless stated), which shaped the GPU probe's decode loop
(`lib/src/decode_thread.cpp`):

- **The decode stop reason tells whether the hardware buffer was
  drained.** `CUpti_PmSampling_DecodeData_Params.decodeStopReason` is
  `END_OF_RECORDS` (or `OTHER`) when everything buffered was decoded, and
  `COUNTER_DATA_FULL` when the counter-data image ran out of slots first.
  With an image large enough for the pass, samples are complete and
  contiguous across passes (each starts where the previous one ended;
  5128 of 5128 at 1 kHz), and a sample still being written stays in the
  buffer for the next pass.
- **A full image loses samples, whatever is done next.** After a decode
  that stopped at `COUNTER_DATA_FULL`, the following decodes (into the same
  image re-initialized, into a second pre-initialized image, or into a
  newly allocated one) return samples with both timestamps 0, and the real
  samples in between are gone: at 1 kHz with a 50-slot image and a pass
  every 200 ms, 150 of every 200 samples. Decoding again without
  re-initializing makes no progress. So the image must hold a whole pass;
  the probe sizes it for one (plus 25% and 64 slots), drops samples without
  timestamps and counts what is missing.
- **A hardware-buffer overflow is permanent.** `DecodeData` then returns
  `CUPTI_ERROR_OUT_OF_MEMORY` with `overflow = 1` and no samples, and so
  does every later call. The probe counts it and warns; it is not a fatal
  error.
- **Hardware-buffer bytes per sample depend on the metrics** and are not
  exposed by the API: about 4.2–4.6 KB with 1–2 metrics and 5.9–7.0 KB
  with 4 (an 8 MiB buffer at 1 kHz overflows between 1.2 and 1.4 s of
  undecoded samples with 4 metrics). The probe checks `hw_buffer_size`
  against 16 KiB per sample.
- **Counter-data image: ~16.3 KB per slot** with 4 metrics
  (`cuptiPmSamplingGetCounterDataSize`: 817,256,212 bytes for 50,000
  slots). Re-initializing it (`cuptiPmSamplingCounterDataImageInitialize`)
  costs time in proportion to its size: ~60 ms for 50,000 slots.
- **`GetCounterDataSize` with another `maxSamples` breaks later image
  initializations**: after querying the size for a different slot count,
  `CounterDataImageInitialize` on an existing image fails with
  `CUPTI_ERROR_UNKNOWN`. Two images for double-buffering must have the
  same size.
- **Zero-length samples after stop.** The decode after
  `cuptiPmSamplingStop` returns one or two samples with start == end,
  stamped at the time sampling started. They cover no time.
- The 13.3 header says nothing on whether calls on one PM sampling object
  (or its host object) may run concurrently; the probe serializes them.
