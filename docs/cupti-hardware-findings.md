---
title: "CUPTI hardware findings"
tags:
  - cupti
  - nvidia
  - gpu
---

# CUPTI hardware findings

Three behaviours of CUPTI and the hardware under it, found while trying to
add per-SM kernel tracing next to PM Sampling. That attempt was abandoned,
and the code it used (a kernel-activity capture path on the branch
`feature/sass-exploration`, commit `c1319c6`) is **not part of this
repository**. They are kept here because they are properties of the
hardware and the API, not of that code, and they bear on choices this
repository makes. For the overhead of each CUPTI subsystem, see
[cupti-overhead-analysis.md](cupti-overhead-analysis.md).

> [!IMPORTANT]
> **Recorded on CUDA 12.8 / CUPTI 12.8, H100.** This repository now builds
> and runs on **CUDA 13.3** (with forward compatibility on a 12.8 driver).
> None of these findings has been re-tested on 13.3; treat them as true for
> 12.8 and unverified for 13.3.

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
