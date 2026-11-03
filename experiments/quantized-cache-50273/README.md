# Quantized KV-cache rollback — evidence for Transformers issue #50273 / PR #50277

> **Status: correctness investigation, not a claim of supported quantized speculative decoding or a speedup.**
> These are experimental results reported from a Kaggle run on **2026-11-03**, using
> **2 × NVIDIA Tesla T4**, PyTorch `2.11.0+cu128`, and Transformers development source.
> Measurements were supplied from the Kaggle output; they are **not independently reproduced in this repository**.

**Issue:** https://github.com/huggingface/transformers/issues/50273  
**PR:** https://github.com/huggingface/transformers/pull/50277

## What the issue demonstrated

The original unpatched `QuantizedCache.crop(-2)` raised
`IndexError: too many indices for tensor of dimension 1` after updating each of
FP8, Quanto 4-bit and HQQ 4-bit KV caches. The `DynamicCache` control did not.
Original pretrained Qwen2.5 target (1.5B) / assistant (0.5B) tests with forced
draft rejection also hit the FP8 rollback exception, whereas DynamicCache succeeded.
Reproduction commands and full system details are in [the issue](https://github.com/huggingface/transformers/issues/50273).

Maintainer feedback clarified quantized-cache assisted decoding **was not previously supported**;
thus this is an unsupported-operation/API-capability defect, **not a regression of a supported feature**.

## Why the initial slicing-based FP8 rollback implementation was unsafe

A synthetic stress test used 8 retained tokens valued `0.01`, then 4 speculative
(rejected) tokens valued `10000.0`. The initial per-tensor FP8 scale was calibrated
using the full 12-token sequence. After truncating the packed FP8 states to 8 tokens:

| Metric | Separately quantized retained prefix | Simulated FP8 slicing rollback |
|---|---:|---:|
| KV length | 8 | 8 |
| Quantization scale | 0.0000223261977 | 22.3214302063 |
| Mean dequantized retained KV | 0.0100021362 | 0.0 |
| Max absolute retained KV discrepancy | — | 0.0100021362 |

This demonstrates irreversible loss of retained values in an extreme input case:
removing quantized suffix tokens **cannot undo** their influence on the earlier scale.
Run [`kaggle_fp8_scale_audit.py`](kaggle_fp8_scale_audit.py) against the current
PR checkout. It **simulates the superseded patch's slice** because the current PR
correctly rejects `crop(-n)`.

The stress-test values are **not representative model activations**, and this test
alone does **not** isolate the cause of the separate Qwen2.5 parity mismatch.

## Historical pretrained benchmark (superseded feature implementation)

Tests used the **former rollback-support commit**
`43b2226edb2de97e796a53b8948132ae42eab0a5`, before it was replaced with an
explicit unsupported-operation guard. Target: `Qwen/Qwen2.5-1.5B-Instruct` on
GPU 0; draft: `Qwen/Qwen2.5-0.5B-Instruct` on GPU 1. Input: excerpts of the
Transformers KV-cache documentation. Values below are end-to-end generation
throughput, **including prefill**; each cell represents the mean of two runs.

| Mode | 1K target context (tok/s) | 4K target context (tok/s) |
|---|---:|---:|
| Dynamic greedy | 26.24 | 10.71 |
| Dynamic assisted | 17.09 | 6.79 |
| FP8 greedy | 23.82 | 8.52 |
| **FP8 assisted — superseded patch** | **16.76** | **4.51** |

**Important:** At the 4K setting, output lengths differed: Dynamic generated
41 tokens, FP8 greedy 30 tokens, and FP8 assisted 24 tokens; the throughput
numbers are **not** directly comparable at equal generated-token counts.

| Context | Dynamic greedy cache (MiB) | FP8 greedy cache (MiB) | Dynamic assisted cache (MiB) | FP8 assisted cache (MiB) |
|---|---:|---:|---:|---:|
| 1K | 25.16 | 12.58 | 25.16 | 12.58 |
| 4K | 108.96 | 54.33 | 108.96 | 54.25 |

Approx. 50% reduction in *stored KV tensor bytes* did **not** translate to a
similar reduction in peak GPU allocated memory (e.g. 4K Dynamic assisted
4886.68 MiB vs FP8 assisted 4835.81 MiB).

- **Natural decoding:** Dynamic assisted matched Dynamic greedy 4/4; FP8 assisted
  diverged from FP8 greedy 4/4 (first divergent generated token was 4 at 1K,
  and 11 at 4K).
- **Forced rejected draft tokens:** both cache types processed negative crop
  calls; Dynamic assisted preserved own-cache greedy-token parity, while FP8
  assisted **did not** (first mismatch at generated token 3).
- `BENCHMARK_VALID = False`; the benchmark's previous publishing safeguard
  correctly refused to claim validated results.

Reproduction source: [`historical_benchmark.py`](historical_benchmark.py).
It requires a checkout of the **historical feature commit**, a properly configured
Python environment, two CUDA devices, and `BENCH_REPO`/`BENCH_OUT` environment
variables. **Do not run it against the present PR**: that revision intentionally
rejects quantized assisted generation. If present from the original Kaggle run,
[`results.json`](results.json) and [`run.log`](run.log) provide the complete
per-run token sequences, memory, and exception details. The CSV files contain
values transcribed from the output, **not reconstructed raw per-run logs**.

## Current scope of PR #50277

The revised PR reports quantized layers as `is_croppable = False` and raises an
actionable `ValueError` for unsupported rollback. Assisted decoding rejects
`QuantizedCache` specifically, without rejecting other non-croppable cache
implementations. This is a **narrow API correctness fix**, not a performance
feature or proof that quantized speculative decoding works.

## Files

- [`kaggle_fp8_scale_audit.py`](kaggle_fp8_scale_audit.py): standalone deterministic
  FP8 scale counterexample, plus expected-current-PR rejection behavior
- [`historical_benchmark.py`](historical_benchmark.py): exact benchmark worker
  source extracted from the Kaggle cell used for the measurements
- [`reported_run_metrics.csv`](reported_run_metrics.csv): 16 reported wall-time/
  throughput measurements and output lengths
- [`reported_memory_metrics.csv`](reported_memory_metrics.csv): reported median
  KV tensor and peak PyTorch allocated GPU-0 memory figures
- `results.json` and `run.log`, **only if actually present** in the original
  Kaggle results directory at publication time

No secret values are stored here. The evidence branch does not modify the PR branch.
