# Experiments: faster agent loops on Qwen3.8-27B (TPU v5e-8)

Goal: agent loops (Claude Code, miniagent) resend the whole conversation every step, so the server
mostly spends its time on prefill. Three changes are prepared, **none of them run on a TPU yet**;
the defaults stay the proven setup (vllm-tpu 0.28.0, bf16, MTP 3, 4 sequences, 262k context).

Everything below runs inside **one** TPU session: start the proven setup, then switch with the
`reconfigure` control command (vLLM restarts in place; a failed switch restores the previous
settings). No new Kaggle queue between steps.

## Facts from the 2026-09-16 session (vllm.log)

| | |
|---|---|
| HBM | 8 x 15.75 GiB = 126 GiB, vLLM cap 115.9 GiB (utilization 0.92) |
| bf16 weights | 51.75 GiB |
| KV cache | 57 GiB = 399,419 tokens shared by all requests; 1.52 full 262k contexts |
| layers | 17 attention (51.8 GiB of KV), 48 linear-attention/GDN (33 state slots, 4.8 GiB) |
| state slots | `max(max_num_seqs, 8) x (1 + MTP) + 1` = 33 |
| today, 3 parallel Claude Code sessions | ~250-360 tok/s generated in total, prefill 3-9k tok/s |

## Research summary (tpu-inference / vLLM source, 2026-09-16)

- **Prefix caching** for this hybrid model was broken in 0.28 (a cache hit restored another request's
  GDN state; gsm8k 0.01 vs 0.97, tpu-inference #3358) and is disabled there. Main fixes it (PRs #3422,
  #3550, merged Aug 29 / Sep 10) **only without speculative decoding**: MTP must be off. No PR combines
  them. Main keeps a GDN state snapshot per 16-token block, `custom_mamba_cache_multiplier` blocks per
  request (default 8). At `max_model_len` 262144 the snapshot pool silently shrinks to the minimum unless
  the attention pool is pinned with `--num-gpu-blocks-override` (22000 with bf16 weights).
- **Concurrency**: `max_num_seqs` 8 costs no KV memory (state slots are sized for 8 anyway), only a cold
  compile; keep it a multiple of 4 (GDN kernel out-of-bounds, #3453). 16 costs ~4.7 GiB (~36k tokens).
- **8-bit weights**: v5e has native int8, not fp8. Online `--quantization fp8` on the bf16 checkpoint
  does not load on TPU. Use a pre-quantized int8 W8A8 (compressed-tensors) checkpoint:
  `Avesed/Qwen3.8-27B-INT8-W8A8` (29.1 GiB; lm_head, MTP, vision, small GDN projections stay bf16).
  Expected: ~22 GiB less weights, ~+160k KV tokens. Marked "untested" upstream for this model.
- **FP8 KV cache**: code path exists, reported as a no-op on v5e (#3126). Low priority.

## Tools

- `qwen38-27b/tools/bench_agent.py`: simulated agents resending a growing conversation. Reports TTFT
  per turn (prefill of the history), decode speed, and `recall_accuracy` (each tool result hides a
  value asked for two turns later: a server restoring wrong cached state fails it). Run it with the
  server otherwise idle.
- control command `reconfigure {json}` (`python launch.py cmd 'reconfigure {...}'`, or POST to
  `/ktl/cmd`), `reconfigure` (show), `reconfigure reset`.
- DeployMan env switches on `qwen38-api` for a session that should start directly in a variant:
  `RUNTIME`, `WEIGHTS_MODEL_ID`, `PREFIX_CACHING`, `MAMBA_CACHE_MULTIPLIER`, `NUM_GPU_BLOCKS_OVERRIDE`,
  `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `GPU_MEM_UTIL`, `KV_CACHE_DTYPE`, `QUANTIZATION`.

## Plan for the session (one step at a time, measured)

Bench command for every step (same settings each time):

```bash
python qwen38-27b/tools/bench_agent.py --agents 4 --turns 12 --tool-tokens 3000 --system-tokens 12000 \
    --label <step> --out runs/exp-<step>.json
```

| Step | `reconfigure` JSON | Check in `log` before benchmarking | Success means |
|---|---|---|---|
| 0 baseline | (session start, defaults) | `GPU KV cache size: 399,419 tokens` | reference numbers |
| 1 concurrency | `{"max_num_seqs": 8}` | `--max-num-seqs 8`, no errors after compile | same recall, higher total throughput with `--agents 8` |
| 2 int8 weights | `{"max_num_seqs": 8, "weights_model_id": "Avesed/Qwen3.8-27B-INT8-W8A8"}` | `Init model` HBM ~22 GiB lower; KV cache size ~+160k tokens | recall 1.0, a few real prompts look sane, decode not much slower |
| 3 prefix cache | `{"max_num_seqs": 8, "runtime": "tpu-main-20260915", "prefix_caching": true, "additional_config": {"custom_mamba_cache_multiplier": 3}, "num_gpu_blocks_override": 22000}` (add the int8 weights and a larger override if step 2 passed) | no `Disabling prefix caching`; `TPUHybridKVCacheCoordinator` lines; `Compact-mamba KV cache (align mode) ... _mamba_num_blocks` not 17 | `Prefix cache hit rate` > 50 % during the bench, TTFT of later turns far below baseline, **recall_accuracy 1.0** |

After each step: keep it (next step builds on it) or `reconfigure reset`. Compare end-to-end agent
time (`wall_s`, TTFT per turn), not decode tok/s alone: step 3 turns MTP off (~3x slower decode).

## Risks

- Prefix caching on main is a week old and nightly-only: wrong output is the failure mode, hence the
  recall check. Open issues: #3565 (core halt in align mode, probably not single-host), #3489
  (not enough KV memory at startup in align mode: lower the block override).
- The `tpu-main-20260915` runtime builds vLLM from source (estimated a few minutes); every variant is
  a cold compile (budget 25-40 min). The structured-output sanitizer stays on in every variant.
- int8 W8A8 for Qwen3_5 is untested upstream: check quality on real prompts, not only the bench.
