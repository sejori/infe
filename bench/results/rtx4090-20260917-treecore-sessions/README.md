# TreeCore follow-up — 2026-09-17

The original approximately 11–12% end-to-end regression was **not reproduced**
under this warmed, counterbalanced protocol. The new data do not establish
equivalence or rule out a smaller regression at concurrency 256. They provide
no demonstrated optimisation opportunity for `infe-kv` and no evidence for the
previous CPU-saving/blocking explanation.

Recommendation: close SGLang issue
[#38536](https://github.com/sgl-project/sglang/issues/38536) as unconfirmed / not
reproduced under the follow-up protocol, preserving both datasets and reopening
if a controlled reproduction or profile identifies a backend effect. Do not
describe the original observation as disproven, fixed, or caused by temperature.
No upstream comment or issue-state change was made as part of this run.

## Design

- Hardware: one RTX 4090 on a shared host. No other workload was stopped.
- SGLang v0.5.19, resolved to immutable image ID and digest in `manifest.json`.
  Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`,
  stock `qwen` tool parser, identical configuration except TreeCore backend.
- Six preselected pairs, twelve fresh containers, three Python-first and three
  Rust-first pairs. Exact schedule and harness hashes are in the manifest.
- Per session: concurrency 8, 64, 256; two discarded warmup rounds at each
  level, then three measured rounds. 11,808 measured requests and 7,872 warmup
  requests all passed HTTP/tool-call structural checks. Delta counts matched.
- Container cgroup CPU counter differences bracket each measured round. GPU
  telemetry is sampled within each measured level. Session medians are compared
  within pairs; neither individual requests nor rounds are independent replicates.

This changes the original warmup protocol and fixes the server random seed.
It tests warmed short-prefix service, not the original mix of cold/warm states
or a long-prefix/cache-pressure workload. It uses the same GPU model, not a
verified identical physical GPU to Round 6. The findings are scoped accordingly.

## Results

Positive effects mean higher latency for Rust. Each effect is the median of six
paired percentage changes, not the percentage change between overall medians.
Ranges are observed ranges, not confidence intervals.

| Concurrency | E2E p50 effect | E2E pair range | TTFT p50 effect | Stream-span effect |
|---|---:|---:|---:|---:|
| 8 | −0.33% | −1.16% to +3.06% | −0.35% | +0.06% |
| 64 | −1.03% | −5.38% to +6.07% | +1.23% | −2.90% |
| 256 | +2.38% | −2.40% to +6.51% | +1.74% | +2.05% |

Rust's E2E p50 was slower in two of six pairs at concurrency 64 and four of six
at 256. This is substantially less consistent than the original report suggested.
Six pairs are a small sample; no significance or equivalence claim is made.

At concurrency 64/256, median paired CPU differences were +0.15%/+0.55%, with
ranges −0.72% to +2.09% / −0.67% to +1.16%. These do not support the earlier 6%
CPU-saving claim. Concurrency-8 CPU was highly variable across sessions
(paired range −39.33% to +72.53%); its subsecond measurement rounds are unsuitable
for a strong CPU conclusion without longer steady measurement windows.

All 84 GPU telemetry samples succeeded. Sampled SM/memory clocks were constant
at 2610/10251 MHz; sampled temperatures ranged 35–47°C. This does not show clock
drift explaining the differences, and cannot diagnose the original run. Host
one-minute load averages ranged 4.70–8.11. Shared-host interference and the short
measurement windows remain limitations.

## Artifacts and verification

- `manifest.json`: fixed schedule, configuration, image digest, GPU and source hashes.
- `pair-*.json.gz`: losslessly compressed raw session data, including warmups,
  request timings, CPU counters/windows and GPU telemetry.
- `pair-*.log.gz`: losslessly compressed server logs; `run.log`: runner output.
- `comparison.json`: paired effects; `summary.md`: all metrics and per-pair E2E effects.

All request-level summaries were recomputed from raw rows, all session counts
and concurrency levels checked, and `comparison.json` exactly reproduced from
the session files. The recorded source hashes identify the harness version used for collection;
the current harness additionally omits machine identity metadata. All twelve server logs confirm the intended Python/Rust TreeCore class.
Every experiment container was removed and the benchmark GPU was idle after completion.

From the repository root, regenerate the table with:

```bash
python3 bench/harness/summarize_treecore.py \
  bench/results/rtx4090-20260917-treecore-sessions
```

The five harness tests cover balanced schedules, paired aggregation, incomplete
archives, invalid tool-call responses, and exclusion of legacy whole-run CPU
from concurrency rows. They passed locally and are included in CI. Two additional tests cover metadata
privacy and log redaction.

Machine identifiers, SSH identity and local paths were removed after collection.
The recorded measurements and paired comparison are unchanged.
