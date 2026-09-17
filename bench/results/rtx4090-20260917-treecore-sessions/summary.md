# TreeCore paired-session results

12 fresh sessions / 6 counterbalanced pairs on one GPU.

Values are medians across sessions; effects are medians of paired percentage changes. These differ from the percentage change between the two displayed medians. Positive latency effects mean Rust is slower. Ranges are observed pair ranges, not confidence intervals.

| Concurrency | Metric | Python | Rust | Paired effect | Pair range |
|---|---|---:|---:|---:|---:|
| 8 | e2e_p50 | 260.53 | 260.78 | -0.33% | -1.16% to +3.06% |
| 8 | ttft_p50 | 74.79 | 74.97 | -0.35% | -4.23% to +13.84% |
| 8 | itl_p50 | 5.11 | 5.09 | -0.69% | -1.43% to +0.15% |
| 8 | itl_p99 | 44.06 | 44.07 | +0.08% | -0.25% to +0.47% |
| 8 | stream_span_p50 | 180.04 | 180.10 | +0.06% | -0.31% to +0.35% |
| 8 | deltas_per_request | 10.25 | 10.25 | +0.00% | +0.00% to +0.00% |
| 8 | requests_per_s | 30.56 | 30.49 | +0.15% | -2.94% to +1.08% |
| 8 | cpu_percent | 207.28 | 204.46 | -1.50% | -39.33% to +72.53% |
| 64 | e2e_p50 | 630.53 | 623.65 | -1.03% | -5.38% to +6.07% |
| 64 | ttft_p50 | 230.84 | 228.26 | +1.23% | -4.53% to +3.22% |
| 64 | itl_p50 | 11.58 | 11.31 | -2.07% | -8.00% to +8.42% |
| 64 | itl_p99 | 92.49 | 91.79 | -1.84% | -6.84% to +15.40% |
| 64 | stream_span_p50 | 385.27 | 375.92 | -2.90% | -7.07% to +11.48% |
| 64 | deltas_per_request | 10.38 | 10.38 | +0.00% | +0.00% to +0.00% |
| 64 | requests_per_s | 99.50 | 100.56 | +0.95% | -5.57% to +5.54% |
| 64 | cpu_percent | 151.01 | 151.23 | +0.15% | -0.72% to +2.09% |
| 256 | e2e_p50 | 1153.75 | 1168.88 | +2.38% | -2.40% to +6.51% |
| 256 | ttft_p50 | 636.56 | 643.26 | +1.74% | -2.68% to +5.20% |
| 256 | itl_p50 | 24.43 | 23.64 | -2.54% | -12.76% to +7.84% |
| 256 | itl_p99 | 123.11 | 122.51 | -1.61% | -12.94% to +14.86% |
| 256 | stream_span_p50 | 503.90 | 520.69 | +2.05% | -3.96% to +9.58% |
| 256 | deltas_per_request | 10.40 | 10.40 | +0.00% | +0.00% to +0.00% |
| 256 | requests_per_s | 211.36 | 207.52 | -2.81% | -5.99% to +3.70% |
| 256 | cpu_percent | 205.36 | 205.80 | +0.55% | -0.67% to +1.16% |

Latency units: milliseconds. CPU: percent of one core. Throughput: requests/second. Delta count: meaningful tool-call deltas/request.

| Pair | Order | c=8 e2e effect | c=64 e2e effect | c=256 e2e effect |
|---|---|---:|---:|---:|
| 0 | rust → python | -1.16% | +0.99% | +3.57% |
| 1 | python → rust | +0.89% | -5.34% | -2.40% |
| 2 | rust → python | -0.62% | +6.07% | +1.18% |
| 3 | python → rust | +3.06% | -0.11% | -2.39% |
| 4 | rust → python | -0.08% | -1.95% | +6.51% |
| 5 | python → rust | -0.58% | -5.38% | +5.50% |
