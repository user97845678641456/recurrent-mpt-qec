## Decoding speed

Ryzen9 9950X (16 cores/32 threads), RTX5070, p=.005; warm full-pipeline measurements including transfers, process IPC and identical CS8. Both batch implementations use32 CPU worker processes. Three repeats;64 samples for single-block latency and512 for batch throughput.

| Code | BP single ms | MPT single ms | BP batch64 ms/block | MPT batch64 ms/block | Batch throughput ratio |
|---|---:|---:|---:|---:|---:|
| BB72 | 33.82 | 46.22 | 2.864 | 0.896 | 3.20x |
| BB90 | 50.76 | 32.90 | 3.971 | 0.775 | 5.13x |
| BB144 | 96.22 | 32.15 | 7.372 | 1.155 | 6.38x |

Batch amortized time measures throughput; multiply by64 for the complete batch service time. Single-block latency and batch throughput are reported separately. MPT uses GPU BF16 inference, BP uses native CPU double precision. These are measurements of the specified implementations and hardware.

[Full timings and protocol](results.json), [CSV](timings.csv), [figure](../../report/parallel_throughput.png).

```bash
python scripts/benchmark_speed.py --output runs/speed --workers 4 8 16 32
```
