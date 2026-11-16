# Recurrent MPT + CS8 for BB quantum memory

Research release (2026-11-16). Two-layer, four-head, d=64 recurrent CrossMPT; 12 noisy measurement rounds; parallel DEM fault logits followed by shared CS8. **Training uses fault BCE only.** BB72 and BB90 trained from scratch on 1M samples each; BB144 from scratch and continued to 6M. One training seed per code.

## Circuit noise p=0.005

| Code | Training samples | Test frames | Min-sum+CS8 | Standard BP+CS8 | MPT+CS8 | Relative error reduction vs standard BP |
|---|---:|---:|---:|---:|---:|---:|
| BB72 | 1000000 | 5000 | 30.60% | 28.34% | **24.18%** | 14.7% |
| BB90 | 1000000 | 5000 | 17.28% | 15.52% | **11.52%** | 25.8% |
| BB144 | 6000000 | 10000 | 12.29% | 11.65% | **8.33%** | 28.5% |

Errors are logical frame error rates, lower is better. Every method shares the same test frames within each code/noise point. Four noise strengths, 32000 final test frames in total. Standard BP means sum-product, parallel schedule, max200 iterations; min-sum is reported separately. Same signed zero-centered, prior-scored CS8 with shortcuts disabled. Models are selected by independent validation fault BCE, never by the final test FER. Different codes have different training budgets.

## Report and artifacts

- [Technical report (PDF)](output/pdf/recurrent_mpt_qec_technical_report.pdf), [Markdown](report/technical_report.md), [LaTeX source](report/technical_report.tex)
- [Main results, confidence intervals and paired statistics](results/sum_product/results.json)
- Model weights: `checkpoints/`; circuits and code matrices: `assets/`
- Test sets and packed fault truth: `data/report_test/`; per-frame predictions: `results/sum_product/`
- [Protocol and checkpoint hashes](provenance/protocol.json); training histories and selected sample counts: `provenance/`

## Setup and replay

```bash
python -m pip install -e .
python scripts/fetch_upstream.py
python tests/verify_release.py --device cuda
qec-mpt evaluate --code all --noise all --device cuda --workers 8 --output runs/replay
```

Verification checks truth-to-detector/logical mappings, legal CS8 recovery, and the first8 reference logits produced in an original64-frame BF16 batch. Exact logits may depend on Torch/CUDA/hardware; see `provenance/environment.json`.

## Training

```bash
qec-mpt train --code bb72 --samples 1000000 --schedule two_stage --lr .0003 --seed 21801187 --output runs/bb72 --device cuda
qec-mpt train --code bb90 --samples 1000000 --schedule two_stage --lr .0003 --seed 21801205 --output runs/bb90 --device cuda
qec-mpt train --code bb144 --samples 1000000 --schedule two_stage --lr .0003 --seed 21801259 --output runs/bb144_1m --device cuda
python scripts/resume_training.py --source runs/bb144_1m --output runs/bb144_4m --samples 4000000
python scripts/resume_training.py --source runs/bb144_4m --output runs/bb144_6m --samples 6000000
```

The first100k samples use cosine3e-4 to3e-5, then AdamW is reset at1e-4, halved every200k with floor1e-5. Batch64, EMA.99, gradient clip1, weight decay1e-4;1024 fixed validation examples per noise. Continuation preserves AdamW, EMA and RNG states. An interrupted run resumes using its exact saved command/config. Published inference checkpoints omit optimizer states; regenerate them by training if needed.

## Experimental scope

The experiments use three BB codes with 12 noisy measurement rounds. MPT receives both detector classes and BP operates on the selected single-class detector graph; they share the output fault variables and CS8 recovery constraints. The metric is the logical frame error rate for the tracked commuting logical-observable set. Each code uses one training seed.

## Decoding speed

Ryzen9 9950X (16 cores/32 threads), RTX5070, p=.005; warm full-pipeline measurements including transfers, process IPC and identical CS8. Both batch implementations use32 CPU worker processes. Three repeats;64 samples for single-block latency and512 for batch throughput.

| Code | BP single ms | MPT single ms | BP batch64 ms/block | MPT batch64 ms/block | Batch throughput ratio |
|---|---:|---:|---:|---:|---:|
| BB72 | 33.82 | 46.22 | 2.864 | 0.896 | 3.20x |
| BB90 | 50.76 | 32.90 | 3.971 | 0.775 | 5.13x |
| BB144 | 96.22 | 32.15 | 7.372 | 1.155 | 6.38x |

Batch amortized time measures throughput; multiply by64 for the complete batch service time. Single-block latency and batch throughput are reported separately. MPT uses GPU BF16 inference, BP uses native CPU double precision. These are measurements of the specified implementations and hardware.

[Full timings and protocol](results/speed/results.json), [CSV](results/speed/timings.csv), [figure](report/parallel_throughput.png).

```bash
python scripts/benchmark_speed.py --output runs/speed --workers 4 8 16 32
```

## Sources and attribution

[THIRD_PARTY.md](THIRD_PARTY.md) records upstream projects and rights. CrossMPT runtime sources are pinned and fetched separately. This project adapts CrossMPT and recurrent measurement-history processing, and combines them with CS8; it does not claim those components as original inventions. Only the report-related implementation and artifacts are included; exploratory research directories are excluded.
