# 循环 MPT＋CS8 的 BB 码量子纠错解码

本仓库提供 BB72（6 轮，4 块 MPT）与 BB144（12 轮，6 块 MPT）的最终模型。两者均使用 96 维隐变量、共同故障连边，以及按绝对 LLR 排序的混合评分 CS8。

| 模型 | 训练样本 | 测试逻辑块失败率，p=0.005 |
| --- | ---: | ---: |
| [BB72](checkpoints/bb72_r6.pt) | 4000 万 | 7.244%（3,622 / 50,000） |
| [BB144](checkpoints/bb144.pt) | 6000 万 | 2.351%（1,058 / 45,000） |

- [技术报告 PDF](output/pdf/recurrent_mpt_qec_technical_report.pdf)
- [同帧基线对照原始计数](results/final_p005.json)
- [电路和上游代码来源](THIRD_PARTY.md)

## 复现

在仓库根目录运行。CUDA 评估使用 BF16，与报告中的准确率评估口径一致。CPU 使用 FP32，小批次可用于安装检查。

```bash
python -m pip install -e .
python scripts/fetch_upstream.py
qec-mpt evaluate --code bb72_r6 --device cuda
qec-mpt evaluate --code bb144 --device cuda
```

默认评估对照表中的 3,000 帧（从零基下标 5000 开始）。BB72 大样本评估用 `--start 0 --shots 50000`，BB144 用 `--start 5000 --shots 45000`。逐帧对照结果在 `results/*_paired.npz`。换噪声强度时可使用 `--dataset fresh --noise 0.004` 重新采样。

用 XeLaTeX 重建报告时运行 `python scripts/build_report_pdf.py`。
