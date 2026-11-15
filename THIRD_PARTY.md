# 来源与第三方材料

## CrossMPT

原仓库：https://github.com/iil-postech/crossmpt

固定提交：`ecf1ffc5b9ddaf1f17ba786c15119abd2ecf4670`。

运行时需要`Model_CrossMPT.py`和`Codes.py`。配置中的固定URL和规范化LF哈希记录于`configs/upstream.json`。下载脚本不修改源码；下载目录不进入Git。本地原始快照未附顶层LICENSE，因此本项目不为其推定开源许可，也不将其源码纳入本仓库重新分发。使用和进一步分发应核对上游权利及作者要求。

## BB电路与码数据

电路生成来源：https://github.com/jblue1/ml_bb_decoding ，固定提交`c6c97a9e23918fe561c54e916ebadd396897c728`。

论文：Blue et al., Machine Learning Decoding of Circuit-Level Noise for Bivariate Bicycle Codes, Quantum10,2149(2026)。本仓库保留研究生成的Stim电路和矩阵，明确其来源，不包含作者整个项目或将其代码重新授权。BB72/144使用原公开逻辑算子；BB90采用同一调度方法推广和独立验证的逻辑算子。矩阵与电路生成数据不等同于本项目独创的码构造。

## 软件和其他思想来源

PyTorch、NumPy、SciPy、Numba、Stim和LDPC通过包管理器安装，许可证遵循各项目。AlphaQubit仅作为循环时序建模思想参考，本仓库未包含其代码。报告中列出相关论文和贡献边界。

本项目自有部分暂未指定统一开源许可证；公开可读不等于自动授予任意再许可权。仓库所有者可在核实权利和个人意愿后选择自有代码许可证。
