# 简历与 SOP 定位

## 建议项目标题

**Causal Multi-Resolution Representation Learning for Selective S&P 500 Forecasting**

中文可以理解为：面向选择性预测的 S&P 500 因果多分辨率表示学习。

## 简历 bullet 的目标形态

等模型和实验完成后，建议写成“方法 + 验证”，不要只写“预测股票价格”：

```text
Designed a causal multi-resolution filter-bank front-end for asynchronous
market, volatility, flow, and sentiment signals, with strict walk-forward
validation to prevent look-ahead leakage.
```

```text
Built a multi-branch temporal encoder with state-conditioned frequency-band
gating and calibrated abstention; evaluated robustness across volatility
regimes using Brier score, calibration error, interval coverage, and drawdown.
```

```text
Implemented reproducible ablations against raw-lag, fixed-filter, wavelet,
and static-fusion baselines; reported accuracy, uncertainty quality, latency,
and transaction-cost-aware out-of-sample performance.
```

只有在 RL 层完成严格的成本、基线和 walk-forward 对照后，才加入：

```text
Added an offline cash/long decision policy conditioned on predictive
uncertainty and volatility regime, with transaction-cost and turnover penalties.
```

## 面试时的核心解释

一句话版本：

> I treated financial variables as asynchronous, non-stationary sensor signals and studied whether causal multi-resolution representations can help a model recognize both predictable and unpredictable market states.

不要把主要卖点说成“我用了 LSTM、Transformer 或 PPO”。这些是实现工具；申请 CS/ECE 时更重要的是因果性、表示、路由、校准、消融和可复现评估。

