# 第二阶段：LLM 异质交易行为与可信策略路由

## 目标

研究 LLM 是否可以作为可控的异质交易行为生成器，并检验可信推理信号能否帮助策略系统在市场状态切换时选择更可靠的行为。

核心问题不是“哪个 LLM 预测收益最高”，而是：

> 对同一个 market state，具有不同风险偏好、时间视野和证据使用习惯的 agent 会如何决策？这些 agent 的可靠性是否可以被测量，并用于动态策略路由？

## 与第一阶段的连接

第一阶段输出结构化状态：

```text
market_state = {
    "regime": "low_vol" | "high_vol" | "transition",
    "p_up": 0.0..1.0,
    "interval_width": float,
    "band_summary": {...},
    "cross_market_disagreement": float,
    "drawdown": float,
}
```

LLM agent 不需要处理原始长序列。这样可以减少数值幻觉、token 成本和 prompt 对结果的干扰，也能把信号处理和 agent reasoning 的贡献分开做 ablation。

## Agent 设计

建议先做 6 类 agent，每类固定 system instruction 和风险约束：

| Agent | 行为假设 | 主要关注 |
| --- | --- | --- |
| Trend | 相信趋势延续 | 低频 trend、动量 |
| Reversion | 极端偏离后均值回归 | 高低频差、回撤 |
| Volatility targeter | 目标是控制波动 | VIX、区间宽度 |
| Risk averse | 不确定时持有 cash | calibration、drawdown |
| Sentiment | 对情绪和资金流敏感 | sentiment、flow |
| Contrarian | 对拥挤信号反向 | cross-market disagreement |

每个 agent 的输出必须是机器可解析的 JSON：

```json
{
  "agent_id": "trend_01",
  "action": "cash" | "long",
  "target_exposure": 0.0,
  "horizon_days": 5,
  "confidence": 0.0,
  "rationale_claims": ["..."],
  "evidence_tags": ["low_frequency_trend", "vix_state"]
}
```

MVP 不让 LLM 自由输出价格、数量或隐藏推理过程；交易数量由环境和风险约束层决定。

## Trust-aware routing

你的可信推理模块可以对每个 action 计算独立的可靠性信号：

```text
agent decision
    -> claims
    -> supporting / attacking evidence
    -> claim and evidence confidence
    -> trust score
```

trust score 不应只等于 agent 自报 confidence。至少包含：

- rationale-action consistency；
- confidence calibration on past windows；
- repeated-state behavioral consistency；
- out-of-sample performance under the same regime；
- cross-model/prompt agreement。

策略路由器可以先从简单规则开始：

```text
exposure = sum_i(normalized_trust_i * agent_exposure_i)
```

再与 static equal-weight、majority vote 和 probability-threshold baseline 比较。最后才考虑 contextual bandit 或 offline RL。

## 两种模拟环境

### A. Historical replay

市场价格路径固定，agent 的交易不会反过来改变价格。它适合回答：

- 哪种 agent/routing 在历史 out-of-sample 上更稳健；
- trust score 是否能识别错误决策；
- 交易成本和 abstention 是否改变结论。

它不能证明 agent 造成了市场价格变化。

### B. Calibrated closed-loop simulator

定义一个简单、可解释的价格更新过程：

```text
return_t = exogenous_shock_t + impact(order_imbalance_t)
```

让 agent 的订单影响 `order_imbalance`，并使用真实数据校准以下 stylized facts：

- heavy-tailed returns；
- volatility clustering；
- near-zero raw-return autocorrelation；
- volume-volatility relationship；
- drawdown/recovery distribution。

只在模拟器能复现这些统计性质后，才讨论群体行为和市场形成机制。

## 实验矩阵

| 实验 | Agent | Router | 目的 |
| --- | --- | --- | --- |
| E0 | 无 | buy-and-hold | 市场基线 |
| E1 | 规则策略 | static equal weight | 非 LLM 行为基线 |
| E2 | LLM personas | majority vote | 测试 LLM 行为本身 |
| E3 | LLM personas | self-confidence weight | 测试自报置信度 |
| E4 | LLM personas | trust-aware weight | 测试可信推理信号 |
| E5 | LLM personas | contextual bandit/offline RL | 测试动态策略选择 |

所有实验使用相同的 as-of state、交易成本、风险上限和 walk-forward 时间划分。

## 必须控制的风险

1. **LLM 数值能力**：数字计算和价格更新交给 Python 环境，不交给 LLM；
2. **prompt 敏感性**：固定模板、温度、模型版本，并做多次重复；
3. **历史数据污染**：日期、新闻和测试窗口要做 contamination 检查，必要时匿名化日期；
4. **行为漂移**：每个 agent 的 persona 需要做 consistency test；
5. **API 成本**：缓存 state→decision，先用日/周频和小规模 agent population；
6. **模拟器幻觉**：先验证 stylized facts，再解释策略结果；
7. **回测过拟合**：不能用测试期结果挑 persona、prompt、trust 权重或 RL 超参数。

## 简历中的贡献表达

当结果完成后，可以写成：

```text
Built a trust-aware multi-agent market simulation in which LLM personas
generated structured trading decisions from causal multi-resolution market
states; calibrated agent reliability from claim-level evidence consistency,
confidence calibration, and out-of-sample behavior.
```

如果闭环模拟器也完成，再补充：

```text
Implemented a stylized closed-loop market environment calibrated to empirical
return and volatility statistics, and compared static, majority-vote, and
trust-aware strategy routing under transaction costs.
```

