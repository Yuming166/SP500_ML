# 第二阶段：LLM 异质交易行为与证据溯源路由

## 目标

研究 LLM 是否可以作为可控的异质交易行为生成器，并检验时间有效的证据溯源信号能否识别多 agent 的虚假共识，在市场状态切换时选择更可靠的行为。

核心问题不是“哪个 LLM 预测收益最高”，而是：

> 当多个 agent 因为复用同一条错误、过期或未来泄漏的证据而达成共识时，系统能否识别这种相关错误，并路由到由独立、及时证据支持的 agent，或主动 abstain？

本阶段的研究定义、benchmark 干预和通过条件以 [`asof_provenance_faithfulness.md`](asof_provenance_faithfulness.md) 为准。

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

其中任务约束和决策时间属于 shared context；支持交易结论的特征必须注册为带 `source_id` 和派生关系的 evidence catalog，并按 persona 形成 agent-specific evidence packet。多个 agent 使用同一原始数据派生的不同特征时，仍视为存在共同来源，不能重复计算独立支持。

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
  "decision_time": "2024-03-15T21:00:00Z",
  "action": "cash",
  "target_exposure": 0.0,
  "horizon_days": 5,
  "confidence": 0.0,
  "claims": [
    {
      "claim_id": "c1",
      "text": "...",
      "stance": "supports",
      "evidence_ids": ["e1"]
    }
  ]
}
```

MVP 不让 LLM 自由输出价格、数量或隐藏推理过程；交易数量由环境和风险约束层决定。完整 evidence catalog（`source_id`、时间戳和派生关系）由环境提供并校验，agent 只能引用分配给它的 ID，不能自行编造。

## As-of provenance-faithfulness routing

可信推理模块先构造团队级证据谱系图，再对每个 action 计算可分解的可靠性信号：

```text
source -> evidence -> claim -> agent decision -> routed action
```

trust score 不应只等于 agent 自报 confidence，也不能把 agent agreement 直接当作独立支持。至少包含：

- evidence 是否在 `decision_time` 前可见；
- claim-evidence grounding；
- agent 之间的来源与信息路径独立性；
- 删除、替换或反转证据时 action 是否合理变化；
- confidence calibration on past windows；
- out-of-sample performance under the same regime。

策略路由器可以先从简单规则开始：

```text
exposure = sum_i(normalized_provenance_trust_i * agent_exposure_i)
```

共享 `source_id` 的意见先去重，未来证据直接拒绝；若所有 agent 都依赖同一可疑来源，则输出 `cash/abstain`。再与 static equal-weight、majority vote、agreement weighting、self-confidence 和 recent-performance baseline 比较。最后才考虑 contextual bandit 或 offline RL。

## 虚假共识压力测试

所有干预使用 paired clean/corrupted sample，并保持底层 market state 与预测任务一致：

1. 同一来源经不同措辞复制给多个 agent；
2. 污染所有 agent 共同依赖的来源；
3. 注入过期证据；
4. 注入决策时刻尚不可见的证据；
5. 删除或反转被引用的关键证据；
6. regime shift 后保留只在旧状态成立的证据关系。

优先评价 false-consensus detection、future-evidence violation、AURC、routing regret 和 shared-corruption robustness；收益、回撤和换手作为决策价值的外部验证。

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
| E4 | LLM personas | agreement/consensus weight | 测试群体一致性是否会受共享错误欺骗 |
| E5 | LLM personas | recent-performance/regime router | 测试不使用证据谱系的动态路由 |
| E6 | LLM personas | provenance-aware selective router | 核心方法：去重、时效校验、因果干预与 abstention |
| E7 | LLM personas | contextual bandit/offline RL | 扩展：测试长期动态策略选择 |

所有实验使用相同的 as-of state、交易成本、风险上限和 walk-forward 时间划分。

## 必须控制的风险

1. **LLM 数值能力**：数字计算和价格更新交给 Python 环境，不交给 LLM；
2. **prompt 敏感性**：固定模板、温度、模型版本，并做多次重复；
3. **历史数据污染**：日期、新闻和测试窗口要做 contamination 检查，必要时匿名化日期；
4. **行为漂移**：每个 agent 的 persona 需要做 consistency test；
5. **API 成本**：缓存 state→decision，先用日/周频和小规模 agent population；
6. **模拟器幻觉**：先验证 stylized facts，再解释策略结果；
7. **回测过拟合**：不能用测试期结果挑 persona、prompt、trust 权重或 RL 超参数。
8. **伪独立证据**：不同文本不代表不同来源，必须依据 `source_id` 和派生路径判断；
9. **引用不等于使用**：必须通过 evidence removal/reversal 检验引用证据是否影响 action。

## 简历中的贡献表达

当结果完成后，可以写成：

```text
Built a provenance-aware multi-agent decision benchmark that detects false
consensus caused by shared, stale, or future-leaked evidence and selectively
routes decisions using as-of validity, source independence, evidence grounding,
causal intervention, and historical calibration.
```

如果闭环模拟器也完成，再补充：

```text
Implemented a stylized closed-loop market environment calibrated to empirical
return and volatility statistics, and compared static, majority-vote, and
provenance-aware selective routing under transaction costs.
```
