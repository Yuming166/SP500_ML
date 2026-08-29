# 研究计划：因果多分辨率与可预测性驱动的 S&P 500 学习

## 1. 面向 ECE/CS 的项目主张

项目不把自己包装成“又一个股票预测器”，而是研究一个更一般的机器学习/信号处理问题：

> 对非平稳、多通道、低信噪比时间序列，因果多分辨率表示和状态条件路由能否提升未来窗口预测的可靠性？

金融数据只是实验载体。S&P 500 的价格、VIX、资金流、期权和情绪可以视为不同传感器产生的异步多通道信号。

## 2. 最小可验证假设

设模型在交易日 `t` 收盘后看到截至 `t` 的信息，预测 `t+1` 到 `t+5` 的累计收益。

主假设：

> 当价格、波动率、期权、资金流、情绪和宏观信号彼此冲突时，方向模型的误差和预测区间宽度会增大；因果多分辨率表示配合状态条件路由，能更好地识别这种低可预测性状态。

第二个假设：

> 只在高置信度时输出 forecast，在低置信度时 abstain，可以在相同 coverage 下改善校准和风险表现。

这里的“创新”是一个可复现的系统设计和验证问题，不声称发明了新的小波变换或金融模型。

## 3. 输出定义

每个测试时点输出：

| 字段 | 含义 |
| --- | --- |
| `p_up` | 未来 5 日收益大于 0 的概率 |
| `point_forecast` | 未来 5 日收益点预测 |
| `lower`, `upper` | 由过去校准窗口得到的 conformal 区间 |
| `confidence` | `max(p_up, 1-p_up)` |
| `forecast_status` | `forecast` 或 `abstain` |
| `state` | 低/中/高波动或其他可解释市场状态 |

MVP 只比较 `forecast` 和 `abstain`，不直接加入杠杆、期权或自动交易执行。

## 4. 信号处理前端

### 4.1 先做因果滤波器组

给定一个信号 `x_t`，用只依赖 `x_<=t` 的递归/因果滤波器得到：

- high-frequency detail：`x_t - EMA_fast(x)_t`；
- mid-frequency bands：相邻两个因果平滑结果之差；
- low-frequency trend：最长窗口的因果平滑结果。

这些分量具有可解释的 telescoping reconstruction，可以先验证“滤波器有没有泄漏”和“信息是否被丢失”。

### 4.2 再加入小波对照

固定小波（Haar、Daubechies）或 MODWT 作为对照，不直接宣称创新。需要明确记录边界处理和可见窗口；不能对全量序列先做带未来边界信息的分解，再切 train/test。

### 4.3 自适应频带路由

对每个频带建立轻量 TCN/GRU 分支，使用一个 gate 根据 VIX、跨市场分歧和各分支 hidden state 产生权重：

```text
z_t = concat(h_high, h_mid, h_low, context_t)
gate_t = softmax(g(z_t))
h_t = sum(gate_t[k] * h_k)
```

必须做静态平均融合和无 gate 消融，证明增益来自路由，而不是参数量增加。

## 5. 特征组

1. **价格与波动**：1/5/10 日收益、滚动波动率、回撤、VIX 水平和变化；
2. **宏观与信用**：10Y 收益率、信用利差及其变化；
3. **期权与资金流**：put/call ratio、SPY/IVV/VOO flow、共同基金 flow；
4. **情绪**：recession、inflation、unemployment 的水平和周度变化；
5. **跨市场分歧**：先对各变量做 trailing z-score，再计算统一经济方向上的横截面离散程度。

最后一组是本项目的主解释变量：分歧高并不等于必然下跌，而是预期“方向更难判断”。

## 6. 模型阶梯

按以下顺序增加复杂度，避免一开始无法判断增益来自哪里：

1. 常数概率 / always-invested buy-and-hold 基线；
2. Raw lag features + Logistic Regression；
3. Fixed causal filter bank + Logistic/MLP；
4. Fixed wavelet decomposition + same predictor；
5. Multi-branch TCN/GRU + static fusion；
6. Multi-branch encoder + adaptive gating；
7. calibration + conformal + abstention 层。

所有模型使用 expanding-window walk-forward；不能用随机 train/test split。

## 7. 第二阶段：LLM 异质行为仿真

### 7.1 LLM 的职责边界

LLM 不直接读取长串 OHLCV，也不负责输出精确的未来价格。第一阶段的 encoder 先把输入压缩为结构化 market state：

```text
{regime, p_up, interval_width, band_summary, vix_state,
 cross_market_disagreement, recent_drawdown, flow_state}
```

不同 agent 使用不同、可复现的行为配置：

- trend follower：偏好持续的低频趋势；
- mean reversion：在极端偏离时反向交易；
- volatility targeter：根据波动率调整暴露；
- risk-averse allocator：不确定性过高时持有 cash；
- sentiment/noise trader：对情绪和短期冲击更敏感；
- contrarian：对拥挤交易或信号一致性做反向判断。

每个 agent 必须输出严格 schema：`action`、`target_exposure`、`horizon_days`、`confidence`、`rationale_claims`、`evidence_tags`。不保存或依赖隐藏 chain-of-thought；保存可审计的短 claims 和证据标签即可。

### 7.2 与可信推理研究的接口

把一个交易决策转成：

```text
market state -> claims -> supporting/attacking evidence
             -> claim/evidence confidence -> trust assessment
```

trust assessment 至少包含：

1. **逻辑/证据一致性**：理由是否支持最终 action；
2. **历史校准**：该 agent 的 confidence 是否与真实结果匹配；
3. **行为稳定性**：相同 state 和 persona 下，输出是否过度随机；
4. **跨模型一致性**：换 LLM 或 prompt 后，结论是否保持。

这会把你现有的可信推理经验转化为一个可量化的 agent reliability signal，而不是把“LLM 会解释”当作结果。

### 7.3 仿真层级

分三层实现，避免一开始把问题做成不可验证的宏大市场模拟：

1. **Historical replay**：市场路径外生，比较不同 agent/router 的决策；
2. **Stylized closed-loop simulator**：订单失衡影响价格，外加外生冲击；用真实数据校准厚尾、波动聚集、成交量-波动关系和回撤分布；
3. **Trust-aware router**：根据 market state 和 agent reliability 动态分配 `cash/long` 暴露。

第一层能回答策略是否有用；第二层才回答群体行为是否能产生类似市场的统计现象。不能把第一层的回测结果描述成“证明 LLM agent 改变了市场”。

## 8. RL 的位置：策略层，而不是预测层

RL 不直接从原始价格学习“下一天涨跌”。只有在监督预测器、LLM 行为仿真和不确定性层稳定后，才增加一个小型 offline decision layer：

- state：`p_up`、预测区间宽度、VIX regime、当前持仓、agent trust distribution；
- action：`cash` 或 `long`，先不用连续杠杆；
- reward：净收益 - 交易成本 - 换手惩罚 - 回撤惩罚；
- baseline：固定概率阈值策略、buy-and-hold、简单 volatility targeting、静态 agent mixture。

这样 RL 的问题变成“在不确定性和交易成本下如何决策”，而不是一个难以解释的黑箱价格预测器。金融 RL 已有大量 portfolio/index-tracking 工作，因此 PPO/SAC 本身不能作为项目创新点。[Financial Index Tracking with RL](https://arxiv.org/abs/2308.02820)

## 9. 评价标准

预测层：LogLoss、Brier score、ECE、ROC-AUC 仅作为补充。

不确定性层：名义 80%/90% 区间的实际覆盖率、平均区间宽度、覆盖率-宽度曲线。

选择性层：在 coverage 为 20%、40%、60%、80% 时，比较 selective accuracy、Brier score、平均 forward return 和错误方向造成的回撤。

策略层：加入保守交易成本后，比较年化收益、波动率、Sharpe、最大回撤、换手率；同时报告按年份和 VIX 分位数的结果。

## 10. 通过/不通过条件

- 训练、校准、测试严格按时间分离；
- conformal 区间覆盖率接近名义水平；
- 提升必须在多个时间段成立，而不是只在一个危机窗口成立；
- 若只提高 accuracy 但恶化回撤或校准，不视为成功；
- 若没有稳定优势，保留完整结果并把“市场短期不可预测”作为结论之一。

## 11. CS/ECE 方向的工程证据

- 因果 filter bank 的单元测试和未来扰动测试；
- 统一的模型接口、配置文件、随机种子和数据版本；
- 参数量、训练时间、推理延迟和显存/内存占用对比；
- 每个组件的 ablation，而不只展示最终收益曲线；
- 对 regime shift、缺失值、异步频率和数据发布延迟的处理说明。
