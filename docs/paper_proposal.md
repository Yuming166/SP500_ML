# Paper Proposal: FinTrustSim

## Working title

**FinTrustSim: Reliability-Calibrated LLM Agents for Regime-Aware Financial Market Simulation**

备选标题：

**Trust-Aware Routing of Heterogeneous LLM Investors under Temporal Distribution Shift**

## 1. 论文主张

大多数金融 LLM agent 工作关注最终收益或交易流程，但一个 agent 是否值得被采纳，还应由以下问题决定：

1. 它能否保持自己的行为设定；
2. 它的 confidence 是否经过校准；
3. 它给出的 claims 是否真正支持 action；
4. 市场状态发生变化后，它是否知道自己不可靠。

本文把这些属性统一为 **agent reliability**，并研究 reliability-aware routing 是否比静态混合、简单多数投票和自报 confidence 更稳健。

金融市场只是一个有真实时间顺序、状态切换和成本约束的 agent evaluation environment。论文的中心对象是可靠的 LLM agent 与动态路由，而不是“预测 S&P 500 会涨多少”。

## 2. 适合的投稿层级

- **ACL main/Findings**：需要把方法或 benchmark 讲成一般的 LLM agent reliability / evaluation 问题，金融只是重要测试场景；不能只有收益曲线。
- **FinNLP**：当前主题直接覆盖 financial LLM、synthetic/genuine datasets、benchmarks、evaluation、hallucination mitigation 和 transfer learning，适合作为第一目标。[FinNLP 2026](https://www.aclweb.org/portal/content/finnlp-2026-11th-workshop-financial-technology-and-natural-language-processing)
- **TrustNLP**：如果 trust assessment、解释性、因果时间对齐和 robustness 是主贡献，则很匹配。[TrustNLP 2026](https://www.aclweb.org/portal/content/6th-trustworthy-nlp-workshop-acl-2026)

ACL 2026 的提交窗口已经结束；后续应关注新的 ARR cycle 和 workshop CFP，时间以官方页面为准。[ACL 2026 CFP](https://2026.aclweb.org/calls/main_conference_papers/)

## 3. 论文中的三个核心贡献

### C1. 金融 agent reliability benchmark

为每个时间点构造：

```text
as-of market state + persona + agent response + realized outcome
```

评估四种可靠性：

- schema/constraint validity；
- persona consistency；
- confidence calibration；
- rationale-action/evidence consistency。

加入 counterfactual perturbation：只改变 VIX、低频趋势或跨市场分歧中的一个因素，测试 agent 是否做出方向合理且幅度可解释的变化。

### C2. Reliability-aware structured generation

使用 Qwen3.5-4B 作为可复现的开源 agent backbone，通过 LoRA/QLoRA 学习：

- 稳定输出结构化 action；
- 遵守 persona 和风险约束；
- 在证据冲突时降低 confidence 或选择 abstain；
- 用短 claims 和 evidence tags 表达可审计依据。

不训练模型去背诵未来价格，也不把隐藏 chain-of-thought 当作监督目标。

### C3. Trust-aware strategy routing

设第 `i` 个 agent 的决策为 `a_i,t`，trust assessment 为 `q_i,t`，router 根据市场状态 `s_t` 产生权重：

```text
w_t = Router(s_t, q_1,t, ..., q_n,t)
final_exposure_t = sum_i w_i,t * exposure_i,t
```

比较：

1. buy-and-hold；
2. rule-based mixture；
3. majority vote；
4. self-confidence weighting；
5. trust-aware weighting；
6. contextual bandit/offline RL。

## 4. 研究任务

| Task | 问题 | 主要指标 |
| --- | --- | --- |
| T1 | agent 能否输出合法结构化决策 | JSON validity、constraint violation |
| T2 | persona 是否稳定 | persona adherence、repeated-state consistency |
| T3 | confidence 是否可信 | ECE、Brier、selective risk |
| T4 | 理由是否支持 action | claim-action consistency、evidence alignment |
| T5 | trust-aware router 是否稳健 | regime-wise return、drawdown、turnover |
| T6 | 闭环 simulator 是否像市场 | heavy tails、volatility clustering、回撤分布 |

## 5. 最小实验版本

论文第一版不必同时完成所有内容。建议先完成：

1. 第一阶段的 causal multi-resolution encoder；
2. 6 类固定 persona；
3. `cash/long`、5-day horizon；
4. historical replay；
5. base Qwen、LoRA Qwen、规则 agent 三类对照；
6. trust-aware routing 与 majority/self-confidence baseline；
7. 至少两个时间外推窗口和多个随机种子。

闭环市场 simulator 和 offline RL 可作为扩展实验；它们不是第一版论文能否成立的前提。

## 6. 不能作为主要贡献的内容

- “用了小波变换”；
- “用了 PPO/SAC”；
- “让 LLM 扮演牛派和熊派”；
- 一条看起来很好的累计收益曲线；
- 只用同一个 LLM 评价自己生成的 rationale。

这些可以是组件或 baseline，但不能单独支撑 ACL 级别的 novelty。

## 7. 必须做的可信实验

- chronological train/validation/test split；
- future perturbation test：修改未来数据不能影响过去 agent response；
- prompt/model seed sensitivity；
- base model vs SFT vs preference-tuned；
- 去掉 claims、去掉 trust、去掉 market-state encoder 的 ablation；
- 年份、VIX regime、high-disagreement/low-disagreement 分组；
- 交易成本、换手和最大回撤；
- 对人工或独立 evaluator 标注的一小批 rationale 做 agreement 检查。

