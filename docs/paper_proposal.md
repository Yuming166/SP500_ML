# Paper Proposal: When Consensus Lies

## Working title

**When Consensus Lies: As-of Provenance Faithfulness for Multi-Agent LLM Decisions under Distribution Shift**

备选标题：

**Detecting Shared-Evidence Failures in Regime-Aware LLM Investment Agents**

## 1. 论文主张

大多数金融 LLM agent 工作关注最终收益、角色分工或群体共识，但多个 agent 给出相同答案，并不意味着获得了多份独立证据。它们可能复用了同一条错误、过期或未来泄漏的信息，形成高 confidence 的 **false consensus**。

本文研究 **As-of Provenance Faithfulness**：一个决策是否由在决策时刻真实可见、可溯源、彼此独立，并且对最终 action 具有因果影响的证据支持。系统显式构建：

```text
source -> evidence -> claim -> agent -> action
```

并研究 provenance-aware router 能否在市场状态切换和共享证据污染下，比多数投票、自报 confidence、agreement 和近期表现路由更可靠；当所有 agent 依赖同一可疑来源时，系统应选择 abstain/cash。

金融市场提供真实时间顺序、异步发布时间、状态切换和成本约束，是检验这一问题的压力测试环境。论文中心不是“预测 S&P 500 会涨多少”，也不是泛化的 trust score。完整定义和实验约束见 [`asof_provenance_faithfulness.md`](asof_provenance_faithfulness.md)。

## 2. 适合的投稿层级

- **ACL main/Findings**：需要把方法或 benchmark 讲成一般的 LLM agent reliability / evaluation 问题，金融只是重要测试场景；不能只有收益曲线。
- **FinNLP**：当前主题直接覆盖 financial LLM、synthetic/genuine datasets、benchmarks、evaluation、hallucination mitigation 和 transfer learning，适合作为第一目标。[FinNLP 2026](https://www.aclweb.org/portal/content/finnlp-2026-11th-workshop-financial-technology-and-natural-language-processing)
- **TrustNLP**：如果 trust assessment、解释性、因果时间对齐和 robustness 是主贡献，则很匹配。[TrustNLP 2026](https://www.aclweb.org/portal/content/6th-trustworthy-nlp-workshop-acl-2026)

ACL 2026 的提交窗口已经结束；后续应关注新的 ARR cycle 和 workshop CFP，时间以官方页面为准。[ACL 2026 CFP](https://2026.aclweb.org/calls/main_conference_papers/)

## 3. 论文中的三个核心贡献

### C1. False-consensus provenance benchmark

为每个时间点构造：

```text
as-of market state + persona + agent response + realized outcome
```

为每个 agent 决策记录 source、evidence、claim 和 action 的谱系，并构造 paired clean/corrupted 样本。核心干预包括：

- 同源证据经改写后分发给多个 agent；
- 所有 agent 的共同来源被污染；
- 过期证据和未来尚不可见证据注入；
- 删除或反转 agent 声称依赖的关键证据；
- regime shift 后保留旧状态下才成立的证据关系。

### C2. As-of provenance-faithfulness assessment

把可信度拆为不可相互掩盖的维度：

- `AsOfValid`：证据在决策时刻是否可见；
- `Grounded`：证据是否支持对应 claim；
- `Independent`：多个 agent 是否真正依赖独立信息路径；
- `CausalEffect`：干预证据是否会合理改变 action/confidence；
- `Calibrated`：历史相似状态下的 confidence 是否可信。

未来证据违反是 hard failure，不能被其他高分抵消。Qwen3.5-4B 的 LoRA/QLoRA 用于学习结构化输出、证据引用和 abstention，但不是中心方法贡献；不监督隐藏 chain-of-thought。

### C3. Provenance-aware selective routing

Router 对重复来源降权、拒绝未来证据，并根据 evidence intervention 结果选择最小可信 agent 子集：

```text
w_t = Router(s_t, provenance_graph_t, trust_components_t)
final_exposure_t = sum_i w_i,t * exposure_i,t
```

比较：

1. buy-and-hold；
2. rule-based mixture；
3. majority vote；
4. self-confidence weighting；
5. agreement/consensus weighting；
6. recent-performance/regime-aware routing；
7. provenance-aware routing；
8. contextual bandit/offline RL 扩展。

## 4. 研究任务

| Task | 问题 | 主要指标 |
| --- | --- | --- |
| T1 | agent 能否输出合法结构化决策 | JSON validity、constraint violation |
| T2 | 能否识别虚假共识 | false-consensus AUROC/AUPRC、shared-corruption robustness |
| T3 | 证据是否时间有效 | future-evidence violation、stale-evidence detection |
| T4 | 引用证据是否真正影响 action | intervention sensitivity、claim-evidence alignment |
| T5 | provenance router 是否稳健 | AURC、routing regret、oracle gap、regime-wise return/drawdown |
| T6 | 方法是否迁移到非金融时序决策 | false-consensus transfer、跨域 AURC |

## 5. 最小实验版本

论文第一版不必同时完成所有内容。建议先完成：

1. 第一阶段的 causal multi-resolution encoder；
2. 6 类固定 persona；
3. `cash/long`、5-day horizon；
4. historical replay；
5. base Qwen、LoRA Qwen、规则 agent 三类对照；
6. provenance-aware routing 与 majority/self-confidence/agreement/recent-performance baseline；
7. 至少两个时间外推窗口和多个随机种子。

闭环市场 simulator 和 offline RL 可作为扩展实验；它们不是第一版论文能否成立的前提。若目标是 ACL/NAACL main，优先增加一个带发布时间约束的非金融 temporal decision/QA 迁移实验，而不是先扩展交易模拟器。

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
- 去掉 `AsOfValid`、`Independent`、`CausalEffect`、abstention 和 regime conditioning 的 ablation；
- 年份、VIX regime、high-disagreement/low-disagreement 分组；
- 交易成本、换手和最大回撤；
- 对人工或独立 evaluator 标注的一小批 rationale 做 agreement 检查。
