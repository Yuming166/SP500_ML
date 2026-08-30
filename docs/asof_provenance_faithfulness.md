# 研究主线：As-of Provenance Faithfulness

## 1. 一句话主张

> 多个 LLM agent 的一致意见不等于可靠；只有当它们的决策由彼此独立、在决策时刻确实可见、能够支持对应 claim，且对 action 具有因果影响的证据产生时，共识才值得信任。

本文暂称这一性质为 **As-of Provenance Faithfulness（时间有效的证据溯源忠实性）**。核心现象是 **false consensus**：多个 agent 看似独立地同意，实际上复用了同一条错误、过期或未来泄漏的证据，因此多数投票、自报 confidence 和普通 agreement score 都会同时失效。

推荐工作标题：

**When Consensus Lies: As-of Provenance Faithfulness for Multi-Agent LLM Decisions under Distribution Shift**

金融版副标题可使用：

**Detecting Shared-Evidence Failures in Regime-Aware LLM Investment Agents**

## 2. 研究缺口与边界

已有工作分别研究了多 agent 交易与共识、单 agent 推理忠实性、confidence calibration、拒答和模型路由。本项目不把“多个金融 agent”“反事实测试”“confidence routing”或“使用强化学习”单独作为创新点，而研究它们之间尚未被充分建模的连接：

1. 多 agent 的错误往往不是独立错误，而是由共享证据源造成的相关错误；
2. 金融证据具有 `event_time`、`publication_time` 和 `available_at`，事实正确但当时不可见的证据仍构成不忠实决策；
3. rationale 中引用某条证据，不代表该证据真正影响了 action；
4. regime shift 会改变证据的有效性，但不会自动体现在 agent 的 confidence 或群体 agreement 中。

因此，论文的主贡献应是团队级、时间约束的证据溯源评估与 anti-herding 路由。Qwen3.5-4B 微调、金融信号处理和 offline RL 都是实现与验证组件，不是中心 novelty。

更稳妥的 novelty 表述是：**据现有相关工作定位，本项目研究的是 agreement、faithfulness、temporal validity 与 routing 的交叉缺口，而不是声称其中任一单项从未出现。** 投稿前仍需围绕 multi-agent evidence provenance、correlated errors、source-aware routing 和 temporal faithfulness 做一次系统检索，并据结果收窄 claim。

## 3. 证据谱系图

每个决策都显式记录如下有向图：

```text
source -> evidence item -> claim -> agent decision -> routed action
```

节点至少包含：

- `source_id`：原始数据、公告、指标或派生信号的唯一来源；
- `event_time`：事件实际发生时间；
- `publication_time`：来源公开时间；
- `available_at`：系统最早能够读取该信息的时间；
- `evidence_id`：带版本的数据或文本证据；
- `claim_id`：agent 给出的可检查短 claim；
- `agent_id`、`action` 和 `decision_time`。

Agent 输出契约建议扩展为：

```json
{
  "agent_id": "trend_01",
  "decision_time": "2024-03-15T21:00:00Z",
  "action": "cash",
  "target_exposure": 0.0,
  "horizon_days": 5,
  "confidence": 0.72,
  "claims": [
    {
      "claim_id": "c1",
      "text": "波动率上升且趋势信号转弱",
      "stance": "supports",
      "evidence_ids": ["e_vix_1", "e_trend_1"]
    }
  ],
  "evidence": [
    {
      "evidence_id": "e_vix_1",
      "source_id": "cboe_vix_close",
      "event_time": "2024-03-15T20:00:00Z",
      "publication_time": "2024-03-15T20:15:00Z",
      "available_at": "2024-03-15T20:16:00Z",
      "value": 14.4
    }
  ]
}
```

不要求模型输出或保存隐藏 chain-of-thought；只监督可审计的 claim、证据引用和最终动作。

### 3.1 共享状态与独立证据包

所有 agent 可以共享不参与“独立票数”计算的任务上下文，例如决策时间、交易限制和当前持仓；但用于支持 action 的信息应通过 agent-specific evidence packet 提供：

```text
shared context: decision_time, current exposure, risk constraints

trend agent packet: price-derived trend evidence
volatility agent packet: VIX and realized-volatility evidence
flow agent packet: ETF/fund-flow evidence
sentiment agent packet: timestamped sentiment evidence
```

独立性按来源图而不是文本差异判断。两个指标即使名称不同，只要由同一个原始时间序列或同一篇报道派生，就必须保留共同祖先，不能被计算成两份独立支持。干预实验再有控制地复制、污染或交换这些 packet。

## 4. 可信度定义

对 agent `i` 在时刻 `t` 的决策，分别计算：

- `AsOfValid`：证据在 `decision_time` 前是否真实可得；
- `Grounded`：claim 是否由引用证据支持，而非仅在措辞上相关；
- `Independent`：相对于其他 agent，证据是否来自独立来源或独立信息路径；
- `CausalEffect`：删除、替换或反转证据后，action/confidence 是否发生方向合理的变化；
- `Calibrated`：该 agent 在相似历史状态下的 confidence 是否与实际错误率匹配。

其中前四项定义 provenance faithfulness，历史 calibration 作为路由器的补充信号。一个可解释的起始分数为：

```text
PF_i,t = AsOfValid_i,t
         * Grounded_i,t
         * Independent_i,t
         * CausalEffect_i,t

Trust_i,t = PF_i,t * Calibrated_i,t
```

正式方法可以学习组合函数，但必须保留逐项分数和 hard constraint：出现未来证据时不得用其他高分抵消。

## 5. 核心 benchmark：制造“虚假共识”

保持 market state 和最终任务不变，只干预证据谱系：

| 干预 | 构造方式 | 希望检测的失败 |
| --- | --- | --- |
| Source duplication | 将同一来源改写后分发给多个 agent | 把相关意见误当独立共识 |
| Shared corruption | 同时污染所有 agent 依赖的共同来源 | agreement 很高但群体共同出错 |
| Stale evidence | 用过期但表面合理的数据替换当前证据 | 忽略证据时效性 |
| Future leakage | 注入决策时刻尚未发布的信息 | 用未来信息制造虚假高可信度 |
| Evidence removal | 删除 agent 声称依赖的关键证据 | rationale 与真实决策机制脱节 |
| Evidence reversal | 反转关键证据方向但保留其他输入 | action 对证据不敏感或反应错误 |
| Regime invalidation | 保留旧 regime 中有效的证据关系 | 状态变化后仍盲目沿用旧规律 |

每种干预都需要 paired clean/corrupted sample，使检测结果不被市场难度差异混淆。

## 6. 路由方法

Router 不只选择“历史收益最高”或“confidence 最大”的 agent，而选择由独立、及时且具有因果作用的证据支持的最小 agent 子集：

```text
agent decisions + provenance graph + market regime
    -> false-consensus detector
    -> provenance-faithfulness scores
    -> subset selection / weighting / abstention
    -> final cash-or-long exposure
```

关键机制：

1. 对共享 `source_id` 或高度相似的信息路径去重，降低伪多样性权重；
2. 对 `available_at > decision_time` 的证据执行 hard rejection；
3. 在 routing 前做 evidence intervention，估计 claim/action 对关键证据的敏感性；
4. 当所有 agent 都依赖同一可疑来源时，选择 `abstain/cash`，而不是服从多数；
5. 在 regime shift 后提高独立性和时效性约束，逐步重估历史 calibration。

第一版用规则或可解释的监督模型实现。Contextual bandit/offline RL 只作为后续策略优化扩展，并与同一可信信号的非 RL router 公平比较。

## 7. 对照、指标与消融

必须比较：

1. majority vote；
2. self-confidence weighting；
3. agreement/consensus weighting；
4. recent-performance router；
5. regime-aware router；
6. provenance-aware router；
7. 能看到真实结果的 oracle router，仅用于计算上界和 routing regret。

核心指标优先于收益曲线：

- false-consensus detection AUROC/AUPRC；
- high-confidence error rate；
- risk-coverage / AURC；
- future-evidence violation rate；
- shared-corruption robustness drop；
- routing regret 与 oracle gap；
- 分 regime 的 calibration、return、drawdown 和 turnover。

关键消融：去掉 `AsOfValid`、`Independent`、`CausalEffect`、provenance graph、abstention 和 regime conditioning，并单独报告每项对 false consensus 的贡献。

## 8. 最小可发表闭环

第一版只需完成以下闭环：

1. 从第一阶段信号生成严格 as-of 的结构化 market state 和证据对象；
2. 让 4–6 个 Qwen/规则 agent 输出可审计决策；
3. 构造至少五类 paired provenance perturbation；
4. 训练或设计 false-consensus detector 与 provenance-aware router；
5. 在多个 walk-forward test window 和 regime shift 上评估；
6. 证明方法在共享污染下优于 agreement、confidence 和 recent-performance routing；
7. 在所有 agent 的证据均不可信时，验证 abstention 能降低高置信错误与回撤。

如果只能提升收益，却不能识别共享证据错误，这个核心假设不成立；如果检测能力提升但收益不稳定，仍可作为可信 NLP/agent evaluation 的有效结果。

## 9. 与项目其他模块的关系

- 金融信号处理负责生成不泄漏、带发布时间的 evidence，提供严格测试环境；
- Qwen3.5-4B SFT 负责学习 schema、证据引用和不确定时 abstain；
- provenance-faithfulness evaluator 和 false-consensus benchmark 是论文核心；
- router 验证可信信号是否具有决策价值；
- RL 只研究如何在成本和风险约束下长期使用这些信号。

后续实现和实验必须以本文件为研究指导。若某个新模块不能改善 false-consensus detection、provenance faithfulness 或可靠路由中的至少一项，就不应进入论文主线。

## 10. ACL/NAACL 级别的最低要求

金融收益只证明方法在一个应用中有用，不足以证明一般的 NLP/agent 贡献。主会版本应至少做到：

1. 把任务定义成领域无关的 multi-agent evidence provenance 与 false-consensus detection；
2. 公开可复现的干预生成器、证据谱系 schema 和人工核验子集；
3. 除 S&P 500 外，增加一个具有发布时间约束的非金融决策或 temporal QA 小规模迁移实验；
4. 证明检测提升来自 provenance/causal intervention，而不是参数量、同一 judge LLM 或事后收益标签；
5. 报告失败案例：独立来源同时错误、来源关系缺失、agent 忽略 evidence packet，以及过度 abstention。
