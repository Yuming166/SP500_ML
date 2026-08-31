# Work Log 2

日期：2026-08-31

## 本次工作结论

项目已经完成从 provenance contract 到规则 agent 合成压力测试、历史 replay、V3
条件式来源风险和 V4 学习型路由器的第一轮完整闭环。当前最重要的结果不是“V4
全面优于 V3”，而是一个可用于论文的方法学发现：在完全未见的失效机制下，固定的
结构性 provenance prior 比只从已知机制学习权重的路由器更稳健。

下一阶段不继续针对 synthetic 指标调出 V5，也不立即投入 Qwen 微调或 RL。最短的
论文路径是构建真实 LLM 的 paired-intervention 实验，并加入一个非金融迁移域，验证
当前结论是否能从规则 agent 外推到实际语言模型。

## 自 Work Log 1 以来已完成的工作

### 1. Synthetic false-consensus benchmark

完成了环境持有的合成实验框架，能够生成 clean control、shared corruption、stale
evidence、partial corruption 和 evidence inertia 等机制。实验支持：

- 来源可见性隐藏或别名化；
- 不透明的派生特征名称和共同 provenance root；
- source-quality noise、corruption strength 与 3/5/7/9 个 agent；
- mechanism-held-out 评估；
- majority、confidence、agreement、recent performance、provenance 及其公平消融；
- AUROC、AUPRC、ECE、Brier、AURC、Risk@coverage、selective error、coverage 和
  false rejection；
- 按 base seed 聚类的 bootstrap 置信区间和自动生成图表/报告。

Synthetic V1/V2 验证了框架和压力测试流程；V2 刻意削弱 confidence 与 source quality
之间的耦合，使 provenance 审计与行为基线在信息权限上更加清晰。它们仍是受控规则
agent 结果，不构成 LLM 或市场有效性证据。

### 2. Historical replay V0--V2

完成了严格按时间顺序的 expanding walk-forward replay，包括五日 gap、训练期阈值、
nested-OOF 风险估计、共同来源分组，以及 CBOE/ICI/ETF flow 数据扩展。

Historical V2 中，provenance 与 majority 的 routed error 均为 `0.377`；confidence 为
`0.371`。因此真实数据 replay 尚未证明 provenance 优于多数票或 confidence。当前结果
也没有交易成本、逐行盘中发布时间审计或 LLM 调用，不能被描述为投资优势或因果市场
影响。

### 3. Synthetic V3：条件式 provenance 与 evidence inertia

V3 新增 evidence inertia：agent 表面引用及时且高质量的证据，但在 paired
remove/reverse intervention 下不改变 action。固定的 Conditional provenance 同时使用
来源集中度、来源质量、陈旧比例、时间违规和干预响应。

冻结 V3 的主要结果为：

| 方法 | AUROC | AURC | Risk@80 | 冻结阈值 error | coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quality only | 0.784 | 0.471 | 0.470 | 0.462 | 0.788 |
| Conditional provenance | 0.977 | 0.233 | 0.466 | 0.472 | 0.809 |

V3 明显改善了整体风险排序，但在原始冻结阈值下，由于两种方法 coverage 不同，
Conditional provenance 的 high-confidence error 略高。因此原始 primary hypothesis
不能宣称通过。

随后增加了明确标记为 post-hoc 的 matched-coverage 审计。在相同 80% coverage 下，
Conditional provenance 的 risk 为 `0.466`，Quality only 为 `0.470`，paired
cluster-bootstrap 差值为 `-0.004 [-0.008, -0.001]`。这补充支持排序价值，但不会回溯
改变 V3 的预注册结论。

### 4. Synthetic V4：学习型单调路由器

V4 在独立预注册下完成，未修改 V1--V3 的 seed、结果或主张。它引入：

- 完全互不重叠的训练 seeds 与四组 outer-test seeds；
- leave-one-mechanism-out 外层评估和按 base seed 分组的五折 cross-fitting；
- 带噪 action 和 paired-intervention，避免 observable feature 成为 label 的确定性副本；
- 四个非负系数的 logistic provenance router；
- 单调 Platt calibration；
- 只用训练数据选择目标为 80%--82% coverage 的部署阈值；
- 1,000 次 paired base-seed-cluster bootstrap。

正式结果如下：

| 方法 | Macro AURC | Risk@80 | AUROC | ECE |
| --- | ---: | ---: | ---: | ---: |
| Quality only | 0.352 | 0.442 | 0.756 | 0.226 |
| Conditional provenance V3 | **0.235** | **0.440** | **0.916** | 0.123 |
| Monotonic provenance V4 | 0.316 | 0.447 | 0.769 | **0.074** |

V4 的学习型路由器改善了概率校准，也优于 confidence 的 Macro AURC 和 Risk@80；但
它没有同时超过 Quality only 和固定 V3 score，因此预注册主假设未通过。尤其当
evidence inertia 被完全留出时，V4 学到的 intervention coefficient 只有 `0.013`，测试
coverage 漂移到 `0.998`。这说明训练于已知机制的经验权重没有可靠外推到新的机制，
而 V3 显式保留的 causal-effect prior 更稳健。

### 5. 当前“选择性路由”的确切含义

当前实验先计算一次多 agent 共识，再由 risk score 决定：

```text
agent decisions -> consensus action -> risk score -> accept consensus / abstain
```

在金融表述中，`abstain` 对应不执行该共识、保持 cash。它目前不是在多个 agent 中挑选
另一个更好的 agent，也不是动态调用专家。后续论文可以扩展为三路决策：接受共识、
转交给独立证据支持的 agent、或 abstain；但该扩展尚未实现和验证。

## 论文方向

### 核心问题与暂定题目

论文问题收敛为：当多个 LLM agent 依赖重复、过期、被污染或对 action 没有因果作用的
共同证据时，它们可能形成高度自信但错误的共识。环境持有的来源图和 paired evidence
intervention 能否在未见失效机制下支持可靠的选择性路由？

暂定题目：

> **When Consensus Lies: Provenance-Aware Selective Routing for LLM Agents under
> Correlated Evidence**

金融应作为具有严格 as-of 约束和 distribution shift 的困难验证域，而不是把 S&P 500
收益曲线当作论文中心。核心贡献应落在通用 LLM-agent reliability、benchmark、paired
intervention 和 routing。

### 论文最小闭环

下一项实验定义为 **Pilot-LLM V1**：

1. 先选择 50 个 StrategyQA 样本；
2. 每题调用 5 个 Qwen3.5-4B agent；
3. 对每题构造 original、remove 和 reverse 三种配对证据条件；
4. 强制输出 answer、confidence、cited evidence IDs 和 answer/abstain；
5. 首轮约 750 次调用，仅验证调用、解析、干预敏感性和评估协议；
6. Pilot 通过后扩到约 200 题和 3,000 次调用，并加入 S&P 500 temporal replay；
7. 资源允许时增加第二个模型族或模型尺寸，检验结论是否依赖 Qwen。

不保存或评价隐藏 chain-of-thought；只评价答案、置信度、环境证据引用和干预前后可观察
行为。

### 下一版方法：Hybrid Router

V4 的负结果支持一种受约束的 hybrid 设计，而不是继续自由学习全部权重：

1. 保留 V3 的来源结构与 causal-effect risk 作为固定 prior；
2. 只用训练数据学习校准映射和 deployment threshold；
3. 如需学习残差，对其方向和幅度施加约束，不能覆盖显式 provenance prior；
4. 在完全未见的 failure mechanism 上一次性测试。

正式基线应包括 single agent、majority、confidence、agreement、recent performance、
Quality only、固定 V3、学习型 V4、Hybrid 和 diagnostic Oracle。核心指标继续使用 Macro
AURC、Risk@80、paired intervention sensitivity、false rejection 和 calibration，同时
报告 schema failure、成本和延迟。

## 投稿与工作时间线

截至 2026-08-31，2026-10-12 ARR 是 NAACL/COLING 2027 的可选冲刺线，但时间不足以
把它作为唯一目标。更稳妥的主目标是 ACL 2027 对应的 2027 年 1 月 ARR 周期，具体日期
以官方更新为准：

- [ARR 官方日程](https://aclrollingreview.org/dates)
- [NAACL 2027 Call for Papers](https://www.aclweb.org/portal/content/call-main-conference-papers-naacl-2027)
- [ACL 2027 官网](https://2027.aclweb.org/)
- [2027 联合 Workshop 日程](https://www.aclweb.org/portal/content/joint-call-workshops-proposals-2027)

建议节奏：

- 9 月 1--14 日：冻结 Pilot-LLM V1 协议，完成 50 题 pilot 和一页 research brief；
- 9 月中旬：只有在 paired effect 清楚、数据冻结且论文骨架已形成时才冲刺 10 月 ARR；
- 9--11 月：完成约 200 题、两个任务域、完整基线和消融；
- 11--12 月：人工审计、第二模型、误差分析和写作；
- 2027 年 1 月：以 ACL/Findings 对应 ARR 为主要目标；
- 若模型或跨域证据不足，则转向匹配的 NLP/agent/trustworthiness workshop。

## UIUC 研究联络计划

在只有 synthetic 结果时不直接请求 RA。更合适的联络节点是完成两周 Pilot 后，带着：

- 一页 research brief；
- 可运行的代码和冻结协议；
- 一项真实 LLM paired-intervention 结果及置信区间；
- V4 在 unseen mechanism 下失败、V3 structural prior 更稳健的机制解释。

可按研究匹配度考察：

- [Hao Peng](https://haopeng-nlp.github.io/)：LLM reasoning、agent evaluation、causal
  reasoning；
- [Heng Ji](https://blender.cs.illinois.edu/hengji/research.html)：跨来源信息抽取、LLM
  agents 和 AI safety；
- [Han Zhao](https://hanzhaoml.github.io/)：trustworthy ML、distribution shift 和
  quantitative finance；
- [Hao Wang](https://people.csail.mit.edu/haow/)：multi-agent systems、process
  verification 和 uncertainty。

上述列表只表示研究方向匹配，不表示当前存在 RA 名额。首封邮件应以“共同作者中有一位
UIUC 学生”专业表述合作关系，请求一次约 20 分钟、围绕研究问题的具体反馈；不群发，
不在第一封邮件中直接请求署名或 RA。若反馈形成真实合作，再询问进一步指导或 RA 的
可能性。

## 当前边界与下一动作

目前已经得到一个值得继续研究的 synthetic 机制结论，但尚未得到真实 LLM faithfulness、
跨领域泛化或投资有效性证据。下一动作是建立 `Pilot-LLM V1`，而不是宣称论文主假设已经
成立。
