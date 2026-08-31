# SP500 Forecastability Lab

一个面向 S&P 500 的因果多分辨率信号学习研究项目

## 研究问题

> 在什么市场状态下，S&P 500 的短期走势更容易预测？模型能否识别出“不应该预测”的时刻？

## 第一阶段 固定预测窗口为未来 5 个交易日，模型输出三部分：

- `p_up`：未来 5 日收益为正的概率；
- 不确定性区间：预测可能落入的收益范围；
- `forecast_status`：`forecast` 或 `abstain`，即预测或拒绝预测。

项目的 CS/ECE 主线是 causal multi-resolution representation learning：先用只依赖历史样本的滤波器组/小波分解，把价格、VIX、资金流、情绪和宏观变量拆成不同时间尺度；再让模型学习各频带的动态权重；最后用校准和 abstention 处理非平稳环境中的不确定性。

重点验证：在相同数据和严格时间外推下，自适应多分辨率表示是否比原始序列和静态融合更稳健，尤其是在市场状态切换时。
小波变换本身不是项目的创新声明；已有多尺度小波时序模型。因此创新应落在“因果在线约束 + 自适应频带路由 + 预测可靠性评估”的组合和消融实验上。

## 方法架构

```text
as-of aligned multichannel signals
              |
causal filter bank / causal wavelet levels
              |
band-specific temporal encoder (TCN or small GRU)
              |
state-conditioned gating / cross-channel fusion
              |
point forecast + calibrated uncertainty + abstention
              |
optional offline policy layer: long / cash
```
第一版先实现可解释的 causal filter bank，作为 wavelet 和 learnable filter bank 的基线。任何使用 `filtfilt`、中心滑动窗口或未来边界填充的实现都不能进入正式结果。

## 第二阶段：LLM 虚假共识与证据溯源

第二阶段不把 LLM 当作数字预测器，而把它当作具有不同风险偏好和决策规则的行为生成器：趋势跟随、均值回归、风险规避、噪声交易、情绪驱动等 agent 接收共享任务上下文和各自的 evidence packet，输出 `cash/long` 决策、持有期限、置信度，以及对环境证据目录的可检查引用。来源、发布时间和派生关系只能由环境维护；同一原始数据派生出的不同特征保留共同祖先，不能被当作多份独立支持。

核心创新是 **As-of Provenance Faithfulness**：显式记录 `source -> evidence -> claim -> agent -> action`，判断证据是否在决策时刻可见、是否真正支持 claim、多个 agent 是否依赖独立来源，以及干预证据是否会改变 action。router 根据这些信号选择可信 agent；当所有 agent 因共享的错误、过期或未来证据形成虚假共识时，系统应选择 `abstain/cash`，而不是服从多数。

LLM 行为仿真的研究问题是：

> 系统能否识别由共享证据错误造成的 false consensus，并在 regime shift 下路由到由独立、及时证据支持的 agent？

历史 replay 只能评估策略，不会让 agent 真正改变市场；如果要研究群体行为如何形成价格，需要增加一个经过历史 stylized facts 校准的闭环市场模拟器。`optimal strategy` 应理解为给定环境、交易成本和风险目标下的最优策略，而不是现实市场中的全局最优。

## 论文定位

当前工作版本的论文题目暂定：

**When Consensus Lies: As-of Provenance Faithfulness for Multi-Agent LLM Decisions under Distribution Shift**

论文贡献集中在 false-consensus benchmark、时间证据谱系评估和 provenance-aware selective routing；S&P 500 是具有真实发布时间和状态切换的测试环境，不把单次回测收益作为主要贡献。Qwen3.5-4B 微调、金融信号处理和 RL 是实现组件，不是中心 novelty。具体研究定义、投稿定位和微调实验见：

- [`docs/asof_provenance_faithfulness.md`](docs/asof_provenance_faithfulness.md)
- [`docs/synthetic_benchmark_spec.md`](docs/synthetic_benchmark_spec.md)
- [`docs/synthetic_v2_preregistration.md`](docs/synthetic_v2_preregistration.md)
- [`docs/synthetic_v3_preregistration.md`](docs/synthetic_v3_preregistration.md)
- [`docs/synthetic_v4_preregistration.md`](docs/synthetic_v4_preregistration.md)
- [`docs/paper_proposal.md`](docs/paper_proposal.md)
- [`docs/finetuning_plan.md`](docs/finetuning_plan.md)

阶段性记录：[`Work Log 1`](docs/work_log_1.md) · [`Work Log 2`](docs/work_log_2.md)

## 数据基础

[`Yuming166/SP500_ML`](https://github.com/Yuming166/SP500_ML)

- S&P 500/ETF 行情与收益；
- VIX；
- 10 年期美债收益率和信用利差；
- Google Trends 情绪；
- stock put/call ratio；
- SPY、IVV、VOO 和共同基金资金流。

数据不会直接复制进本仓库。将原始文件放入 `data/raw/`，处理产物放入 `data/processed/`。原仓库中的数据快照截至 2026 年 5 月左右，正式实验前会重新确认数据的最新日期、来源和授权。

## 防止数据泄漏

1. 只使用预测时刻已经可见的数据；
2. 日频数据默认在收盘后生成信号，目标从下一个交易窗口计算；
3. 周频情绪和资金流按发布时间做 as-of 对齐，发布时间不明确时保守滞后一周；
4. 训练、校准、测试按时间顺序划分，不随机打乱；
5. 置信度阈值和 conformal 区间半径只能用过去的校准窗口估计。

## 评估重点

- 方向模型：LogLoss、Brier score、校准误差；
- 选择性预测：coverage、selective accuracy、不同 coverage 下的表现；
- 区间预测：覆盖率、平均区间宽度；
- 策略模拟：累计收益、年化波动、最大回撤、换手率，并与 buy-and-hold 基线比较；
- 稳定性：按年份、VIX 分位数和市场状态分别评估。

## 当前状态

目前已具备选择性预测指标、conformal 区间的基础函数、时间序列 walk-forward 切分、跨市场分歧特征、因果滤波器组，以及环境持有的 provenance-aware agent contract。第二阶段还包含不依赖 LLM 的 synthetic harness；冻结的 V2 压力测试比较行为基线与来源审计，独立的 V3 则进一步比较 quality-only、source-overlap-only、temporal-only 和 conditional-provenance 消融。

V3 增加了 evidence inertia：agent 表面引用及时且高质量的证据，但在 paired remove/reverse intervention 下不随证据改变 action。它用于区分“共享但可靠的证据”与“同源且不完整/陈旧，或仅作事后理由的证据”。所有版本均支持 hidden/aliased provenance、opaque transformation names、带噪 source-quality estimates、可变 agent 数、corruption strength 和 mechanism-held-out split。通过压力测试后，下一步才是用相同契约接入 S&P 500 historical replay 与 Qwen agent。

V3 的冻结协议与已生成结果分别见 [`docs/synthetic_v3_preregistration.md`](docs/synthetic_v3_preregistration.md) 和 [`results/synthetic_v3.md`](results/synthetic_v3.md)。

补充的 matched-coverage 分析明确标记为 post-hoc；在相同 80% coverage 下，conditional provenance 的 error 为 `0.466`，quality-only 为 `0.470`，paired bootstrap 差值为 `-0.004 [-0.008, -0.001]`。这支持 V3 的排序能力，但不回溯修改原始 primary hypothesis。

独立预注册的 V4 使用全新且互不重叠的训练/测试 seeds、带噪 action/intervention、按 seed 分组的 cross-fitting，以及非负 logistic provenance router。正式结果中，学习型 V4 的校准更好（ECE `0.074`），宏平均 AURC 优于 quality-only（`0.316` vs `0.352`），但 Risk@80 略差（`0.447` vs `0.442`），同时落后固定 V3 conditional provenance（宏平均 AURC `0.235`、Risk@80 `0.440`）。因此 V4 的预注册主假设未通过；结果表明完全未知的 evidence-inertia 机制需要显式结构先验，不能只依赖已知机制上的监督学习。详见 [`results/synthetic_v3_posthoc_matched_coverage.md`](results/synthetic_v3_posthoc_matched_coverage.md) 和 [`results/synthetic_v4.md`](results/synthetic_v4.md)。

### Pilot-LLM（真实 Qwen3.5-4B，2026-08-31）

V3 的 causal-risk AUROC = `0.4125`（< 0.5，主假设未通过），post-hoc 诊断定位到三个协议缺陷而非模型缺陷：StrategyQA 触发 parametric prior fallback、V3 把同一证据发给所有 agent、causal-risk 定义过窄。

V4-LLM 修复三个缺陷并正式跑完：TruthfulQA 复合题（更难领域）、partitioned 2-of-3 evidence packets（恢复 citation 方差）、新增 substitute 干预（强制 model 用错证据而非 fallback）。**主假说通过**：D_OR AUROC = `0.676`，95% question-cluster bootstrap CI `[0.515, 0.821]`。三 co-registered secondary endpoints 一并报告：`D_inert` AUROC `0.686 [0.524, 0.838]`、`D_conf` AUROC `0.625 [0.463, 0.761]`。Platt LOO 校准 brier=`0.244`、ECE=`0.100`。LOAO 鲁棒性 median=`0.664 [0.627, 0.724]`，5 个变体无一下跌到 0.5 之下。shared_citation_signal（V3 diagnostic Adjustment 6）AUROC=`0.611 [0.473, 0.747]` —— 第一次在真实 LLM 上验证该信号。

V4 报告、原始 1,000 条记录和冻结协议见 [`results/pilot_llm_v4/formal/report.md`](results/pilot_llm_v4/formal/report.md) 与 [`docs/pilot_llm_v4_preregistration.md`](docs/pilot_llm_v4_preregistration.md)。Work Log 3 记录 pre-formal hardening、Work Log 4 记录 formal run 与 S2 解释。

目录约定和研究设计见：

- [`docs/research_plan.md`](docs/research_plan.md)
- [`docs/data_contract.md`](docs/data_contract.md)
- [`docs/resume_positioning.md`](docs/resume_positioning.md)
- [`docs/stage2_llm_market_simulation.md`](docs/stage2_llm_market_simulation.md)
- [`docs/work_log_3.md`](docs/work_log_3.md)（V4 pre-formal hardening）
- [`docs/work_log_4.md`](docs/work_log_4.md)（V4 formal run + S2 interpretation）

## 本地运行

```bash
python -m pip install -e '.[dev]'
pytest -q
python -m sp500_forecastability.synthetic_v4_experiment
```

本项目仅用于研究和教育，不构成投资建议。
