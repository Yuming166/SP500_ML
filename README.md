# SP500 Forecastability Lab

一个面向 S&P 500 的因果多分辨率信号学习研究项目。

## 研究问题

与其每天强行预测涨跌，不如研究：

> 在什么市场状态下，S&P 500 的短期走势更容易预测？模型能否识别出“不应该预测”的时刻？

第一版固定预测窗口为未来 5 个交易日，模型输出三部分：

- `p_up`：未来 5 日收益为正的概率；
- 不确定性区间：预测可能落入的收益范围；
- `forecast_status`：`forecast` 或 `abstain`，即预测或拒绝预测。

项目的 CS/ECE 主线是 **causal multi-resolution representation learning**：先用只依赖历史样本的滤波器组/小波分解，把价格、VIX、资金流、情绪和宏观变量拆成不同时间尺度；再让模型学习各频带的动态权重；最后用校准和 abstention 处理非平稳环境中的不确定性。

重点验证：在相同数据和严格时间外推下，自适应多分辨率表示是否比原始序列和静态融合更稳健，尤其是在市场状态切换时。

小波变换本身不是项目的创新声明；已有多尺度小波时序模型。因此创新应落在“因果在线约束 + 自适应频带路由 + 预测可靠性评估”的组合和消融实验上。

这是一项研究假设，不预设模型一定能赚钱；如果结果没有稳定改善，也应作为有效结论记录。

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

第二阶段不把 LLM 当作数字预测器，而把它当作具有不同风险偏好和决策规则的行为生成器：趋势跟随、均值回归、风险规避、噪声交易、情绪驱动等 agent 接收共享任务上下文和各自的 evidence packet，输出 `cash/long` 决策、持有期限、置信度，以及带来源和发布时间的可检查证据。同一原始数据派生出的不同特征保留共同祖先，不能被当作多份独立支持。

核心创新是 **As-of Provenance Faithfulness**：显式记录 `source -> evidence -> claim -> agent -> action`，判断证据是否在决策时刻可见、是否真正支持 claim、多个 agent 是否依赖独立来源，以及干预证据是否会改变 action。router 根据这些信号选择可信 agent；当所有 agent 因共享的错误、过期或未来证据形成虚假共识时，系统应选择 `abstain/cash`，而不是服从多数。

LLM 行为仿真的研究问题是：

> 系统能否识别由共享证据错误造成的 false consensus，并在 regime shift 下路由到由独立、及时证据支持的 agent？

历史 replay 只能评估策略，不会让 agent 真正改变市场；如果要研究群体行为如何形成价格，需要增加一个经过历史 stylized facts 校准的闭环市场模拟器。`optimal strategy` 应理解为给定环境、交易成本和风险目标下的最优策略，而不是现实市场中的全局最优。

## 论文定位

当前工作版本的论文题目可以暂定为：

**When Consensus Lies: As-of Provenance Faithfulness for Multi-Agent LLM Decisions under Distribution Shift**

论文贡献集中在 false-consensus benchmark、时间证据谱系评估和 provenance-aware selective routing；S&P 500 是具有真实发布时间和状态切换的测试环境，不把单次回测收益作为主要贡献。Qwen3.5-4B 微调、金融信号处理和 RL 是实现组件，不是中心 novelty。具体研究定义、投稿定位和微调实验见：

- [`docs/asof_provenance_faithfulness.md`](docs/asof_provenance_faithfulness.md)
- [`docs/paper_proposal.md`](docs/paper_proposal.md)
- [`docs/finetuning_plan.md`](docs/finetuning_plan.md)

## 数据基础

你原来的 [`Yuming166/SP500_ML`](https://github.com/Yuming166/SP500_ML) 已经准备了适合这个问题的数据：

- S&P 500/ETF 行情与收益；
- VIX；
- 10 年期美债收益率和信用利差；
- Google Trends 情绪；
- stock put/call ratio；
- SPY、IVV、VOO 和共同基金资金流。

数据不会直接复制进本仓库。请将原始文件放入 `data/raw/`，处理产物放入 `data/processed/`。原仓库中的数据快照截至 2026 年 5 月左右，正式实验前需要重新确认数据的最新日期、来源和授权。

## 防止数据泄漏的约定

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

目前是研究骨架：已经包含选择性预测指标、conformal 区间的基础函数、时间序列 walk-forward 切分、跨市场分歧特征、因果滤波器组和第二阶段的基础 agent 输出契约。下一步先把输出契约扩展为带 `source_id`、`available_at` 和 claim-evidence links 的证据谱系，再实现 source duplication、shared corruption、stale/future evidence、evidence removal/reversal 等 paired benchmark，最后比较 majority、confidence、agreement、recent-performance 与 provenance-aware routing。

目录约定和研究设计见：

- [`docs/research_plan.md`](docs/research_plan.md)
- [`docs/data_contract.md`](docs/data_contract.md)
- [`docs/resume_positioning.md`](docs/resume_positioning.md)
- [`docs/stage2_llm_market_simulation.md`](docs/stage2_llm_market_simulation.md)

## 本地运行

```bash
python -m pip install -e '.[dev]'
pytest -q
```

本项目仅用于研究和教育，不构成投资建议。
