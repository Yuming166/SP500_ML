# Work Log 1

日期：2026-08-30

## 本次完成内容

### 研究主线收敛

将第二阶段的核心问题收敛为 **As-of Provenance Faithfulness**：多 agent 的一致意见只有在证据真实可见、来源独立、能够支持 claim，且对最终 action 有因果影响时才可信。论文要检测的是共享错误、过期信息或未来泄漏造成的 false consensus，而不是泛化的 confidence score 或单次交易收益。

相关的论文提案、阶段二设计、微调计划、数据约定和 README 已统一为这一主线。Qwen3.5-4B 微调用于学习受约束的结构化决策与 abstention；金融信号处理用于提供严格 as-of 的证据；RL 仅作为后续 router 优化扩展。

### 证据谱系与 agent 契约

完成了第二阶段的基础代码契约：

- `EvidenceItem` 记录 `source_id`、`event_time`、`publication_time`、`available_at`、摘要和 `parent_evidence_ids`；
- `Claim` 只保存简短文本、立场和引用的 `evidence_ids`；
- `ProvenanceGraph` 维护环境持有的 evidence catalog，验证父节点、循环依赖、时间可见性和根来源；
- `AgentDecision` 新增 `decision_time` 和结构化 claims；agent 不能自行生成来源、时间戳或派生关系；
- 解析器在运行时拒绝未来证据、catalog 外引用和 agent evidence packet 外引用；
- 独立性按叶子 `source_id` 计算，因此同一原始价格序列派生的 trend 与 momentum 不会被误算为两份独立支持。

### 验证

- 为未来证据、同源派生特征和 catalog/packet 外引用补充了单元测试；
- 完整测试套件通过：`14 passed`；
- Ruff 静态检查和 `git diff --check` 通过；
- 在仓库内建立了被忽略的 `.venv/` 测试环境，避免修改系统 Python 环境。

## 下一步计划

1. 构建不依赖 LLM 的 synthetic false-consensus benchmark：先用规则 agent 和环境持有的 evidence packet 生成有真值标签的决策样本。
2. 实现 paired clean/corrupted 干预：source duplication、shared corruption、stale evidence、future leakage，以及 evidence removal/reversal。
3. 实现最小 baseline：majority vote、self-confidence、agreement weighting、recent-performance 和 provenance-aware selective router。
4. 先报告 false-consensus detection AUROC/AUPRC、high-confidence error、AURC 和 shared-corruption robustness；确认核心假设后，再接入 S&P 500 historical replay。
5. 最后才将 Qwen3.5-4B 接入为受 evidence packet 约束的 agent，并比较 prompt-only、SFT 与后续 preference tuning。

## 当前判断

项目已从“金融多 agent 交易框架”进入可证伪的可信决策研究阶段。下一项工作应优先验证证据谱系信号能否识别共享错误；在此之前，不应把 Qwen 微调或 RL 作为主实验投入。
