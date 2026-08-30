# Qwen3.5-4B 微调计划

## 0. 先确认微调条件

“服务器上部署了 Qwen3.5-4B”可能有两种情况：

1. 只有 OpenAI-compatible 推理接口：可以调用模型，但不能通过接口微调；
2. 有原始 Transformers checkpoint、GPU、训练环境和写入权限：可以做 LoRA/QLoRA。

需要向服务器管理员确认：模型路径、模型类型（Base/Instruct/Thinking）、显存、CUDA/PyTorch/Transformers/PEFT 版本，以及输出目录。官方模型卡确认 Qwen3.5-4B 提供 Transformers 格式，并可用于 Transformers/vLLM 等工具；训练框架对该具体 checkpoint 的支持仍需在服务器上实际验证。[Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)

优先 LoRA 或 QLoRA，不从 full fine-tuning 开始。Qwen 官方训练文档列出了 LoRA/QLoRA 和 LLaMA-Factory 训练路径，但具体模板必须匹配当前模型版本。[Qwen fine-tuning documentation](https://github.com/QwenLM/Qwen3/blob/main/examples/llama-factory/finetune-zh.md)

所有 checkpoint、cache、logs、JSONL 和 adapter 输出放在 `/storage/gaoym` 或训练服务器对应的 storage 路径，不写入 home。

## 1. 数据样本

每个样本只使用时刻 `t` 已经可见的 state：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a risk-averse market behavior agent. Return JSON only."
    },
    {
      "role": "user",
      "content": "<structured market state at t and environment-provided evidence packet>"
    },
    {
      "role": "assistant",
      "content": "{\"decision_time\":\"...\",\"action\":\"cash\",\"target_exposure\":0.0,\"horizon_days\":5,\"confidence\":0.78,\"claims\":[{\"claim_id\":\"c1\",\"text\":\"...\",\"stance\":\"supports\",\"evidence_ids\":[\"e1\"]}]}"
    }
  ],
  "metadata": {
    "timestamp": "...",
    "persona": "risk_averse",
    "split": "train"
  }
}
```

未来收益可以用于给训练期样本生成 outcome label 或 chosen/rejected pair，但不能把验证/测试期结果放进 prompt、persona 选择、trust 权重或超参数搜索。Evidence IDs、来源和时间戳必须由数据管线提供；模型只允许引用候选 evidence，不能自行生成来源或修改 `available_at`。

## 2. 三阶段训练

### Stage A: structured SFT

训练目标：

- JSON schema validity；
- action/exposure/horizon 的约束；
- persona adherence；
- 简短 claims 和合法 evidence IDs；证据的 provenance 字段只由环境维护；
- 高不确定性时 abstain/cash。

数据来源可以混合规则策略、人工检查样本和 teacher LLM 样本。不能全部由同一个 teacher 生成后再声称是独立发现。

### Stage B: reliability-aware preference tuning

为同一个 state 构造 chosen/rejected：

`chosen` 应满足：

- action 与 persona 约束一致；
- claims 能支持 action；
- 引用的证据在 `decision_time` 前可见；
- 多 agent 结论相同时，优先保留来源独立而非重复转述的样本；
- 删除或反转关键 evidence 后，action/confidence 做出合理变化；
- confidence 与训练期 outcome/calibration 相符；
- 不违反暴露、成本和风险限制。

`rejected` 可以包括：高 confidence 但错误、理由和 action 矛盾、违反 schema、引用未来或过期证据、把同源文本当作独立证据，以及对关键证据删除/反转不敏感的输出。

先使用 DPO/ORPO 类方法做小规模实验；如果 pairwise label 不稳定，就保留 SFT，不强行加入 preference optimization。

### Stage C: router / offline policy

微调后的 Qwen 只产生 agent decision。router 由 Python/PyTorch 实现，读取：

```text
p_up, interval_width, regime,
agent confidence, as-of validity, grounding,
source independence, causal effect, current exposure
```

先用规则和 contextual bandit，最后才考虑 offline RL。RL 的 reward 必须包含交易成本、换手和回撤惩罚。

## 3. 训练与评估切分

不能随机切分相邻日期。建议使用 expanding-window：

```text
past window -> calibration window -> held-out future window
```

每个 walk-forward fold 内独立训练 adapter、校准 confidence 和选择 router 超参数。测试 fold 只用于最终评估。

## 4. 必须报告的对照

| Variant | 目的 |
| --- | --- |
| base Qwen prompt-only | 测试微调是否有必要 |
| Qwen LoRA-SFT | 测试结构化行为学习 |
| Qwen preference-tuned | 测试可靠性偏好目标 |
| rule-based persona | 非 LLM 行为基线 |
| majority vote | 简单多 agent 聚合 |
| self-confidence router | 不使用外部 trust 的基线 |
| agreement/consensus router | 检验共享错误是否欺骗群体一致性 |
| recent-performance/regime router | 不使用证据谱系的动态路由基线 |
| provenance-aware selective router | 核心方法 |

指标至少包括 JSON validity、persona consistency、false-consensus AUROC/AUPRC、future-evidence violation、ECE、Brier、AURC、routing regret、shared-corruption robustness、regime-wise performance、turnover、最大回撤和推理成本/延迟。

## 5. 避免的陷阱

1. 不把 LLM 自己写的 rationale 当作真实解释；
2. 不用同一个模型既生成 decision 又独立证明 decision 正确；
3. 不用测试期收益挑选 prompt、persona 或 trust 权重；
4. 不把一次回测胜出写成“找到最优投资策略”；
5. 不保存或训练隐藏 chain-of-thought，保存短 claims 和证据标签即可。
