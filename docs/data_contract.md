# 数据契约与时间对齐

## 原始输入

原始仓库目前提供了以下数据类别：

| 文件/类别 | 预期内容 | 频率 |
| --- | --- | --- |
| `market.csv` | S&P 500 价格、1/5/10 日收益、10 日波动率 | 日 |
| `macro.csv` | 10Y、信用变量 | 日 |
| `vix.csv` | VIX | 日 |
| `sentiment.csv` | Google Trends 情绪词 | 周 |
| `stock_pcr.csv` | stock put/call ratio | 日 |
| `SPY/IVV/VOO_*` | ETF 行情和资金流 | 日 |
| `ICI_mutual_fund_flow.xlsx` | 共同基金资金流 | 周 |
| `CBOE_full.xlsx` | CBOE 期权汇总数据 | 日 |

第一版实验不要求一次性使用全部文件；建议先用 `market + vix + macro + sentiment` 建立可复现实验，再逐组加入资金流和期权。

## 时间语义

- 日频价格、VIX 和期权变量：假设在当日收盘后生成信号；目标窗口从下一交易日开始。
- 周频数据：必须保留其发布日期/可见日期。不能把周日标签直接当作周一开盘前已知。
- 发布时间未知时：使用一周滞后作为保守规则，并在实验报告中标记。
- 缺失值只能使用当时可用的历史值填充；禁止用未来值 `bfill`。
- 合并前统一时区、日期格式和交易日历，并保留每个来源的原始日期列。

## 目标字段

```text
forward_return_5d = close[t + 5] / close[t] - 1
target_up_5d = 1[forward_return_5d > 0]
```

最后 5 个观测没有完整标签，必须丢弃，不能把缺失标签变为 0。

## 版本与质量检查

每次实验记录：数据源 URL、下载时间、文件 hash、覆盖日期、缺失比例和时区。

原仓库的说明文件显示当前数据快照大约覆盖至 2026 年 5 月；它不是实时数据，开始正式实验前必须重新检查日期覆盖和字段定义。FRED 的 S&P 500 页面说明该序列是收盘价、日频，并且是 price index（不含股息），因此若使用它作为基准，不要误称为 total return。[FRED S&P 500 series](https://fred.stlouisfed.org/series/SP500)

VIX 的经济含义是由 S&P 500 期权推导的预期波动率，不应直接解释成“涨跌方向指标”。[Cboe VIX methodology](https://cdn.cboe.com/api/global/us_indices/governance/VIX_Methodology.pdf)

