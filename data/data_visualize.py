import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 读取
# =========================

sentiment = pd.read_csv(
    "/Users/bob/Desktop/quant/data/sentiment.csv",
    index_col=0,
    parse_dates=True
)

market = pd.read_csv(
    "/Users/bob/Desktop/quant/data/market.csv",
    index_col=0,
    parse_dates=True
)

macro = pd.read_csv(
    "/Users/bob/Desktop/quant/data/macro.csv",
    index_col=0,
    parse_dates=True
)

vix = pd.read_csv(
    "/Users/bob/Desktop/quant/data/vix.csv",
    index_col=0,
    parse_dates=True
)

# =========================
# 2. 各自转周频
# =========================

market_w = market.resample("W").last()
macro_w = macro.resample("W").last()
vix_w = vix.resample("W").last()

# sentiment本来就是周频
sentiment_w = sentiment.copy()

# =========================
# 3. merge
# =========================

df = pd.concat([
    sentiment_w,
    market_w[["sp500", "ret_10d", "vol_10d"]],
    macro_w[["10Y", "credit"]],
    vix_w[["vix"]]
], axis=1)

# =========================
# 4. 填充缺失
# =========================

df = df.ffill()

# 只保留共同时间段
#df = df.dropna()

# =========================
# 5. 标准化
# =========================

df_norm = (df - df.mean()) / df.std()

# =========================
# 6. 画图
# =========================

plt.figure(figsize=(18, 9))

for col in df_norm.columns:
    plt.plot(df_norm.index, df_norm[col], label=col)

plt.title("SP500 + Macro + Sentiment Indicators")
plt.xlabel("Date")
plt.ylabel("Z-score")
plt.legend()

plt.grid(True)

plt.show()