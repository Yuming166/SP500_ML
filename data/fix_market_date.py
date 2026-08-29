import pandas as pd

# 读取
df = pd.read_csv("/Users/bob/Desktop/quant/data/market.csv")

# 查看前几行
print(df.head())

# 第一列改成 datetime
date_col = df.columns[0]

df[date_col] = pd.to_datetime(
    df[date_col],
    format="%m/%d/%y"
)

# 转成标准格式
df[date_col] = df[date_col].dt.strftime("%Y-%m-%d")

# 保存
df.to_csv(
    "/Users/bob/Desktop/quant/data/market.csv",
    index=False
)

print("✅ market.csv date format fixed")