import pandas as pd
import polars as pl

# 用pandas读取
df = pd.read_parquet("submission.parquet")
print(df)

# 查看列名和数据类型
print(df.info())

# 若用polars读取（速度更快，适合大文件）
# df = pl.read_parquet("submission.parquet")
# print(df.head())