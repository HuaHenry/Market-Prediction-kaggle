# -*-coding:utf-8 -*-
'''
@File    :   crop_dataset.py
@Modify  :   2025/12/04 22:09:06
@Author  :   Zhouqi Hua 
@Version :   1.0
@Desc    :   重要性阈值筛特征、筛缺失率(<0.4 除了 col M6)、删去前1007行巨量缺失值数据
'''

import pandas as pd

# ---------------------------
# 1. 读取数据
# ---------------------------
train_path = "../data/train.csv"
imp_path = "combined_importance_filtered.csv"

df = pd.read_csv(train_path)
imp_df = pd.read_csv(imp_path, index_col=0)

# 确认 mean_importance 存在
if "mean_importance" not in imp_df.columns:
    raise ValueError("combined_importance_filtered.csv 中缺少 mean_importance 列，请检查。")

# ---------------------------
# 2. 删除原训练集前 1007 行（除了 header）
# ---------------------------
df = df.iloc[1007:].reset_index(drop=True)
print("截断后训练集行数:", len(df))

# ---------------------------
# 3. 特征列 & 缺失率计算（基于截断后的 df）
# ---------------------------
feature_cols = imp_df.index.tolist()  # importance 的行名就是特征名
missing_rate = df[feature_cols].isna().mean()  # 每个特征的缺失比例

# ---------------------------
# 4. 非特征列（例如目标、时间列等，一律保留）
# ---------------------------
non_feature_cols = [c for c in df.columns if c not in feature_cols]
print("非特征列（始终保留）:", non_feature_cols)

# ---------------------------
# 5. 阈值列表（mean_importance 小于阈值的特征删掉）
# ---------------------------
thresholds = [0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

for thr in thresholds:
    # A. importance 过滤：mean_importance >= 阈值 才保留
    imp_mask = imp_df["mean_importance"] >= thr

    # B. 缺失率过滤：
    #    现在规则是：缺失率 < 0.4 的特征保留，且 M6 一定保留
    #    => 只要 (missing_rate < 0.4) 或 (特征名 == "M6") 就保留
    miss_mask = (missing_rate < 0.4) | (missing_rate.index == "M6")

    # C. 最终筛选条件（同时满足 importance 和 缺失率）
    final_mask = imp_mask & miss_mask

    selected_features = imp_df.index[final_mask].tolist()

    # D. 构建最终数据集：非特征列 + 筛选后的特征列
    cols_to_keep = non_feature_cols + selected_features
    filtered_df = df[cols_to_keep]

    # **新增：将剩下的缺失值补 0**
    filtered_df = filtered_df.fillna(0)

    # E. 保存 CSV
    thr_str = str(thr).replace(".", "p")
    out_path = f"../data/cropped/train_filtered_threshold_{thr_str}.csv"
    filtered_df.to_csv(out_path, index=False, encoding="utf-8")

    print(
        f"[阈值 {thr}] 保留特征 {len(selected_features)} 个，总列 {len(cols_to_keep)} → 已保存到: {out_path}"
    )