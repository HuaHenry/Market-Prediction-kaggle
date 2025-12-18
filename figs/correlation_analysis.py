import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
df = pd.read_csv('data/train.csv')

# 2. 定义特征和目标
features = [
    "E20", "E7", "M1", "M13", "M14", "M2", "M3", "M5", "M6",
    "P5", "P6", "P7", "S12", "S3", "S5", "S8",
    "V10", "V13", "V5", "V7", "V9",
]
targets = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]

# 过滤数据中实际存在的列
features_exist = [c for c in features if c in df.columns]
targets_exist = [c for c in targets if c in df.columns]

# 3. 计算三个相关系数矩阵
methods = ['Pearson', 'Spearman', 'Kendall']
corr_results = {m: pd.DataFrame(index=features_exist, columns=targets_exist) for m in methods}

for ft in features_exist:
    for tgt in targets_exist:
        x = df[ft]
        y = df[tgt]
        # 逐个计算
        corr_results['Pearson'].loc[ft, tgt] = x.corr(y, method='pearson')
        corr_results['Spearman'].loc[ft, tgt] = x.corr(y, method='spearman')
        corr_results['Kendall'].loc[ft, tgt] = x.corr(y, method='kendall')

# 转换为 float 类型以防万一
for m in methods:
    corr_results[m] = corr_results[m].astype(float)

# 4. 排序：按 Pearson 绝对值总和排序，让重要的特征排在前面
sort_idx = corr_results['Pearson'].abs().sum(axis=1).sort_values(ascending=False).index
for m in methods:
    corr_results[m] = corr_results[m].loc[sort_idx]

# 5. 可视化绘图
# 设置 ICLR / 学术风格字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

fig, axes = plt.subplots(1, 3, figsize=(20, 10), sharey=True)

# 统一的 Colorbar 轴
cbar_ax = fig.add_axes([.92, .3, .02, .4]) 

for i, method in enumerate(methods):
    sns.heatmap(abs(corr_results[method]), 
                ax=axes[i],
                annot=True,       # 显示数值
                fmt=".2f",        # 保留两位小数
                cmap="coolwarm",  # 红蓝配色
                vmin=0, vmax=1,  # 锁定颜色范围
                cbar= (i == 0),   # 只画一次图例
                cbar_ax=None if i else cbar_ax,
                linewidths=0.5,   # 格子间距
                square=False)     # 不需要强制正方形
    
    axes[i].set_title(f'{method} Correlation', fontsize=14, weight='bold', pad=15)
    axes[i].set_xlabel('')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    if i == 0:
        axes[i].set_ylabel('Features (Sorted by Importance)', fontsize=12)
    else:
        axes[i].set_ylabel('')

plt.suptitle('Correlation Analysis of Missing Features vs Target Variables', 
             fontsize=16, weight='bold', y=0.95)

plt.savefig('correlation_heatmap_comparison.png', bbox_inches='tight', dpi=300)
plt.show()