import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# 读取数据
df = pd.read_csv('data/train.csv')

# --- 设置绘图风格 (ICLR/学术风格) ---
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体 (如 Times New Roman)
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12

# --- 1. 颜色定义 ---
# 0 (False/Present) -> 深色 (e.g., 深海军蓝/黑色)
# 1 (True/Missing)  -> 亮色 (e.g., 浅米色/白色/浅黄)
# 这种配色打印成黑白也不会丢失信息
colors = ['#2C3E50', '#F5B041']  # 深蓝灰 (有数据), 亮橙黄 (缺失)
cmap = ListedColormap(colors)

# --- 2. 绘图 ---
plt.figure(figsize=(12, 6)) # 调整为长宽比 2:1，适合论文栏宽
ax = sns.heatmap(df.isnull(), 
                 cbar=False,  # 关闭默认的颜色条，我们自己画图例
                 cmap=cmap,
                 xticklabels=False, # 隐藏X轴密密麻麻的特征名
                 yticklabels=1000)  # Y轴每1000行显示一个刻度

# --- 3. 标注 "黄金区间" (Golden Interval) ---
# 这一步非常重要，直接呼应你的论文文字分析
clean_start = 6969
clean_end = 9020

# 在Y轴左侧画一条线或者括号来标记这个区间
plt.axhline(y=clean_start, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.8)
plt.axhline(y=clean_end, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.8)

# 添加文字注释
plt.text(x=102, y=(clean_start + clean_end)/2, 
         s='Fully Populated\nInterval\n(Indices 6969-9020)', 
         color='#E74C3C', 
         fontsize=10, 
         va='center', 
         fontweight='bold')

# --- 4. 美化标签与标题 ---
plt.title('Heatmap of Feature Missingness over Time', fontsize=14, pad=20, weight='bold')
plt.xlabel('Features (Total: 94)', fontsize=12)
plt.ylabel('Time Steps (Row Index)', fontsize=12)

# --- 5. 自定义图例 ---
# 创建图例对象
present_patch = mpatches.Patch(color=colors[0], label='Data Present')
missing_patch = mpatches.Patch(color=colors[1], label='Data Missing')
plt.legend(handles=[present_patch, missing_patch], 
           loc='upper center', 
           bbox_to_anchor=(0.5, -0.05), # 放在图下方
           ncol=2, 
           frameon=False,
           fontsize=11)

plt.tight_layout()
plt.savefig('missing_heatmap_refined.png', dpi=300, bbox_inches='tight') # 高DPI保存
plt.show()