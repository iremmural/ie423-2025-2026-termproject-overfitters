import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PATH SETUP
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
figures_path = os.path.join(project_root, "visuals", "figures")
tables_path  = os.path.join(project_root, "visuals", "tables")
os.makedirs(figures_path, exist_ok=True)
os.makedirs(tables_path, exist_ok=True)


df = pd.read_csv(os.path.join(project_root, "data", "processed", "processed_food_crisis_data.csv"))

# ---------------------------------------------------------
# SUMMARY DESCRIPTIVE SUMMARY TABLE
# ---------------------------------------------------------
df_filtered = df.drop(columns=['median_price', 'max_price', 'price_std', 'month', 'crisis_label'], errors='ignore')
stats_summary = df_filtered.describe(percentiles=[.25, .5, .75]).T
stats_summary.columns = ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max']

stats_summary.to_csv(os.path.join(tables_path, "summary_descriptive_statistics.csv"))

# ---------------------------------------------------------
# FIGURE 1: GLOBAL FOOD INSECURITY RANKINGS BAR CHART
# ---------------------------------------------------------
all_avg_scores = df.groupby('country_name')['fao_score'].mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(10, 20))

custom_palette = sns.cubehelix_palette(n_colors=len(all_avg_scores), start=.5, rot=-.5, reverse=True)
sns.barplot(
    data=all_avg_scores,
    x='fao_score',
    y='country_name',
    hue='country_name',
    palette=custom_palette,
    legend=False
)

plt.title('Global Food Insecurity Rankings', fontweight='bold', fontsize=14, pad=15)
plt.xlabel('Average Food Insecurity Score (FAO)', fontsize=12)
plt.ylabel('Country', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, "01_global_food_insecurity_rankings.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# FIGURE 2: GLOBAL PRODUCT CRISIS MAP BUBBLE CHART
# ---------------------------------------------------------
product_analysis = df.groupby('product').agg({
    'crisis_label': 'sum',
    'volatility_ratio_3': 'mean',
    'fao_score': 'mean'
}).reset_index()

top_20 = product_analysis.nlargest(20, 'crisis_label')

plt.figure(figsize=(14, 10))
sns.set_theme(style="whitegrid")

scatter = plt.scatter(
    x=top_20['volatility_ratio_3'],
    y=top_20['fao_score'],
    s=top_20['crisis_label'] * 15, # scale bubble size by crisis count
    alpha=0.6,
    c=top_20['fao_score'],
    cmap='YlOrRd',
    edgecolors="grey",
    linewidth=2
)

for i, row in top_20.iterrows():
    plt.annotate(row['product'], (row['volatility_ratio_3'], row['fao_score']),
                 fontsize=9, fontweight='bold', ha='center', va='center', xytext=(0, 15), textcoords='offset points')

plt.title('Global Product Crisis Map', fontsize=18, fontweight='bold', pad=30)
plt.xlabel('Average Price Volatility Ratio')
plt.ylabel('Mean Food Insecurity Score (FAO)')
plt.colorbar(scatter).set_label('Food Insecurity Intensity')
plt.savefig(os.path.join(figures_path, "02_global_product_crisis_map.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# FIGURE 3: THE RELATIONSHIP BETWEEN PRICE CHANGES AND CRISIS BOX PLOT
# ---------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 7))

df['crisis_label_str'] = df['crisis_label'].astype(str)

sns.boxplot(
    data=df,
    x='crisis_label_str',
    y='pct_change_1m',
    hue='crisis_label_str',
    palette={'0': 'lightgreen', '1': 'salmon'},
    legend=False
)

plt.title('The Relationship Between Price Changes and Crisis',fontweight='bold', fontsize=14, pad=15)
plt.xlabel('Crisis Status (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Monthly Price Change (%)', fontsize=12)

plt.ylim(-25, 40)
plt.grid(True, axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "03_price_change_vs_crisis.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# FIGURE 4: CORRELATION OF CRISIS PREDICTION VARIABLES HEATMAP
# ---------------------------------------------------------
cols_to_corr = [
    'avg_price', 'volatility_ratio_3', 'volatility_ratio_6',
    'price_lag_1', 'price_lag_3', 'price_lag_6',
    'fao_score', 'crisis_label' # food_insecurity_score -> fao_score
]

corr_matrix = df[cols_to_corr].corr()
plt.figure(figsize=(12, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap=sns.color_palette("Blues", as_cmap=True),
    fmt=".2f"
)

plt.title('Correlation of Crisis Prediction Variables', fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "04_correlation_heatmap.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# FIGURE 5: DISTRIBUTION OF CRISIS LABELS BAR CHART
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
ax = sns.countplot(
    x='crisis_label',
    data=df,
    hue='crisis_label',
    palette='hls',
    legend=False
)

plt.title("Distribution of Crisis Labels", fontsize=14, fontweight='bold')
plt.xlabel('Crisis Label', fontsize=12)
plt.ylabel('Count', fontsize=12)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 9),
                textcoords = 'offset points',
                fontsize=11)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "05_crisis_label_distribution.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# FIGURE 6: ANNUAL CRISIS INTENSITY HEATMAP
# ---------------------------------------------------------

pivot_all = df.groupby(['country_name', 'year'])['crisis_label'].mean().unstack()
pivot_no_gaps = pivot_all.dropna()
top_25_countries_data = pivot_no_gaps.assign(avg=pivot_no_gaps.mean(axis=1)) \
                                     .sort_values(by='avg', ascending=False) \
                                     .head(25) \
                                     .drop(columns='avg')

plt.figure(figsize=(14, 12))
sns.heatmap(
    top_25_countries_data,
    annot=True,
    cmap='YlOrRd',
    fmt=".2f",
    linewidths=.5,
    cbar_kws={'label': 'Crisis Intensity (0.0 - 1.0)'}
)

plt.title('Annual Crisis Intensity', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "06_annual_crisis_intensity.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# FIGURE 7: SEASONAL PATTERNS — MONTHLY CRISIS RATE & PRICE VOLATILITY
# ---------------------------------------------------------
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

monthly_crisis = df.groupby('month')['crisis_label'].mean().reset_index()
monthly_crisis['month_name'] = monthly_crisis['month'].apply(lambda x: MONTH_NAMES[int(x) - 1])

monthly_vol = df.groupby('month')['volatility_ratio_3'].mean().reset_index()
monthly_vol['month_name'] = monthly_vol['month'].apply(lambda x: MONTH_NAMES[int(x) - 1])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.set_theme(style="whitegrid")

avg_crisis = monthly_crisis['crisis_label'].mean()
bar_colors = ['#d73027' if v > avg_crisis else '#fee090' for v in monthly_crisis['crisis_label']]
axes[0].bar(monthly_crisis['month_name'], monthly_crisis['crisis_label'],
            color=bar_colors, edgecolor='grey', linewidth=0.7)
axes[0].axhline(avg_crisis, color='dimgrey', linestyle='--', alpha=0.8, label=f'Annual avg: {avg_crisis:.3f}')
axes[0].legend(fontsize=9)
axes[0].set_title('Average Crisis Rate by Month', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Month', fontsize=11)
axes[0].set_ylabel('Average Crisis Rate', fontsize=11)

axes[1].plot(monthly_vol['month_name'], monthly_vol['volatility_ratio_3'],
             marker='o', color='steelblue', linewidth=2, markersize=7)
axes[1].axhline(monthly_vol['volatility_ratio_3'].mean(), color='dimgrey', linestyle='--',
                alpha=0.8, label=f'Annual avg: {monthly_vol["volatility_ratio_3"].mean():.3f}')
axes[1].legend(fontsize=9)
axes[1].set_title('Average Price Volatility by Month', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Month', fontsize=11)
axes[1].set_ylabel('Avg Volatility Ratio (3-month)', fontsize=11)

fig.suptitle('Seasonal Patterns in Food Price Crisis', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "07_seasonal_patterns.png"), bbox_inches="tight")
plt.close()

