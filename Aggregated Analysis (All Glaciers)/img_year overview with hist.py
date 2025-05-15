# code structure and syntax was created with the help of ChatGPT.

# - extracts img_year of each glacier from the combined and Switzerland file
# - plots distributions of img_year for each region 

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
import numpy as np
import geopandas as gpd
# Set global font
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10

# Load data
gdf_ch = gpd.read_file("c:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/Switzerland_combined.gpkg")
gdf_combined = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/all_together/all_regions_combined.gpkg")


gdf_combined.loc[gdf_combined["img_year"] == 0, "img_year"] = pd.NA
df = gdf_combined.dropna(subset=["img_year"])
df["img_year"] = pd.to_numeric(df["img_year"], errors="coerce")
df = df.dropna(subset=["img_year"])
df["img_year"] = df["img_year"].astype(int)





gdf_ch = gdf_ch.dropna(subset=["img_year"]).copy()
gdf_ch["img_year"] = gdf_ch["img_year"].astype(int)
gdf_ch["region"] = "Swiss Alps"

df_swiss = gdf_ch[["region", "source", "img_year"]]


df = pd.concat([df, df_swiss], ignore_index=True)


source_colors = {
    'sgi': '#0072B2',    
    'lx': '#E69F00',    
    'sam': '#c4669a',
    's2': '#009E73'    
}
bins = np.arange(1990, 2025, 1)
final_regions = ['Swiss Alps'] + list(gdf_combined['region'].unique())

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()


for i, region in enumerate(final_regions):
    ax = axes[i]
    region_data = df[df["region"] == region]

    sources_in_region = region_data["source"].unique()
    n_sources = len(sources_in_region)
    
    # Wider bars, adjusted offset range
    bar_width = 0.4
    spacing = np.linspace(-0.3, 0.3, n_sources)  # Dynamic offsets for clarity

    for j, (source, offset) in enumerate(zip(sources_in_region, spacing)):
        sub_data = region_data[region_data["source"] == source]["img_year"]
        counts, _ = np.histogram(sub_data, bins=bins)
        ax.bar(bins[:-1] + offset, counts, width=bar_width,
               color=source_colors.get(source, 'gray'), edgecolor='black', label=source.upper())

    ax.set_xlim(1990, 2024)
    ax.set_xticks(np.arange(1990, 2025, 10))
    ax.set_ylim(0, None)
    ax.set_title(region, fontsize = 20)
    ax.tick_params(axis='both', labelsize=13)
    if i % 3 == 0:
        ax.set_ylabel('Count', fontsize=16.5)
    if i >= 6:
        ax.set_xlabel('Year', fontsize=16.5)



legend_ax = axes[8]
legend_ax.axis('off')
legend_handles = [
    plt.Line2D([0], [0], color=color, lw=6, label=src.upper())
    for src, color in source_colors.items()
]
legend_ax.legend(
    handles=legend_handles,
    loc='center',
    fontsize=18,
    title='Dataset',
    title_fontsize=18,
    frameon=False
)

plt.tight_layout()
plt.show()
