# code structure and syntax was created with the aid of ChatGPT.

# - visualizes the change in fractional debris cover between datasets in histograms using the combined file created in "3-all_match_and_combine_NEW.py".

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as matplotlib

# Set global font
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10


combined = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/all_together/all_regions_combined.gpkg")

# Step 1: Compute modal year per region and source
modal_per_region_source = (
    combined
    .dropna(subset=['img_year'])
    .groupby(['region', 'source'])['img_year']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
    .reset_index()
)

# Step 2: Compute the mode of those modal years for each source
modal_year = (
    modal_per_region_source
    .groupby('source')['img_year']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
)

# Step 3: Unpack to variables
year_lx  = modal_year.get('lx', pd.NA)
year_sam = modal_year.get('sam', pd.NA)
year_s2  = modal_year.get('s2', pd.NA)

print(year_lx)
print(year_sam)
print(year_s2)

# Pivot to wide format: one row per glacier
df_wide = (combined
           .pivot(index="rgi_id", columns="source", values="debris_cover_ratio")
           .reset_index())

# Rename columns for clarity
df_wide = df_wide.rename(columns={"lx": "debris_lx", "sam": "debris_sam", "s2": "debris_s2"})

# Compute changes
df_wide["change_lx_sam"] = df_wide["debris_sam"] - df_wide["debris_lx"]
df_wide["change_sam_s2"] = df_wide["debris_s2"] - df_wide["debris_sam"]
df_wide["change_lx_s2"]  = df_wide["debris_s2"] - df_wide["debris_lx"]

# Plot histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4), tight_layout=True)


# LX -> SAM
data_ls = df_wide["change_lx_sam"].dropna()
μ_ls, σ_ls = data_ls.mean(), data_ls.std()
axes[0].hist(data_ls, bins=40, edgecolor='black')
axes[0].axvline(μ_ls, color='red', linestyle='--', label=f'μ={μ_ls:.3f}')
axes[0].axvline(μ_ls+σ_ls, color='red', linestyle=':', label=f'+1σ={σ_ls:.3f}')
axes[0].axvline(μ_ls-σ_ls, color='red', linestyle=':')
axes[0].set_title(f"LX ({year_lx}) → SAM ({year_sam})",font = 'serif', fontsize = 18)
axes[0].set_xlabel("Fractional Debris Cover Change", font = 'serif', fontsize = 16.5)
axes[0].set_ylabel("Frequency", font = 'serif', fontsize = 16.5)
axes[0].set_ylim(0, 800)
axes[0].set_yticks(range(0, 601, 100))
axes[0].set_xlim(-0.6, 0.6)
axes[0].axvline(0, color='grey', linewidth=2, linestyle='-', zorder=1)
axes[0].legend(fontsize=16.5, loc = 'upper left')
axes[0].tick_params(axis='both', labelsize=13)

# SAM -> S2
data_ss = df_wide["change_sam_s2"].dropna()
μ_ss, σ_ss = data_ss.mean(), data_ss.std()
axes[1].hist(data_ss, bins=40, edgecolor='black')
axes[1].axvline(μ_ss, color='red', linestyle='--', label=f'μ={μ_ss:.3f}')
axes[1].axvline(μ_ss+σ_ss, color='red', linestyle=':', label=f'+1σ={σ_ss:.3f}')
axes[1].axvline(μ_ss-σ_ss, color='red', linestyle=':')
axes[1].set_title(f"SAM ({year_sam}) → S2 ({year_s2})",font = 'serif', fontsize = 18)
axes[1].set_xlabel("Fractional Debris Cover Change", font = 'serif', fontsize = 16.5)
axes[1].set_ylim(0, 800)
axes[1].set_yticks(range(0, 601, 100))
axes[1].set_xlim(-0.6, 0.6)
axes[1].axvline(0, color='grey', linewidth=2, linestyle='-', zorder=1)
axes[1].legend(fontsize=16.5, loc = 'upper left')
axes[1].tick_params(axis='both', labelsize=13)

# LX -> S2
data_ls2 = df_wide["change_lx_s2"].dropna()
μ_ls2, σ_ls2 = data_ls2.mean(), data_ls2.std()
axes[2].hist(data_ls2, bins=40, edgecolor='black')
axes[2].axvline(μ_ls2, color='red', linestyle='--', label=f'μ={μ_ls2:.3f}')
axes[2].axvline(μ_ls2+σ_ls2, color='red', linestyle=':', label=f'+1σ={σ_ls2:.3f}')
axes[2].axvline(μ_ls2-σ_ls2, color='red', linestyle=':')
axes[2].set_title(f"LX ({year_lx}) → S2 ({year_s2})", font = 'serif',fontsize = 18)
axes[2].set_xlabel("Fractional Debris Cover Change", font = 'serif', fontsize = 16.5)
axes[2].set_ylim(0, 800)
axes[2].set_yticks(range(0, 601, 100))
axes[2].set_xlim(-0.6, 0.6)
axes[2].axvline(0, color='grey', linewidth=2, linestyle='-', zorder=1)
axes[2].legend(fontsize=16.5,loc = 'upper left')
axes[2].tick_params(axis='both', labelsize=13)
plt.suptitle("Difference in fractional Debris Cover (All Glaciers Combined)\n", fontsize=26, font='serif')
plt.show()

