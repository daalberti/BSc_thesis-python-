# Structure and syntax was created with the aid of ChatGPT


#what this code does:
# - loop through selected RGI 7.0 regions
# - Takes the combined dataset as input
# - extracts the most common year of each dataset
# - Plots a histogram for change in fractional debris cover between datasets (upwards in time)



import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10

regions = [
    "01Alaska", "10NorthAsia", "11CentralEurope",
    "12CaucasusMiddleEast", "14SouthAsiaWest",
    "15SouthAsiaEast", "18NewZealand"
]

dir = "C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis"

for region in regions:
    out_dir = os.path.join(dir, region)
    combined_path = f"{out_dir}/{region}_combined.gpkg"

    combined = gpd.read_file(combined_path)

    modal_year = (
        combined
        .groupby('source')['img_year']
        .agg(lambda x: x.mode().iloc[0])
    )

    year_lx, year_sam, year_s2 = (
        modal_year['lx'],
        modal_year['sam'],
        modal_year['s2']
    )

    df_wide = (
        combined
        .pivot(index="rgi_id", columns="source", values="debris_cover_ratio")
        .reset_index()
    )

    df_wide = df_wide.rename(columns={"lx": "debris_lx", "sam": "debris_sam", "s2": "debris_s2"})

    df_wide["change_lx_sam"] = df_wide["debris_sam"] - df_wide["debris_lx"]
    df_wide["change_sam_s2"] = df_wide["debris_s2"] - df_wide["debris_sam"]
    df_wide["change_lx_s2"] = df_wide["debris_s2"] - df_wide["debris_lx"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), tight_layout=True)


        # LX -> SAM
    data_ls = df_wide["change_lx_sam"].dropna()
    μ_ls, σ_ls = data_ls.mean(), data_ls.std()
    counts_ls, _, _ = axes[0].hist(data_ls, bins=30, edgecolor='black')
    axes[0].axvline(μ_ls, color='red', linestyle='--', label=f'μ={μ_ls:.3f}')
    axes[0].axvline(μ_ls+σ_ls, color='red', linestyle=':', label=f'+1σ={σ_ls:.3f}')
    axes[0].axvline(μ_ls-σ_ls, color='red', linestyle=':')
    axes[0].set_title(f"LX ({year_lx}) → SAM ({year_sam})", fontsize=17.5)
    axes[0].set_xlabel("Fractional Debris Cover Change",fontsize=17)
    axes[0].set_ylabel("Frequency",fontsize=17)

    axes[0].set_xlim(-0.8, 0.8)
    axes[0].axvline(0, color='grey', linewidth=2, linestyle='-', zorder=1)
    axes[0].legend(fontsize=15, loc='upper left')

    # SAM -> S2
    data_ss = df_wide["change_sam_s2"].dropna()
    μ_ss, σ_ss = data_ss.mean(), data_ss.std()
    counts_ss, _, _ = axes[1].hist(data_ss, bins=30, edgecolor='black')
    axes[1].axvline(μ_ss, color='red', linestyle='--', label=f'μ={μ_ss:.3f}')
    axes[1].axvline(μ_ss+σ_ss, color='red', linestyle=':', label=f'+1σ={σ_ss:.3f}')
    axes[1].axvline(μ_ss-σ_ss, color='red', linestyle=':')
    axes[1].set_title(f"SAM ({year_sam}) → S2 ({year_s2})", fontsize=17.5)
    axes[1].set_xlabel("Fractional Debris Cover Change",fontsize=17)

    axes[1].set_xlim(-0.8, 0.8)
    axes[1].axvline(0, color='grey', linewidth=2, linestyle='-', zorder=1)
    axes[1].legend(fontsize=15, loc='upper left')

    # LX -> S2
    data_ls2 = df_wide["change_lx_s2"].dropna()
    μ_ls2, σ_ls2 = data_ls2.mean(), data_ls2.std()
    counts_ls2, _, _ = axes[2].hist(data_ls2, bins=30, edgecolor='black')
    axes[2].axvline(μ_ls2, color='red', linestyle='--', label=f'μ={μ_ls2:.3f}')
    axes[2].axvline(μ_ls2+σ_ls2, color='red', linestyle=':', label=f'+1σ={σ_ls2:.3f}')
    axes[2].axvline(μ_ls2-σ_ls2, color='red', linestyle=':')
    axes[2].set_title(f"LX ({year_lx}) → S2 ({year_s2})", fontsize=17.5)
    axes[2].set_xlabel("Fractional Debris Cover Change",fontsize=17)

    axes[2].set_xlim(-0.8, 0.8)
    axes[2].axvline(0, color='grey', linewidth=2, linestyle='-', zorder=1)
    axes[2].legend(fontsize=15, loc='upper left')

    max_count = max(np.max(counts_ls), np.max(counts_ss), np.max(counts_ls2)) * 1.05
    axes[0].set_ylim(0, int(max_count))
    axes[1].set_ylim(0, int(max_count))
    axes[2].set_ylim(0, int(max_count))

    plt.suptitle(f"Change in Debris Cover – {region}", fontsize=26)
    plt.savefig(f"{out_dir}/{region}_change_histogramm.png")
    plt.tight_layout()
    plt.close()
