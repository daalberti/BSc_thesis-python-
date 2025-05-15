# code structure and syntax was created with the help of ChatGPT.

# uses the "all_FDC_per_elevation_band.gpkg" file created in "6-all_elevation_band_analysis_FINAL--.py" to plot elevation band vs. fractional debris cover (weighted and unweighted) for all glaciers across regions
# calculates and plots distribution of the maximal difference elevation band


from rasterio.io import MemoryFile
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
from rasterio.mask import mask
from rasterio.features import rasterize
import rasterio
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib as matplotlib

# Set global font
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10


final_gdf = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/all_together/all_FDC_per_elevation_band.gpkg")



#Compute modal year per region and source
modal_per_region_source = (
    final_gdf
    .dropna(subset=['img_year'])
    .groupby(['region', 'source'])['img_year']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
    .reset_index()
)

# Compute the mode of those modal years for each source
modal_year = (
    modal_per_region_source
    .groupby('source')['img_year']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
)

#  Unpack to variables
year_lx  = modal_year.get('lx', pd.NA)
year_sam = modal_year.get('sam', pd.NA)
year_s2  = modal_year.get('s2', pd.NA)

source_labels = {
    'lx': f"lx ({year_lx})",
    'sam': f"sam ({year_sam})",
    's2': f"s2 ({year_s2})"
}

# Melt band_1 to band_20 into long format
band_cols = [f'band_{i}' for i in range(1, 21)]
df_for_plots = final_gdf.melt(
    id_vars=['rgi_id', 'source', 'pixel_weight'], 
    value_vars=band_cols, 
    var_name='band', 
    value_name='ratio'
)

# Extract band number as integer
df_for_plots['band'] = df_for_plots['band'].str.extract(r'band_(\d+)').astype(int)

num_bands = 20
step = 1/num_bands
rel_elevs = [(b-0.5)*step for b in range(1, num_bands+1)]

# color map for only the three sources
colors = {   
    's2': '#009E73',    
    'lx': '#E69F00',    
    'sam': '#c4669a'
}

markers = {
    's2': 'o',     
    'lx': 's',     
    'sam': '^' 
}

#  Compute mean ratio per band and source
band_means_df = df_for_plots.groupby(['band', 'source'])['ratio'].mean().unstack()

# Step 2: Shift s2 values so min(s2) = 0
if 's2' in band_means_df.columns:
    min_x = band_means_df['s2'].min()
    band_means_df['s2'] = band_means_df['s2'] - min_x

s2_shift = min_x
print("s2_shift")
print(s2_shift)
#  Compute range across sources for each band
band_means_df['range'] = band_means_df.max(axis=1) - band_means_df.min(axis=1)

# Identify band with maximum difference
max_diff_band = band_means_df['range'].idxmax()




# (2) mean per band (with s2‐shift)
plt.suptitle("Relative Elevation vs. Debris Cover Ratio (All Glaciers Combined)", fontsize=26, font = 'serif')
plt.subplot(1,2,1)
for src, grp in df_for_plots.groupby('source'):
    means = grp.groupby('band')['ratio'].mean()
    vals = means.values
    marker = markers.get(src)
    plt.plot(vals, rel_elevs, marker=marker, linewidth=1, markersize=4,
             label=source_labels[src],
             color=colors[src])
    if src=='s2':
        shifted = vals - vals.min()
        plt.plot(shifted, rel_elevs, linestyle='--',
                 label="s2 shifted", marker=marker, markersize=4,
                 color=colors[src])
plt.xlabel("Mean Pixel Ratio", font = 'serif', fontsize = 18)
plt.ylabel("Relative Elevation", font = 'serif', fontsize = 18)
plt.title("Mean Ratio per Elevation Band per Dataset", font = 'serif', fontsize = 18)
plt.legend(fontsize = 17)
plt.axvline(0.5, linestyle=':', color='black', linewidth=1)
# Highlight band with biggest difference
band_bottom = (max_diff_band - 1) * step
band_top = max_diff_band * step
plt.axhspan(band_bottom, band_top, color='gray', alpha=0.3)
plt.tick_params(axis='both', labelsize=14)
plt.xlim(0, 1.0)




weighted_data = {}  
for src, group in df_for_plots.groupby('source'):
    weighted_means = []
    for b in range(1, num_bands + 1):
        band_df = group[group['band'] == b]
        weighted_vals = []
        total_weight = 0
        for _, row in band_df.iterrows():
            weight = row['pixel_weight']  
            weighted_vals.append(row['ratio'] * weight)
            total_weight += weight
        weighted_mean = sum(weighted_vals) / total_weight if total_weight > 0 else 0
        weighted_means.append(weighted_mean)
    weighted_data[src] = weighted_means  #save


    # Shift s2 before storing
    if src == 's2':
        min_x = min(weighted_means)
        weighted_means = [x - min_x for x in weighted_means]

    weighted_data[src] = weighted_means


#  Create DataFrame [bands x sources]
weighted_df = pd.DataFrame(weighted_data)
weighted_df.index = range(1, num_bands + 1)  # band 1 to 20

# Calculate inter-source range per band
weighted_df['range'] = weighted_df.max(axis=1) - weighted_df.min(axis=1)

#  Identify band with maximum difference
max_diff_band_weighted = weighted_df['range'].idxmax()

# Compute highlight boundaries
step = 1 / num_bands
band_bottom_w = (max_diff_band_weighted - 1) * step
band_top_w = max_diff_band_weighted * step




# (3) weighted mean by glacier area
plt.subplot(1,2,2)
for src, grp in df_for_plots.groupby('source'):
    weighted = []
    marker = markers.get(src)
    for b in range(1, num_bands+1):
        sub = grp[grp['band'] == b]
        total_w = sub['pixel_weight'].sum()  # <-- direkt summieren
        if total_w > 0:
            weighted_sum = (sub['ratio'] * sub['pixel_weight']).sum()  # <-- gewichtetes Produkt
            weighted.append(weighted_sum / total_w)
        else:
            weighted.append(0)
    plt.plot(weighted, rel_elevs, marker=marker, linewidth=1, markersize=4,
             label=source_labels[src],
             color=colors[src])
    if src == 's2':
        shifted = np.array(weighted) - min(weighted)
        plt.plot(shifted, rel_elevs, linestyle='--',
                 label="s2 shifted", marker=marker, markersize=4,
                 color=colors[src])

plt.xlabel("Weighted Mean Pixel Ratio", font = 'serif', fontsize = 18)
plt.title("Weighted Mean by Glacier Area per Dataset", font = 'serif', fontsize = 18)
plt.legend(fontsize = 17)
plt.axvline(0.5, linestyle=':', color='black', linewidth=1)
plt.axhspan(band_bottom_w, band_top_w, color='gray', alpha=0.3)
plt.tight_layout()
plt.tick_params(axis='both', labelsize=14)
plt.xlim(0, 1.0)
plt.show()


# boxplot: weighetd
# max_diff_band: corrected and weighted
def max_diff_band_per_glacier(df):
    # Step 1: mean ratio per band & source
    df = df.copy()
    
    # Apply s2 shift correction
    if 's2' in df['source'].unique():
        
        df.loc[df['source'] == 's2', 'ratio'] = df.loc[df['source'] == 's2', 'ratio'] - s2_shift
        df.loc[df['source'] == 's2', 'ratio'] = df.loc[df['source'] == 's2', 'ratio'].clip(lower=0)

    # Group and unstack to get mean ratio per band & source
    spread = df.groupby(['band', 'source'])['ratio'].mean().unstack()

    # Step 2: Range per band
    spread['range'] = spread.max(axis=1) - spread.min(axis=1)

    # Step 3: Max difference band
    return spread['range'].idxmax()


# Apply to each glacier group
glacier_max_diff_band = df_for_plots.groupby('rgi_id').apply(max_diff_band_per_glacier, include_groups=False).reset_index()
glacier_max_diff_band.columns = ['rgi_id', 'max_diff_band']

# Merge with original dataframe
df_for_plots = df_for_plots.merge(glacier_max_diff_band, on='rgi_id', how='left')



# Compute rounded mean value for plotting
mean_val = round(glacier_max_diff_band['max_diff_band'].mean(), 2)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(glacier_max_diff_band['max_diff_band'], bins=20, color='gray', edgecolor='black', alpha=0.7)
plt.xlabel('Max difference band per glacier', font = 'serif', fontsize = 18)
plt.ylabel('Number of Glaciers', font = 'serif', fontsize = 18)
plt.title('Maximal Difference Elevetion Bands Across Glaciers (All Glaciers Combined)\n', font = 'serif', fontsize = 26)
plt.tick_params(axis='both', labelsize=14)
# Plot mean line
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'rounded Mean = {mean_val:.2f}')

# Add legend
plt.legend(fontsize = 18)
plt.tight_layout()
plt.show()
