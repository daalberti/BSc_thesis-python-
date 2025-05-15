#what this code does:

# - For each glacier (grouped by rgi_id and source):
#       - Extracts elevation values from the DEM for the glacier extent.
#       - Divides the elevation range into 20 uniform bands.
#       - Calculates, for each band, how much of it is covered by debris
#       - calculate debris cover ratio per elevation band (glacier with debris/ total glacier)
# plot scatterplot, mean plot and weighted mean plot (by glacier area): relative elevation vs. debris ratio



import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
from rasterio.mask import mask
from rasterio.features import rasterize
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from rasterio.plot import plotting_extent

# Parameters
num_bands = 20

# Load data
glacier_rgi7 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/RGI_7_outline_11.geojson")
glacier_sam = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/SAM_matching_glacier_outlines.gpkg")
glacier_sgi = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/SGI_matching_glacier_outlines.gpkg")
debris = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/Switzerland_combined.gpkg")
dem_path = "C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/DEM/DEM_Subregion_11-01.tif" 
src_dem = rasterio.open(dem_path)

print(src_dem.crs)
print(glacier_sam.crs)

# ensure crs is epsg 2056
glacier_rgi7 = glacier_rgi7.to_crs(epsg=2056)
glacier_sam = glacier_sam.to_crs(epsg=2056)
glacier_sgi = glacier_sgi.to_crs(epsg=2056)
debris = debris.to_crs(epsg=2056)

# Prepare list to store results
all_results = []
glacier_pixel_weights = {}






from collections import defaultdict

# Track stats
glaciers_with_nan_bands = set()
nan_counts_by_source_band = defaultdict(lambda: defaultdict(int))

# Loop through each glacier ID in debris
for (rgi_id, source), group in debris.groupby(['rgi_id', 'source']):
    glacier_for_bins = glacier_rgi7  # used for elevation binning

    # Choose glacier dataset for ratio calculation
    if source == 'lx' or source == 's2':
        glacier = glacier_rgi7
    elif source == 'sam':
        glacier = glacier_sam
    elif source == 'sgi':
        glacier = glacier_sgi
    else:
        continue  # Skip if no valid source

    glacier_geom_bins = [mapping(geom) for geom in glacier_for_bins[glacier_for_bins['rgi_id'] == rgi_id].geometry]
    glacier_geom_main = [mapping(geom) for geom in glacier[glacier['rgi_id'] == rgi_id].geometry]
    debris_subset = group.copy()

    try:
        # DEM clipped to glacier (for actual analysis)
        glacier_dem, glacier_transform = mask(src_dem, glacier_geom_main, crop=True)
        # DEM for binning (based on RGI7 glacier)
        bins_dem, _ = mask(src_dem, glacier_geom_bins, crop=True)
    except Exception as e:
        print(f"Skipping {rgi_id} due to masking error: {e}")
        continue

    glacier_dem = glacier_dem[0]
    bins_dem = bins_dem[0]
    glacier_dem = np.where(glacier_dem == src_dem.nodata, np.nan, glacier_dem)
    bins_dem = np.where(bins_dem == src_dem.nodata, np.nan, bins_dem)
    glacier_dem[glacier_dem == 0] = np.nan
    bins_dem[bins_dem == 0] = np.nan

    if np.all(np.isnan(glacier_dem)) or np.all(np.isnan(bins_dem)):
        print(f"Skipping {rgi_id} due to empty DEM")
        continue

    # Total glacier pixels (used for weights)
    glacier_mask = rasterize(
        [(geom, 1) for geom in glacier[glacier['rgi_id'] == rgi_id].geometry],
        out_shape=glacier_dem.shape,
        transform=glacier_transform,
        fill=0,
        dtype='uint8'
    )
    glacier_area_pixels = np.sum(glacier_mask == 1)
    glacier_pixel_weights[rgi_id] = glacier_area_pixels

    # Bin calculation using glacier_for_bins DEM
    elev_min, elev_max = np.nanmin(bins_dem), np.nanmax(bins_dem)
    bin_size = (elev_max - elev_min) / num_bands
    normalized = (glacier_dem - elev_min) / bin_size
    band_indices = np.full_like(normalized, np.nan)  # same shape, all NaN
    valid_mask = ~np.isnan(normalized)
    band_indices[valid_mask] = normalized[valid_mask].astype(int) + 1
    band_indices = np.clip(band_indices, 1, num_bands)

    out_shape = glacier_dem.shape

    # Initialize band columns
    for b in range(1, num_bands + 1):
        debris_subset[f'band_{b}'] = np.nan

    # Loop through debris polygons
    for idx, row in debris_subset.iterrows():
        geom = row.geometry
        geom_mask = rasterize(
            [(geom, 1)],
            out_shape=out_shape,
            transform=glacier_transform,
            fill=0,
            dtype='uint8'
        )

        # Loop through bands
        for b in range(1, num_bands + 1):
            band_mask = (band_indices == b)

            if np.sum(~np.isnan(band_mask)) == 0 or not np.any(band_mask):
                # Band is outside elevation range
                nan_counts_by_source_band[source][b] += 1
                glaciers_with_nan_bands.add(rgi_id)
                continue

            overlap = (geom_mask == 1) & band_mask
            total_pixels = np.sum(band_mask)
            count = np.sum(overlap)

            if total_pixels == 0:
                ratio = np.nan
                print(f'{rgi_id} in {source}: total pixels = 0 in {b} elevation band')
                nan_counts_by_source_band[source][b] += 1
                glaciers_with_nan_bands.add(rgi_id)
            else:
                ratio = float(count) / float(total_pixels)

            debris_subset.at[idx, f'band_{b}'] = ratio

    all_results.append(debris_subset)

print(f"\nTotal glaciers with at least one NaN band: {len(glaciers_with_nan_bands)}")

nan_df = pd.DataFrame(nan_counts_by_source_band).fillna(0).astype(int)
nan_df.index.name = 'band'
nan_df.columns.name = 'source'
print("\nNaN counts by band and source:")
print(nan_df)


# Concatenate results and export
final_gdf = pd.concat(all_results, ignore_index=True)
final_gdf = gpd.GeoDataFrame(final_gdf, geometry='geometry', crs=debris.crs)

# Save to file
output_path = "C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/Switzerland_debris_ratio_per_elevation_band"
#final_gdf.to_file(output_path, driver="GPKG")

print("Processing complete. Output saved.")

# Plotting
step = 1 / num_bands
scatter_data = []
step = 1 / num_bands
rel_elevs = [(b - 0.5) * step for b in range(1, num_bands + 1)]

for i, row in final_gdf.iterrows():
    rgi_id = row['rgi_id']
    source = row['source']
    for b in range(1, num_bands + 1):
        ratio = row[f'band_{b}']
        if ratio >= 0:
            rel_elev = rel_elevs[b - 1]  # b in 1..num_bands → index 0..num_bands-1
            jittered_elev = rel_elev + np.random.uniform(-0.01, 0.01)  # jitter
            scatter_data.append((b, ratio, rgi_id, source, jittered_elev))


scatter_df = pd.DataFrame(scatter_data, columns=['band', 'ratio', 'rgi_id', 'source', 'rel_elev'])

source_order = ['lx', 'sgi', 'sam', 's2']
scatter_df['source'] = pd.Categorical(scatter_df['source'], categories=source_order, ordered=True)

scatter_df = scatter_df.sort_values(by=['source', 'band', 'rel_elev']).reset_index(drop=True)

colors = {
    'sgi': '#4E5C78',   
    's2': '#249f60',    
    'lx': '#e99800',    
    'sam': '#b21f00'    
}


source_years = final_gdf.groupby('source')['img_year'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)


plt.figure(figsize=(18, 5))
plt.suptitle("Glacier Outlines based on RGI 7, Sam Herreid and SGI, respectively (uniform elevation bands)", fontsize=16)
plt.subplot(1, 3, 1)
for src, group in scatter_df.groupby('source'):
    year = source_years.get(src, "n/a")
    label = f"{src} ({year})"
    plt.scatter(group['ratio'], group['rel_elev'],
                alpha=0.7, s=5, edgecolors='none',
                label=label, color=colors.get(src, 'gray'))
plt.xlabel("Pixel Ratio")
plt.ylabel("Relative Elevation (0-1)")
plt.title("Scatterplot: Ratio vs Rel. Elevation by Source")
plt.legend(loc='upper right')


# calculate max difference-band
#Step 1: Compute mean ratio per band and source
band_means_df = scatter_df.groupby(['band', 'source'])['ratio'].mean().unstack()

# Step 2: Shift s2 values so min(s2) = 0
if 's2' in band_means_df.columns:
    min_x = band_means_df['s2'].min()
    band_means_df['s2'] = band_means_df['s2'] - min_x

# Step 3: Compute range across sources for each band
band_means_df['range'] = band_means_df.max(axis=1) - band_means_df.min(axis=1)

# Step 4: Identify band with maximum difference
max_diff_band = band_means_df['range'].idxmax()


# Mean per band (simple average by source)
plt.subplot(1, 3, 2)

for src, group in scatter_df.groupby('source'):
    year = source_years.get(src, "n/a")
    label = f"{src} ({year})"
    band_means = group.groupby('band')['ratio'].mean()
    x_vals = band_means.values
    # Center the elevation value within each band
    
    
    plt.plot(band_means.values, rel_elevs, marker='.', linewidth=1, markersize=5,
             label=label, color=colors.get(src, 'gray'))
    if src == 's2':
        min_x = x_vals.min()
        shifted_vals = x_vals - min_x
        plt.plot(shifted_vals, rel_elevs, marker='.', linewidth=1, linestyle='--',
                 markersize=5, label="s2 shifted", color=colors.get(src, 'gray'))

plt.xlabel("Mean Pixel Ratio")
plt.ylabel("Relative Elevation (0-1)")
plt.title("Mean Ratio per Elevation Band by Source")

# Highlight band with biggest difference
band_bottom = (max_diff_band - 1) * step
band_top = max_diff_band * step
plt.axhspan(band_bottom, band_top, color='gray', alpha=0.3)
plt.axvline(x=0.5, color='black', linestyle=':', linewidth=1)
plt.legend()
plt.xlim(0, 1.0)



# Weighted mean by glacier area per source


#calculate max_diff band for weighted mean (shifted s32)

weighted_data = {}  # {source: [mean_b1, mean_b2, ..., mean_b20]}
for src, group in scatter_df.groupby('source'):
    weighted_means = []
    for b in range(1, num_bands + 1):
        band_df = group[group['band'] == b]
        weighted_vals = []
        total_weight = 0
        for _, row in band_df.iterrows():
            rgi_id = row['rgi_id']
            weight = glacier_pixel_weights.get(rgi_id, 0)
            weighted_vals.append(row['ratio'] * weight)
            total_weight += weight
        weighted_mean = sum(weighted_vals) / total_weight if total_weight > 0 else 0
        weighted_means.append(weighted_mean)

    # Shift s2 before storing
    if src == 's2':
        min_x = min(weighted_means)
        weighted_means = [x - min_x for x in weighted_means]

    weighted_data[src] = weighted_means


# Step 2: Create DataFrame [bands x sources]
weighted_df = pd.DataFrame(weighted_data)
weighted_df.index = range(1, num_bands + 1)  # band 1 to 20

# Step 3: Calculate inter-source range per band
weighted_df['range'] = weighted_df.max(axis=1) - weighted_df.min(axis=1)

# Step 4: Identify band with maximum difference
max_diff_band_weighted = weighted_df['range'].idxmax()

# Step 5: Compute highlight boundaries
step = 1 / num_bands
band_bottom_w = (max_diff_band_weighted - 1) * step
band_top_w = max_diff_band_weighted * step



plt.subplot(1, 3, 3)
bar_height = 0.01  # spacing

offsets = {
    src: i * bar_height for i, src in enumerate(scatter_df['source'].unique())
}

ordered_sources = ['lx', 'sgi', 'sam', 's2']
offsets = {
    src: i * bar_height for i, src in enumerate(ordered_sources)
}

for src in ordered_sources:
    group = scatter_df[scatter_df['source'] == src]
    offset = offsets[src]
    if src == 's2':
        # Calculate original and shifted weighted means
        weighted_means = []
        for b in range(1, num_bands + 1):
            band_df = group[group['band'] == b]
            weighted_vals = []
            total_weight = 0
            for _, row in band_df.iterrows():
                rgi_id = row['rgi_id']
                weight = glacier_pixel_weights.get(rgi_id, 0)
                weighted_vals.append(row['ratio'] * weight)
                total_weight += weight
            weighted_mean = sum(weighted_vals) / total_weight if total_weight > 0 else 0
            weighted_means.append(weighted_mean)
        
        min_x = min(weighted_means)
        shifted = [x - min_x for x in weighted_means]
        offset = offsets['s2']

        # Plot old S2 as transparent
        for i, val in enumerate(weighted_means):
            plt.barh(y=rel_elevs[i] + offset, width=val, height=bar_height * 0.9,
                     color=colors.get('s2', 'green'), alpha=0.3, label="s2 original" if i == 0 else "")

        # Plot shifted S2 as main bar
        for i, val in enumerate(shifted):
            plt.barh(y=rel_elevs[i] + offset, width=val, height=bar_height * 0.9,
                     color=colors.get('s2', 'green'), alpha=1.0, label="s2 shifted" if i == 0 else "")

    else:
        # Normal plotting for other sources
        weighted_means = []
        year = source_years.get(src, "n/a")
        label = f"{src} ({year})"
        for b in range(1, num_bands + 1):
            band_df = group[group['band'] == b]
            weighted_vals = []
            total_weight = 0
            for _, row in band_df.iterrows():
                rgi_id = row['rgi_id']
                weight = glacier_pixel_weights.get(rgi_id, 0)
                weighted_vals.append(row['ratio'] * weight)
                total_weight += weight
            weighted_mean = sum(weighted_vals) / total_weight if total_weight > 0 else 0
            weighted_means.append(weighted_mean)

        offset = offsets[src]
        for i, val in enumerate(weighted_means):
            plt.barh(y=rel_elevs[i] + offset, width=val, height=bar_height * 0.9,
                     color=colors.get(src, 'gray'), label=label if i == 0 else "")

plt.xlabel("Weighted Mean Pixel Ratio")
plt.ylabel("Relative Elevation (0-1)")
plt.title("Weighted Mean by Glacier Area per Source")
plt.axhspan(band_bottom_w, band_top_w, color='gray', alpha=0.3)
plt.axvline(x=0.5, color='black', linestyle=':', linewidth=1)
plt.xlim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.show()


# boxplot:
def max_diff_band_per_glacier(df):
    # Step 1: mean ratio per band & source
    spread = df.groupby(['band', 'source'])['ratio'].mean().unstack()
    
    # Step 2: Shift 's2' if needed
    if 's2' in spread.columns:
        min_x = spread['s2'].min()
        spread['s2'] = spread['s2'] - min_x

    # Step 3: Range per band
    spread['range'] = spread.max(axis=1) - spread.min(axis=1)

    # Step 4: Max difference band
    return spread['range'].idxmax()


# Apply to each glacier group
glacier_max_diff_band = scatter_df.groupby('rgi_id').apply(max_diff_band_per_glacier, include_groups = False).reset_index()
glacier_max_diff_band.columns = ['rgi_id', 'max_diff_band']

# Compute statistics
mean_val = round(glacier_max_diff_band['max_diff_band'].mean(),2)


# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(glacier_max_diff_band['max_diff_band'], bins=20, color ='gray',edgecolor='black', alpha=0.7)
plt.xlabel('Max difference band per glacier')
plt.ylabel('Number of glaciers')
plt.title('Histogram of Max Difference Bands Across Glaciers')

# Plot mean and std lines
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'rounded Mean = {mean_val:.2f}')


# Add legend
plt.legend()
plt.tight_layout()
plt.show()

