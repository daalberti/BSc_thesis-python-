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
import matplotlib as matplotlib
from rasterio.warp import calculate_default_transform, reproject, Resampling
# Set global font
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10

regions = [
    "01Alaska", 
    "10NorthAsia", 
    "11CentralEurope",
    "12CaucasusMiddleEast", 
    "14SouthAsiaWest",
    "15SouthAsiaEast", 
    "18NewZealand"
]

numbers = [1, 10, 11, 12, 14, 15, 18]
region_crs = {
    "01Alaska": "EPSG:3338",
    "10NorthAsia": "EPSG:5940",
    "11CentralEurope": "EPSG:3035",
    "12CaucasusMiddleEast": "EPSG:8857",
    "14SouthAsiaWest": "EPSG:8857",
    "15SouthAsiaEast": "EPSG:8857",
    "18NewZealand": "EPSG:2193"
}

dir = f"C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis"

for region, number in zip(regions, numbers):
    crs_aea = region_crs[region]
    out_dir = os.path.join(dir, region)
    dem_path = f"C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/DEM/global/DEM_{region}_merged_FINAL.tif"

    glacier_rgi7 = gpd.read_file(f"{dir}/RGI7_outline_{number}.geojson").to_crs(crs_aea)
    glacier_sam = gpd.read_file(f"{out_dir}/{region}_sam_outlines_with_rgiid.gpkg").to_crs(crs_aea)
    debris = gpd.read_file(f"{out_dir}/{region}_combined.gpkg").to_crs(crs_aea)
    debris = debris[debris['source'].isin(['s2','lx','sam'])].copy()


    print(glacier_rgi7.crs)
    print(glacier_sam.crs)
    print(debris.crs)
  
    src_dem = rasterio.open(dem_path)

    num_bands = 20
    all_results = []
    glacier_pixel_weights = {}
    glaciers_with_nan_bands = set()
    nan_counts_by_source_band = defaultdict(lambda: defaultdict(int))

    for (rgi_id, source), group in debris.groupby(['rgi_id','source']):
        if source in ('lx','s2'):
            glacier_for_calc = glacier_rgi7
        elif source == 'sam':
            glacier_for_calc = glacier_sam
        else:
            continue

        geom_main = [mapping(g.buffer(1)) for g in glacier_for_calc[glacier_for_calc['rgi_id']==rgi_id].geometry]
        geom_bins = [mapping(g.buffer(1)) for g in glacier_rgi7[glacier_rgi7['rgi_id']==rgi_id].geometry]

        try:
            dem_main, trf = mask(src_dem, geom_main, crop=True)
            dem_bins, _   = mask(src_dem, geom_bins, crop=True)
        except Exception as e:
            print(f"Skipping {rgi_id} ({source}) due to DEM mask error: {e}")
            continue

        dem_main = dem_main[0]
        dem_bins = dem_bins[0]
        dem_main = np.where(dem_main == src_dem.nodata, np.nan, dem_main)
        dem_bins = np.where(dem_bins == src_dem.nodata, np.nan, dem_bins)
        dem_main[dem_main==0] = np.nan
        dem_bins[dem_bins==0] = np.nan

        if np.all(np.isnan(dem_main)) or np.all(np.isnan(dem_bins)):
            print(f"Skipping {rgi_id}: empty DEM")
            continue

        mask_gl = rasterize(
            [(g,1) for g in glacier_for_calc[glacier_for_calc['rgi_id']==rgi_id].geometry],
            out_shape=dem_main.shape, transform=trf, fill=0, dtype='uint8'
        )
        glacier_pixel_weights[rgi_id] = np.sum(mask_gl==1)

        elev_min, elev_max = np.nanmin(dem_bins), np.nanmax(dem_bins)
        bin_size = (elev_max - elev_min) / num_bands
        normed = (dem_main - elev_min) / bin_size
        band_idx = np.full(normed.shape, np.nan)
        valid_mask = ~np.isnan(normed)
        band_idx[valid_mask] = np.floor(normed[valid_mask]).astype(int) + 1
        band_idx = np.clip(band_idx, 1, num_bands)

        debris_sub = group.copy()
        for b in range(1, num_bands+1):
            debris_sub[f'band_{b}'] = np.nan

        for ix, row in debris_sub.iterrows():
            geo_mask = rasterize(
                [(row.geometry,1)], out_shape=dem_main.shape,
                transform=trf, fill=0, dtype='uint8'
            )
            for b in range(1, num_bands+1):
                band_mask = (band_idx==b)
                if not np.any(band_mask):
                    nan_counts_by_source_band[source][b] += 1
                    glaciers_with_nan_bands.add(rgi_id)
                    continue
                overlap = (geo_mask==1)&band_mask
                total = np.sum(band_mask)
                count = np.sum(overlap)
                ratio = float(count)/float(total) if total > 0 else np.nan
                if total == 0:
                    nan_counts_by_source_band[source][b] += 1
                    glaciers_with_nan_bands.add(rgi_id)
                debris_sub.at[ix, f'band_{b}'] = ratio

        all_results.append(debris_sub)

    print(f"[{region}] Glaciers w/ NaN bands: {len(glaciers_with_nan_bands)}")
    nan_df = pd.DataFrame(nan_counts_by_source_band).fillna(0).astype(int)
    nan_df.index.name = 'band'
    nan_df.columns.name = 'source'
    print(nan_df)

    final_gdf = gpd.GeoDataFrame(pd.concat(all_results, ignore_index=True), geometry='geometry', crs=debris.crs)
    weights_df = pd.DataFrame(list(glacier_pixel_weights.items()), columns=["rgi_id", "pixel_weight"])
    final_gdf = final_gdf.merge(weights_df, on="rgi_id", how="left")
    final_gdf.to_file(f"{out_dir}/{region}_debris_cover_per_elevation_band.gpkg", driver="GPKG")

    step = 1/num_bands
    rel_elevs = [(b-0.5)*step for b in range(1, num_bands+1)]

    scatter_data = []
    for _, row in final_gdf.iterrows():
        for b in range(1, num_bands+1):
            r = row[f'band_{b}']
            if r >= 0:
                jitter = np.random.uniform(-0.01, 0.01)
                scatter_data.append((
                    b, r, row['rgi_id'], row['source'],
                    rel_elevs[b-1] + jitter,
                    row['pixel_weight'], row['img_year']
                ))
    scatter_df = pd.DataFrame(scatter_data, columns=['band','ratio','rgi_id','source','rel_elev','pixel_weight','img_year'])

    colors = {'s2': '#249f60', 'lx': '#e99800', 'sam': '#c4669a'}
    source_years = final_gdf.groupby('source')['img_year'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    plt.figure(figsize=(18,5))
    plt.suptitle(f"Relative Elevation versus Debris Cover Ratio ({region})", fontsize = 28)

    plt.subplot(1,3,1)
    for src, grp in scatter_df.groupby('source'):
        y = grp['rel_elev']
        plt.scatter(grp['ratio'], y, s=5, alpha=0.7,
                    label=f"{src} ({source_years[src]})",
                    color=colors[src], edgecolors='none')
    plt.xlabel("Pixel Ratio", fontsize = 18)
    plt.ylabel("Relative Elevation", fontsize = 18)
    plt.title("Ratio per Elevation Band", fontsize = 21)
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(loc='upper right',fontsize = 18)

    band_means_df = scatter_df.groupby(['band', 'source'])['ratio'].mean().unstack()
    if 's2' in band_means_df.columns:
        min_x = band_means_df['s2'].min()
        band_means_df['s2'] = band_means_df['s2'] - min_x
    s2_shift = min_x
    band_means_df['range'] = band_means_df.max(axis=1) - band_means_df.min(axis=1)
    max_diff_band = band_means_df['range'].idxmax()

    plt.subplot(1,3,2)
    for src, grp in scatter_df.groupby('source'):
        means = grp.groupby('band')['ratio'].mean()
        vals = means.values
        plt.plot(vals, rel_elevs, marker='.', linewidth=1, markersize=5,
                 label=f"{src} ({source_years[src]})",
                 color=colors[src])
        if src=='s2':
            shifted = vals - vals.min()
            plt.plot(shifted, rel_elevs, linestyle='--',
                     label="s2 shifted", marker='.', markersize=5,
                     color=colors[src])
    plt.xlabel("Mean Ratio per Elevation Band\nper Dataset", fontsize = 18)
    plt.title("Mean Pixel Ratio", fontsize = 21)
    plt.legend(loc='upper right',fontsize = 18)
    plt.axvline(0.5, linestyle=':', color='black', linewidth=1)
    plt.axhspan((max_diff_band-1)*step, max_diff_band*step, color='gray', alpha=0.3)
    plt.tick_params(axis='both', labelsize=15)
    plt.xlim(0, 1.0)

    weighted_data = {}
    for src, group in scatter_df.groupby('source'):
        weighted_means = []
        for b in range(1, num_bands + 1):
            band_df = group[group['band'] == b]
            total_weight = sum(glacier_pixel_weights.get(r, 0) for r in band_df['rgi_id'])
            if total_weight > 0:
                weighted_vals = sum(band_df['ratio'] * band_df['rgi_id'].map(glacier_pixel_weights))
                weighted_mean = weighted_vals / total_weight
            else:
                weighted_mean = 0
            weighted_means.append(weighted_mean)
        if src == 's2':
            min_x = min(weighted_means)
            weighted_means = [x - min_x for x in weighted_means]
        weighted_data[src] = weighted_means

    weighted_df = pd.DataFrame(weighted_data)
    weighted_df.index = range(1, num_bands + 1)
    weighted_df['range'] = weighted_df.max(axis=1) - weighted_df.min(axis=1)
    max_diff_band_weighted = weighted_df['range'].idxmax()

    plt.subplot(1,3,3)
    for src, grp in scatter_df.groupby('source'):
        weighted = []
        for b in range(1, num_bands+1):
            sub = grp[grp['band']==b]
            total_w = sum(glacier_pixel_weights.get(r,0) for r in sub['rgi_id'])
            if total_w>0:
                weighted.append(sum(sub['ratio']*sub['rgi_id'].map(glacier_pixel_weights)) / total_w)
            else:
                weighted.append(0)
        plt.plot(weighted, rel_elevs, marker='.', linewidth=1, markersize=5,
                 label=f"{src} ({source_years[src]})",
                 color=colors[src])
        if src=='s2':
            shifted = np.array(weighted) - min(weighted)
            plt.plot(shifted, rel_elevs, linestyle='--',
                     label="s2 shifted", marker='.', markersize=5,
                     color=colors[src])
    plt.xlabel("Weighted Mean by Pixel Area\nper Dataset", fontsize = 18)
    plt.title("Weighted Mean pixel Ratio", fontsize = 21)
    plt.legend(loc='upper right',fontsize = 18)
    plt.axvline(0.5, linestyle=':', color='black', linewidth=1)
    plt.axhspan((max_diff_band_weighted - 1) * step, max_diff_band_weighted * step, color='gray', alpha=0.3)
    plt.tight_layout()
    plt.xlim(0, 1.0)
    plt.tick_params(axis='both', labelsize=15)
    plt.savefig(f"{out_dir}/{region}_elevation_band_plot.png")
    plt.close()

    def max_diff_band_per_glacier(df):
        df = df.copy()
        if 's2' in df['source'].unique():
            df.loc[df['source'] == 's2', 'ratio'] -= s2_shift
            df.loc[df['source'] == 's2', 'ratio'] = df.loc[df['source'] == 's2', 'ratio'].clip(lower=0)
        spread = df.groupby(['band', 'source'])['ratio'].mean().unstack()
        spread['range'] = spread.max(axis=1) - spread.min(axis=1)
        return spread['range'].idxmax()

    glacier_max_diff_band = scatter_df.groupby('rgi_id').apply(max_diff_band_per_glacier, include_groups=False).reset_index()
    glacier_max_diff_band.columns = ['rgi_id', 'max_diff_band']
    scatter_df = scatter_df.merge(glacier_max_diff_band, on='rgi_id', how='left')
    os.makedirs(f"{dir}/final_gdfs", exist_ok=True)
    scatter_df.to_csv(f"{dir}/final_gdfs/{region}.csv", index=False)

    mean_val = round(glacier_max_diff_band['max_diff_band'].mean(), 2)
    plt.figure(figsize=(8, 5))
    plt.hist(glacier_max_diff_band['max_diff_band'], bins=20, color='gray', edgecolor='black', alpha=0.7)
    plt.xlabel('Max difference band per glacier', fontsize = 16.5)
    plt.ylabel('Number of glaciers', fontsize = 16.5)
    plt.title(f'Histogram of Max Difference Bands – {region}', fontsize = 19)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'rounded Mean = {mean_val:.2f}')
    plt.legend(fontsize = 18)
    plt.tight_layout()
    plt.tick_params(axis='both', labelsize=14)
    plt.savefig(f"{out_dir}/{region}_max_diff_band_histogram.png")
    plt.close()


# After the loop over regions
combined_rgi7_list = []

# Rebuild the full list of regions and numbers (in case some were commented out)
region_number_map = {
    "01Alaska": 1,
    "10NorthAsia": 10,
    "11CentralEurope": 11,
    "12CaucasusMiddleEast": 12,
    "14SouthAsiaWest": 14,
    "15SouthAsiaEast": 15,
    "18NewZealand": 18
}

for region in regions:
    number = region_number_map[region]
    crs = region_crs[region]
    path = os.path.join(dir, f"RGI7_outline_{number}.geojson")
    
    if os.path.exists(path):
        gdf = gpd.read_file(path).to_crs(crs)
        gdf["region_code"] = number
        combined_rgi7_list.append(gdf)
    else:
        print(f"[Warning] RGI7 file not found for {region}")

# Combine and save RGI7 outlines and debris cover per elevation band
combined_rgi7_list = []
all_debris_gdfs = []

all_together_dir = os.path.join(dir, "all_together")
os.makedirs(all_together_dir, exist_ok=True)

for region in regions:
    number = region_number_map[region]

    # --- RGI7 outlines ---
    rgi_path = os.path.join(dir, f"RGI7_outline_{number}.geojson")
    if os.path.exists(rgi_path):
        gdf_rgi = gpd.read_file(rgi_path).to_crs(epsg=8857)
        gdf_rgi["region_code"] = number
        combined_rgi7_list.append(gdf_rgi)
    else:
        print(f"[Warning] RGI7 file not found for {region}")

    # --- Debris cover GeoPackage ---
    debris_path = os.path.join(dir, region, f"{region}_debris_cover_per_elevation_band.gpkg")
    if os.path.exists(debris_path):
        gdf_debris = gpd.read_file(debris_path).to_crs(epsg=8857)
        gdf_debris["region"] = region
        all_debris_gdfs.append(gdf_debris)
    else:
        print(f"[Warning] Debris cover file not found for {region}")

# Save combined RGI7 outlines
if combined_rgi7_list:
    combined_rgi7 = gpd.GeoDataFrame(pd.concat(combined_rgi7_list, ignore_index=True), crs=combined_rgi7_list[0].crs)
    rgi7_output_path = os.path.join(all_together_dir, "RGI7_all_combined.gpkg")
    combined_rgi7.to_file(rgi7_output_path, driver="GPKG", layer="rgi7_combined")
    print("[Done] Combined RGI7 outlines saved.")
else:
    print("[Error] No RGI7 files found to combine.")

# Save combined debris cover per elevation band
if all_debris_gdfs:
    combined_debris_gdf = gpd.GeoDataFrame(pd.concat(all_debris_gdfs, ignore_index=True), crs=all_debris_gdfs[0].crs)
    debris_output_path = os.path.join(all_together_dir, "all_FDC_per_elevation_band.gpkg")
    combined_debris_gdf.to_file(debris_output_path, driver="GPKG")
    print("[Done] Combined debris cover saved to all_FDC_per_elevation_band.gpkg.")
else:
    print("[Error] No debris cover files found to combine.")
