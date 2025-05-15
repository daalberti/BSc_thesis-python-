import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely.ops import unary_union

regions = [
    "01Alaska", 
    "10NorthAsia", 
    "11CentralEurope",
    "12CaucasusMiddleEast", 
    "14SouthAsiaWest",
    "15SouthAsiaEast", 
    "18NewZealand"
]

numbers = [1,10,11,12,14,15,18]

region_crs = {
    "01Alaska": "EPSG:3338",
    "10NorthAsia": "EPSG:5940",
    "11CentralEurope": "EPSG:3035",
    "12CaucasusMiddleEast": "EPSG:8857",
    "14SouthAsiaWest": "EPSG:8857",
    "15SouthAsiaEast": "EPSG:8857",
    "18NewZealand": "EPSG:2193"
}

dir = "C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis"

for region, number in zip(regions, numbers):
    crs_aea = region_crs[region]
    
    # Split region into number and name
    region_num = ''.join(filter(str.isdigit, region))
    region_name = region[len(region_num):].lower()
    region_formatted = f"{region_num}_{region_name}"

    out_dir = os.path.join(dir, region)

    if region == '01Alaska':
        s2_path = f"{dir}/S2_RGI_v7_01_alaska_debris_cover.gpkg"
        lx_path = f"{dir}/LX-C02-TOA_RGI-v7-01.geojson"
    elif region == '10NorthAsia':
        s2_path = f"{dir}/S2_RGI_v7_10_north_asia_debris_cover.gpkg"
        lx_path = f"{dir}/LX-C02-TOA_RGI-v7-{number}.geojson"
    elif region == '11CentralEurope':
        s2_path = f"{dir}/S2_RGI_v7_11_central_europe_debris_cover.gpkg"
        lx_path = f"{dir}/LX-C02-TOA_RGI-v7-{number}.geojson"
    elif region == '12CaucasusMiddleEast':
        s2_path = f"{dir}/S2_RGI_v7_12_caucasus_middle_east_debris_cover.gpkg"
        lx_path = f"{dir}/LX-C02-TOA_RGI-v7-{number}.geojson"
    elif region == '14SouthAsiaWest':
        s2_path = f"{dir}/S2_RGI_v7_14_south_asia_west_debris_cover.gpkg"
        lx_path = f"{dir}/LX-C02-TOA_RGI-v7-{number}.geojson"
    elif region == '15SouthAsiaEast':
        s2_path = f"{dir}/S2_RGI_v7_15_south_asia_east_debris_cover.gpkg"
        lx_path = f"{dir}/LX-C02-TOA_RGI-v7-{number}.geojson"
    elif region == '18NewZealand':
        s2_path = f"{dir}/S2_RGI_v7_18_new_zealand_debris_cover.gpkg"
        lx_path = f"{dir}/LX-C02-TOA_RGI-v7-{number}.geojson"

    
    
    sam_path = f"{out_dir}/{region}_SAM_Debris_with_rgiid.gpkg"
    
    
    output_dir = out_dir

    rgi_glaciers = gpd.read_file(f"{dir}/RGI7_outline_{number}.geojson")
    sam = gpd.read_file(sam_path)
    s2 = gpd.read_file(s2_path)
    lx = gpd.read_file(lx_path)

    sam = sam.to_crs(crs_aea)
    s2 = s2.to_crs(crs_aea)
    lx = lx.to_crs(crs_aea)
    rgi_glaciers = rgi_glaciers.to_crs(crs_aea)

    print(sam.crs)
    print(s2.crs)
    print(lx.crs)
    print(rgi_glaciers.crs)

    lx = lx.rename(columns={"debris_ratio": "debris_cover_ratio", "Landsat_Scene_Date": "img_time"})
    s2 = s2.rename(columns={"fdc_ratio": "debris_cover_ratio", "year_S2": "img_time"})
    sam = sam.rename(columns={"year_acq": "img_time", "gl_km2": "area_km2"})

    lx = lx[lx["area_km2"] >= 2].copy()
    s2 = s2[s2["area_km2"] >= 2].copy()

    ids_sam = set(sam['rgi_id'].dropna().astype(str))
    ids_s2 = set(s2['rgi_id'].dropna().astype(str))
    ids_lx = set(lx['rgi_id'].dropna().astype(str))
    common_ids = ids_sam & ids_s2 & ids_lx
    print(f"[{region}] Number of RGI IDs in all three datasets: {len(common_ids)}")

    datasets = {'sam': sam, 's2': s2, 'lx': lx}
    for name, df in datasets.items():
        original = df['img_time']
        df['img_time'] = pd.to_datetime(df['img_time'], errors='coerce')
        df['img_year'] = df['img_time'].dt.year
        if df['img_year'].nunique() == 1 and df['img_year'].iloc[0] == 1970:
            try:
                df['img_year'] = pd.to_numeric(original, errors='coerce').astype('Int64')
            except Exception as e:
                print(f"Error for img_time in {name}: {e}")
        df['img_year'] = df['img_year'].fillna(0).astype('Int64')
        print(name, df["img_year"].unique())

    sam = datasets['sam']
    s2 = datasets['s2']
    lx = datasets['lx']

    ids_sam = set(sam['rgi_id'].dropna().astype(str))
    ids_s2 = set(s2['rgi_id'].dropna().astype(str))
    ids_lx = set(lx['rgi_id'].dropna().astype(str))
    common_ids = ids_sam & ids_s2 & ids_lx

    sam = sam[sam['rgi_id'].isin(common_ids)].copy()
    s2 = s2[s2['rgi_id'].isin(common_ids)].copy()
    lx = lx[lx['rgi_id'].isin(common_ids)].copy()

    def handle_duplicates(gdf, rgi_glaciers, id_col='rgi_id', area_col='area_km2', rgi_overlap_thresh=0.90):
        stats = {'kept': {'single_entry': 0, 'duplicate_overlap': 0, 'duplicate_inside_rgi': 0},
                 'omitted': {'from_overlap': 0, 'from_low_rgi_overlap': 0, 'from_missing_rgi': 0}}
        kept = []
        for rgi_id, group in gdf.groupby(id_col):
            n_entries = len(group)
            if n_entries == 1:
                kept.append(group.iloc[0])
                stats['kept']['single_entry'] += 1
                continue
            sorted_grp = group.sort_values(by=area_col, ascending=False)
            largest = sorted_grp.iloc[0]
            geoms = group.geometry.values
            self_overlap = any(geoms[i].intersects(geoms[j]) for i in range(len(geoms)) for j in range(i + 1, len(geoms)))
            if self_overlap:
                kept.append(largest)
                stats['kept']['duplicate_overlap'] += 1
                stats['omitted']['from_overlap'] += n_entries - 1
                continue
            ref = rgi_glaciers[rgi_glaciers[id_col] == rgi_id]
            if ref.empty:
                stats['omitted']['from_missing_rgi'] += n_entries
                continue
            ref_geom = ref.geometry.union_all()
            ratios = [row.geometry.intersection(ref_geom).area / (row.geometry.area or 1.0) for _, row in group.iterrows()]
            if all(r >= rgi_overlap_thresh for r in ratios):
                merged_geom = unary_union(group.geometry)
                sums = {col: group[col].sum() for col in ['Area', 'glacier_area_km2', 'debris_cover_ratio', 'debris_cover_area_km2'] if col in group.columns}
                new_rec = largest.copy()
                new_rec.geometry = merged_geom
                for col, val in sums.items():
                    new_rec[col] = val
                kept.append(new_rec)
                stats['kept']['duplicate_inside_rgi'] += 1
                stats['omitted']['from_overlap'] += n_entries - 1
            else:
                stats['omitted']['from_low_rgi_overlap'] += n_entries
        return gpd.GeoDataFrame(kept, columns=gdf.columns, crs=gdf.crs), stats

    sam_clean, sam_counts = handle_duplicates(sam, rgi_glaciers)
    s2_clean, s2_counts = handle_duplicates(s2, rgi_glaciers)
    lx_clean, lx_counts = handle_duplicates(lx, rgi_glaciers)

    print(f"[{region}] SAM before handling: {len(sam)}, after: {len(sam_clean)}")
    print(f"[{region}] LX before handling: {len(lx)}, after: {len(lx_clean)}")
    print(f"[{region}] S2 before handling: {len(s2)}, after: {len(s2_clean)}")

    sam, s2, lx = sam_clean, s2_clean, lx_clean

    ids_sam = set(sam['rgi_id'].dropna().astype(str))
    ids_s2 = set(s2['rgi_id'].dropna().astype(str))
    ids_lx = set(lx['rgi_id'].dropna().astype(str))
    common_ids = ids_sam & ids_s2 & ids_lx

    sam = sam[sam['rgi_id'].isin(common_ids)].reset_index(drop=True)
    s2 = s2[s2['rgi_id'].isin(common_ids)].reset_index(drop=True)
    lx = lx[lx['rgi_id'].isin(common_ids)].reset_index(drop=True)

    datasets = [(sam, 'sam'), (s2, 's2'), (lx, 'lx')]
    for df, name in datasets:
        dup_mask = df['rgi_id'].duplicated(keep=False)
        dupes = df.loc[dup_mask, 'rgi_id']
        if dupes.empty:
            print(f"[{region}] No duplicates in {name}")
        else:
            counts = dupes.value_counts()
            print(f"[{region}] Dataset '{name}' has {len(counts)} duplicated RGI IDs:")
            print(counts.to_string(), '\n')

    sam.to_file(f"{output_dir}/{region}_SAM_matched_clean.gpkg", driver="GPKG")
    s2.to_file(f"{output_dir}/{region}_S2_matched_clean.gpkg", driver="GPKG")
    lx.to_file(f"{output_dir}/{region}_LX_matched_clean.gpkg", driver="GPKG")

    def relevant_attributes(df, source_name):
        df = df[['geometry', 'area_km2', 'debris_cover_ratio', 'img_time', 'img_year', 'rgi_id']].copy()
        df['source'] = source_name
        return df

    s2_rel = relevant_attributes(s2, 's2')
    lx_rel = relevant_attributes(lx, 'lx')
    sam_rel = relevant_attributes(sam, 'sam')

    combined = gpd.GeoDataFrame(pd.concat([s2_rel, lx_rel, sam_rel], ignore_index=True), crs=s2_rel.crs)
    combined.to_file(f"{out_dir}/{region}_combined.gpkg", driver="GPKG", layer="combined")
    print(f"[{region}] Successfully saved single and combined datasets.")

    # Create output directory for combined file
all_together_dir = os.path.join(dir, "all_together")
os.makedirs(all_together_dir, exist_ok=True)

# After the loop over regions
all_combined = []

for region in regions:
    region_num = ''.join(filter(str.isdigit, region))
    region_name = region[len(region_num):].lower()
    region_formatted = f"{region_num}_{region_name}"

    combined_path = os.path.join(dir, region, f"{region}_combined.gpkg")
    if os.path.exists(combined_path):
        gdf = gpd.read_file(combined_path)
        gdf["region"] = region  # or region_formatted if preferred
        all_combined.append(gdf)
    else:
        print(f"[Warning] Combined file not found for region: {region}")

# Concatenate all and save
if all_combined:
    all_combined = [gdf.to_crs(epsg=8857) for gdf in all_combined]
    full_combined = gpd.GeoDataFrame(pd.concat(all_combined, ignore_index=True), crs=all_combined[0].crs)
    output_path = os.path.join(all_together_dir, "all_regions_combined.gpkg")
    full_combined.to_file(output_path, driver="GPKG", layer="combined_all")
    print(f"[Done] All regions combined and saved to: {output_path}")
else:
    print("[Error] No combined files found.")
   