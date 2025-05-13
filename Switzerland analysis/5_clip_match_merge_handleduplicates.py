# code structure and syntax was created with the aid of ChatGPT


#what this code does:
# - Renames columns across datasets for consistency
# - filter glaciers to keep only glaciers >= 2km^2
# - merge glacier geometries/attributes with same sgi_id 
# - add source to geodataframe (sgi, sam, lx, s2)
# - Identifies glaciers present in all four datasets (based on shared rgi_id) and filters each dataset to retain only these matched glaciers.
# - identify, visualize and handle remaining duplicate entries:
#       - Keep only one entry of "true" duplicates
#       - merge duplicate sgi glaciers that represent the sam "rgi-glacier" (SGI differenciates between more glaciers thus the discrepancy)
# - extract "img_year" from "img_time"
# - save combined and single geodataframes (only matching glaciers) as .gpkg





import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from geopandas import GeoDataFrame
from shapely.ops import unary_union

directory = "C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new"
#output path for split files
output_folder_combined = directory


# Define file paths
LX = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/LX-C02-TOA_RGI-v7-11.geojson")    
SAM = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/SAM_debris_with_rgiid.gpkg")   
S2 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/S2_RGI_v7_11_central_europe_debris_cover.gpkg")
SGI = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/SGI_debris_with_rgiid.gpkg")   


# Rename columns for consistency
S2 = S2.rename(columns={"fdc_ratio": "debris_cover_ratio", "year_S2": "img_time"})
LX = LX.rename(columns={"debris_ratio": "debris_cover_ratio", "Landsat_Scene_Date": "img_time"})
SGI = SGI.rename(columns={"year_acq": "img_time"})




# Print initial counts
print(f"SGI (before filtering): {len(SGI)}")
SGI = SGI.loc[SGI["area_km2_total"] >= 2].copy()
print(f"SGI (after filtering): {len(SGI)}\n")

print(f"LX (before filtering): {len(LX)}")
LX = LX.loc[LX["total_glacier_area_m2"] >= 2000000].copy()
print(f"LX (after filtering): {len(LX)}\n")

print(f"S2 (before filtering): {len(S2)}")
S2 = S2.loc[S2["area_gl_m2"] >= 2000000].copy()
print(f"S2 (after filtering): {len(S2)}\n")

print(f"SAM (total count): {len(SAM)}")

# Group by "sgi-id"
columns_to_sum = ['area_km2_total', 'area_km2_debris', 'debris_cover_ratio']


crs = SGI.crs
columns_to_sum = ['area_km2_total', 'area_km2_debris', 'debris_cover_ratio']

merged_rows = []

for sgi_id, group in SGI.groupby('sgi-id'):
    # Sum specified columns
    summed = group[columns_to_sum].sum()

    # Get the first row for non-summed attributes
    first = group.iloc[0].drop(columns=columns_to_sum + ['geometry'])

    # Merge geometries
    merged_geom = unary_union(group.geometry)

    # Combine into dictionary
    row = {
        'sgi-id': sgi_id,
        **first.to_dict(),
        **summed.to_dict(),
        'geometry': merged_geom
    }

    merged_rows.append(row)

# Convert to GeoDataFrame
SGI = gpd.GeoDataFrame(merged_rows, crs=crs)


# Create dictionary of datasets
glaciers = {'lx': LX, 'sam': SAM, 's2': S2, 'sgi': SGI}



for name, data in glaciers.items():
    original = data['img_time']

    # Try standard parsing
    data['img_time'] = pd.to_datetime(data['img_time'], errors='coerce')
    data['img_year'] = data['img_time'].dt.year

    # Fix for misinterpreted microsecond years (like 1970 + 2,000,000 μs)
    if data['img_year'].nunique() == 1 and data['img_year'].iloc[0] == 1970:
        # Try converting original values to integers and treat them as years
        try:
            data['img_year'] = pd.to_numeric(original, errors='coerce').astype('Int64')
        except Exception as e:
            print(f"Error for img_time in {name}: {e}")
    print(name)
    print(data["img_year"])




for name, data in glaciers.items():
    print(name)
    print(data.crs)

# Ensure same CRS
for name, data in glaciers.items():
    transformed = data.to_crs(epsg=2056).copy()
    glaciers[name] = transformed  # update dictionary
   

for name, data in glaciers.items():
    print(name)
    print(data.crs)


# Ensure 'rgi_id' is of the same type in all datasets
for name in glaciers:
    df = glaciers[name].copy()
    df["rgi_id"] = df["rgi_id"].astype(str)
    df["source"] = name
    glaciers[name] = df  # Save changes back


if "name" in glaciers["sgi"].columns:
    glaciers["sgi"] = glaciers["sgi"].rename(columns={"name": "name_sgi"})


# Drop rows with missing rgi_id or debris_cover_ratio in each dataset
for name in glaciers:
    glaciers[name] = glaciers[name].dropna(subset=["rgi_id", "debris_cover_ratio"])

#  Find the intersection of rgi_ids across all datasets
rgi_sets = [set(df["rgi_id"]) for df in glaciers.values()]
common_rgi_ids = set.intersection(*rgi_sets)

# Filter each dataset to only keep common rgi_ids
for name in glaciers:
    glaciers[name] = glaciers[name][glaciers[name]["rgi_id"].isin(common_rgi_ids)].copy()

#save files separataley with 4 matching glaciers only
for name, data in glaciers.items():
       
    output_path = f"{output_folder_combined}/{name}_matched_clean.gpkg"
    data.to_file(output_path, driver="GPKG")


# Combine all datasets into one
combined = pd.concat(glaciers.values(), ignore_index=True)

#print duplicate entries
#identify duplicates
duplicates = combined[combined.duplicated(subset=["rgi_id", "source"], keep=False)]
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Print duplicates if any
if not duplicates.empty:
    print("Duplicate entries in combined (same rgi_id within one source):")
    print(duplicates[["rgi_id", "source"]].value_counts().reset_index(name="count"))
    print("\nFull duplicate rows:")
    print(duplicates[["rgi_id", "source", "img_time", "name_sgi"]])
else:
    print("No duplicate rgi_id entries")


# Get unique rgi_id values that are duplicated in "sgi"
duplicate_rgi_ids = duplicates[duplicates["source"] == "sgi"]["rgi_id"].unique()



# Define colors for SGI glaciers
sgi_colors = ["blue", "yellow"]


#visualize "duplicates"

# Loop through each duplicate rgi_id and create a separate plot
for rgi_id in duplicate_rgi_ids:
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filter SGI duplicates for this specific rgi_id
    sgi_subset = combined[(combined["source"] == "sgi") & 
                                     (combined["rgi_id"] == rgi_id)]
    
    # Filter corresponding SAM glacier for overlay
    sam_subset = combined[(combined["source"] == "sam") & 
                                     (combined["rgi_id"] == rgi_id)]

    # Plot SGI duplicates with manually assigned colors
    for i, (name, glacier) in enumerate(sgi_subset.groupby("name_sgi")):
        glacier.plot(ax=ax, color=sgi_colors[i % 2], edgecolor="black", alpha=0.7, label=name)

    # Overlay the SAM glacier with a continuous **red outline**
    sam_subset.plot(ax=ax, facecolor="none", edgecolor="red", linestyle="-", linewidth=2, alpha=1)

    # Titles and labels
    plt.title(f"Duplicate Glacier: {rgi_id} (SGI) with SAM Overlay", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # Show each plot separately
    plt.show()



#Post-processing: Remove duplicates in SAM and SGI datasets

# Keep only the first entry for duplicates in the SAM dataset
df_combinations_all = combined[~((combined["source"] == "sam") & 
                                            (combined.duplicated(subset=["rgi_id", "source"], keep="first")))]

# Merge SGI duplicates

# Separate SGI duplicates
sgi_duplicates = df_combinations_all[(df_combinations_all["source"] == "sgi") & 
                                (df_combinations_all["rgi_id"].isin(duplicate_rgi_ids))]

# Function to merge duplicate SGI entries
def merge_sgi_group(group):
    if len(group) == 1:
        return group

    # Identify the largest glacier (by area)
    idx_largest = group["area_km2_total"].idxmax()
    base = group.loc[[idx_largest]].copy()

    # Sum relevant numerical columns
    base["area_km2"] = group["area_km2"].sum()
    base["area_km2_total"] = group["area_km2_total"].sum()
    base["debris_cover_ratio"] = group["debris_cover_ratio"].sum()

    # Merge geometries using unary_union
    base["geometry"] = unary_union(group.geometry)

    return base

# Ensure SGI duplicates has proper CRS
crs = sgi_duplicates.crs

# Apply the function (remove include_group for older pandas versions!)
sgi_merged = sgi_duplicates.groupby("rgi_id", group_keys=False).apply(merge_sgi_group, include_groups = False)

# After applying the groupby function, ensure sgi_merged is a GeoDataFrame
sgi_merged = gpd.GeoDataFrame(sgi_merged, geometry="geometry", crs=crs).reset_index(drop=True)

# Now you should have a valid GeoDataFrame
# Proceed with further processing

# Remove old SGI duplicates and add merged ones
df_combinations_all = df_combinations_all[~((df_combinations_all["source"] == "sgi") & 
                                            (df_combinations_all["rgi_id"].isin(duplicate_rgi_ids)))]
df_combinations_all = pd.concat([df_combinations_all, sgi_merged], ignore_index=True)

# Ensure all column names are lowercase to prevent duplication
df_combinations_all.columns = df_combinations_all.columns.str.lower()

# Combine img_year column
year_cols = [col for col in df_combinations_all.columns if col == "img_year"]
df_combinations_all['img_year'] = df_combinations_all[year_cols].bfill(axis=1).iloc[:, 0]

# Ensure it's a GeoDataFrame before saving
df_combinations_all = gpd.GeoDataFrame(df_combinations_all, geometry="geometry", crs=crs)




# Remove true duplicates (columns with same name)
df_combinations_all = df_combinations_all.loc[:, ~df_combinations_all.columns.duplicated()]



    
df_combinations_all.to_file(f"{output_folder_combined}/Switzerland_combined.gpkg", driver="GPKG")