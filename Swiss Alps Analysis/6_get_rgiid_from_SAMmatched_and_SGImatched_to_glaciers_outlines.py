import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.plot import plotting_extent

# Load the data
match_sgi = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/sgi_matched_clean.gpkg")
match_sam = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/sam_matched_clean.gpkg")
glacier_sgi = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/SGI_2016_glaciers.shp")
glacier_sam = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/11CentralEurope_minGl2km2.shp")

print(len(match_sgi))
print(len(match_sam))

# Reproject to the same CRS as match_sam and match_sgi
glacier_sam = glacier_sam.to_crs(match_sam.crs)
glacier_sgi = glacier_sgi.to_crs(match_sgi.crs)

# Rename columns to match naming conventions
glacier_sam = glacier_sam.rename(columns={"RGIId": "rgiid"})
glacier_sgi = glacier_sgi.rename(columns={"sgi-id": "sgi_id"})
match_sam = match_sam.rename(columns={"RGIId": "rgiid"})
match_sgi = match_sgi.rename(columns={"sgi-id": "sgi_id"})

# Handle duplicates in glacier_sgi and glacier_sam
duplicates_sgi_glacier = glacier_sgi[glacier_sgi.duplicated(subset='sgi_id', keep=False)]
duplicates_rgi_glacier = glacier_sam[glacier_sam.duplicated(subset='rgiid', keep=False)]

# Print duplicate entries for debugging
print("Duplicates in glacier_sgi (sgi_id):")
print(duplicates_sgi_glacier[['name', 'sgi_id', 'area_km2']])

print("Duplicates in glacier_sam (rgiID):")
print(duplicates_rgi_glacier[['rgiid']])

# Filter glacier_sam where deb_km2 != 0 and remove duplicates based on 'rgiid'
filtered_glacier_sam = glacier_sam[glacier_sam['deb_km2'] != 0]
glacier_sam = filtered_glacier_sam.drop_duplicates(subset='rgiid', keep='first')

# Check duplicates after filtering
duplicates_rgi_glacier = glacier_sam[glacier_sam.duplicated(subset='rgiid', keep=False)]
print("Duplicates in glacier_sam (rgiid) AFTER elimination:")
print(duplicates_rgi_glacier[['rgiid']])



# Remove duplicates in match_sam
match_sam = match_sam.drop_duplicates(subset='rgiid', keep='first')

# Check for duplicates in match_sgi
duplicates_sgi_match = match_sgi[match_sgi.duplicated(subset='sgi_id', keep=False)]

# Print duplicates in match_sgi
print("Duplicates in match_sgi (sgi_id):")
print(duplicates_sgi_match[['sgi_id', 'name_sgi']])

# Check uniqueness of IDs
print("Unique rgiid in match_sam:", match_sam['rgiid'].nunique())
print("Total rgiid in match_sam:", len(match_sam))
print("Unique sgi_id in match_sgi:", match_sgi['sgi_id'].nunique())
print("Total sgi_id in match_sgi:", len(match_sgi))

# Extract relevant glaciers from glacier_sam based on rgiID from match_sam
relevant_sam = glacier_sam[glacier_sam['rgiid'].isin(match_sam['rgiid'])]
relevant_sam = relevant_sam.merge(match_sam[['rgiid', 'rgi_id']], on='rgiid', how='left')

# Check for duplicates in the 'rgi_id' column for relevant_sam
duplicates_sam = relevant_sam[relevant_sam.duplicated(subset='rgi_id', keep=False)]

# Print duplicates in relevant_sam
print("Duplicates in relevant_sam (AFTER merging):")
print(duplicates_sam[['rgi_id']])

# List of glaciers to pair for merging (larger and smaller)
pairs_to_merge = [
    ("Furgggletscher", "Oberer Theodulgletscher"),
    ("Unterer Theodulgletscher", "Gornergletscher"),
    ("Unterer Grindelwaldgletscher", "Obers Ischmeer E")
]

# Merge glaciers (Larger + Smaller)
for larger, smaller in pairs_to_merge:
    # Filter glaciers based on their name
    larger_glacier = glacier_sgi[glacier_sgi['name'] == larger]
    smaller_glacier = glacier_sgi[glacier_sgi['name'] == smaller]
    
    # Ensure there's only one row per glacier (we expect that, but just in case)
    larger_glacier = larger_glacier.iloc[0]
    smaller_glacier = smaller_glacier.iloc[0]
    
    # Sum the area_km2
    new_area_km2 = larger_glacier['area_km2'] + smaller_glacier['area_km2']
    
    # Create the merged glacier
    merged = larger_glacier.copy()
    merged['area_km2'] = new_area_km2
    
    # Combine the geometries (union the geometries)
    merged_geometry = larger_glacier['geometry'].union(smaller_glacier['geometry'])
    merged['geometry'] = merged_geometry
    
    # Drop the old glaciers (both larger and smaller) by their 'name'
    glacier_sgi = glacier_sgi[~glacier_sgi['name'].isin([larger, smaller])]
    
    # Add the merged glacier back to the GeoDataFrame
    merged_geo_df = gpd.GeoDataFrame([merged], geometry='geometry', crs=glacier_sgi.crs)
    glacier_sgi = pd.concat([glacier_sgi, merged_geo_df], ignore_index=True)

# Extract relevant glaciers from glacier_sgi based on sgi_id from match_sgi
relevant_sgi = glacier_sgi[glacier_sgi['sgi_id'].isin(match_sgi['sgi_id'])]
relevant_sgi = relevant_sgi.merge(match_sgi[['sgi_id', 'rgi_id']], on='sgi_id', how='left')

# Check for duplicates in the 'rgi_id' column for relevant_sgi
duplicates_sgi = relevant_sgi[relevant_sgi.duplicated(subset='rgi_id', keep=False)]

# Print duplicates in relevant_sgi
print("Duplicates in relevant_sgi (AFTER merging):")
print(duplicates_sgi[['rgi_id', 'name']])

# Check uniqueness of IDs
print("Unique rgiid in match_sam:", relevant_sam['rgi_id'].nunique())
print("Total rgiid in match_sam:", len(relevant_sam))
print("Unique sgi_id in match_sgi:", relevant_sgi['rgi_id'].nunique())
print("Total sgi_id in match_sgi:", len(relevant_sgi))



# Save relevant data to GeoPackages
relevant_sam.to_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/SAM_matching_glacier_outlines.gpkg", driver="GPKG")
relevant_sgi.to_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/SGI_matching_glacier_outlines.gpkg", driver="GPKG")
