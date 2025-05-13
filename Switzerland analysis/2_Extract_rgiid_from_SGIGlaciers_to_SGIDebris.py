# code structure and syntax was created with the aid of ChatGPT


# what this code does:
# - Adds RGI 7.0 ID (previously added to the SGI outlines) to the SGI debris outline-dataset.


import geopandas as gpd

directory = "C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new"
# Define file paths
debris_path = f"{directory}/SGI_2016_debriscover.shp"
glacier_path = f"{directory}/SGI_glacier_outlines_with_rgiid.gpkg"
output_path = f"{directory}/SGI_debris_with_rgiid.gpkg"



glacier = gpd.read_file(glacier_path)
debris  = gpd.read_file(debris_path)
print(glacier.crs)
print(debris.crs)


glacier = glacier.rename(columns={"area_km2": "area_km2_total"})
debris = debris.rename(columns={"area_km2": "area_km2_debris"})


debris_with_rgi = debris.merge(
    glacier[['sgi-id', 'rgi_id', 'area_km2_total', 'slope_deg']],
    on='sgi-id',
    how='left'
)

debris_with_rgi['debris_cover_ratio'] = debris_with_rgi['area_km2_debris'] / debris_with_rgi['area_km2_total']

# check for any debris that did not find a match
missing = debris_with_rgi['rgi_id'].isna().sum()
print(f"Debris features without a matching rgi_id: {missing}")

#Write out as GeoPackage
debris_with_rgi.to_file(output_path, driver="GPKG")