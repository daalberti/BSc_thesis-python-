import geopandas as gpd


directory = "C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new"
# Define file paths
rgi_path = f"{directory}/RGI_7_outline_11.geojson"
sgi_path = f"{directory}/SGI_2016_glaciers.shp"
output_path = f"{directory}/SGI_glacier_outlines_with_rgiid.gpkg"

# Load SGI and RGI datasets
sgi = gpd.read_file(sgi_path)
rgi = gpd.read_file(rgi_path)

rgi = rgi.to_crs("EPSG:2056")
sgi = sgi.to_crs("EPSG:2056")
   
print(sgi.crs)
print(rgi.crs)

# Perform spatial join (SGI with RGI based on maximum overlap)
sgi["rgi_id"] = None
sgi["glims_id"] = None

for index, sgi_geom in sgi.iterrows():
    # Find intersecting glaciers
    intersects = rgi[rgi.geometry.intersects(sgi_geom.geometry)].copy()

    if not intersects.empty:
        # Compute overlap area
        intersects["overlap_area"] = intersects.geometry.intersection(sgi_geom.geometry).area

        # Select glacier with maximum overlap
        best_match = intersects.loc[intersects["overlap_area"].idxmax()]

        # Assign rgi_id and glims_id
        sgi.at[index, "rgi_id"] = best_match["rgi_id"]
        sgi.at[index, "glims_id"] = best_match["glims_id"]

# Save the joined dataset
sgi.to_file(output_path, driver="GPKG")

print(f"Spatial join completed. Output saved at: {output_path}")
