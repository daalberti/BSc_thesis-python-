import geopandas as gpd


directory = "C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new"
# Define file paths
rgi_path = f"{directory}/RGI_7_outline_11.geojson"
sam_path = f"{directory}/11CentralEurope_minGl2km2.shp"
output_path = f"{directory}/SAM_glacier_outlines_with_rgiid.gpkg"

# Load SGI and RGI datasets
sam = gpd.read_file(sam_path)
rgi = gpd.read_file(rgi_path)

# Ensure both datasets have the same CRS

sam = sam.to_crs("EPSG:2056")
rgi = rgi.to_crs("EPSG:2056")

# Perform spatial join (SGI with RGI based on maximum overlap)
sam["rgi_id"] = None
sam["glims_id"] = None

for index, sgi_geom in sam.iterrows():
    # Find intersecting glaciers
    intersects = rgi[rgi.geometry.intersects(sgi_geom.geometry)].copy()

    if not intersects.empty:
        # Compute overlap area
        intersects["overlap_area"] = intersects.geometry.intersection(sgi_geom.geometry).area

        # Select glacier with maximum overlap
        best_match = intersects.loc[intersects["overlap_area"].idxmax()]

        # Assign rgi_id and glims_id
        sam.at[index, "rgi_id"] = best_match["rgi_id"]
        sam.at[index, "glims_id"] = best_match["glims_id"]

# Save the joined dataset
sam.to_file(output_path, driver="GPKG")

print(f"Spatial join completed. Output saved at: {output_path}")
