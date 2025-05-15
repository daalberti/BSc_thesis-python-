import geopandas as gpd

directory = "C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new"
# Define file paths
debris_path = f"{directory}/SAM_11CentralEurope_minGl2km2_combinedDebrisCover.geojson"
glacier_path = f"{directory}/SAM_glacier_outlines_with_rgiid.gpkg"
output_path = f"{directory}/SAM_debris_with_rgiid.gpkg"


glacier = gpd.read_file(glacier_path)
debris  = gpd.read_file(debris_path)

glacier = glacier.to_crs("EPSG:2056")
debris = debris.to_crs("EPSG:2056")

debris_with_rgi = debris.merge(
    glacier[['GLIMSId', 'rgi_id']],
    on='GLIMSId',
    how='left'
)



# check for any debris that did not find a match
missing = debris_with_rgi['rgi_id'].isna().sum()
print(f"Debris features without a matching rgi_id: {missing}")

#Write out as GeoPackage
debris_with_rgi.to_file(output_path, driver="GPKG")