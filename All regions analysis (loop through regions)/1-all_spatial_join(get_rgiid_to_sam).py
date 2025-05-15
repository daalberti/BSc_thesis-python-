import geopandas as gpd
import os

# Get RGI id to SAM
regions = [
    #"01Alaska", 
    #"10NorthAsia", 
    #"11CentralEurope",
    #"12CaucasusMiddleEast", 
    #"14SouthAsiaWest",
    "15SouthAsiaEast", 
    #"18NewZealand"
]

numbers = [1,10,11,12,14,15,18]

# Region-specific CRS
region_crs = {
    "01Alaska": "EPSG:3338",
    "10NorthAsia": "EPSG:5940",
    "11CentralEurope": "EPSG:3035",
    "12CaucasusMiddleEast": "EPSG:8857",
    "14SouthAsiaWest": "EPSG:8857",
    "15SouthAsiaEast": "EPSG:8857",
    "18NewZealand": "EPSG:2193"
}

base_dir = "C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis"

for region, number in zip(regions, numbers):
    crs_aea = region_crs[region]

    out_dir = os.path.join(base_dir, region)
    os.makedirs(out_dir, exist_ok=True)

    # Define file paths
    rgi_path = f"{base_dir}/RGI7_outline_{number}.geojson"
    sam_path = f"{base_dir}/../The state of rock debris covering earth's glaciers/S1/{region}/{region}_minGl2km2.shp"
    output_path = f"{out_dir}/{region}_sam_outlines_with_rgiid.gpkg"

    # Load SGI and RGI datasets
    rgi = gpd.read_file(rgi_path)
    sam = gpd.read_file(sam_path)

    # Reproject
    rgi = rgi.to_crs(crs_aea)
    sam = sam.to_crs(crs_aea)

    print(f"{region}: RGI count =", len(rgi))
    print(f"{region}: SAM count =", len(sam))

    sam = sam.dissolve(by='GLIMSId').reset_index()
    rgi = rgi.dissolve(by='rgi_id').reset_index()

    # Perform spatial join
    sam["rgi_id"] = None
    sam["glims_id"] = None

    # Clean geometries
    sam['geometry'] = sam.geometry.buffer(0)
    rgi['geometry'] = rgi.geometry.buffer(0)

    for index, sam_geom in sam.iterrows():
        intersects = rgi[rgi.geometry.intersects(sam_geom.geometry)].copy()

        if not intersects.empty:
            intersects["overlap_area"] = intersects.geometry.intersection(sam_geom.geometry).area
            best_match = intersects.loc[intersects["overlap_area"].idxmax()]
            sam.at[index, "rgi_id"] = best_match["rgi_id"]
            sam.at[index, "glims_id"] = best_match["glims_id"]

    print(f"{region}: SAM after join =", len(sam))

    # Save the joined dataset
    sam.to_file(output_path, driver="GPKG")
    print(f"{region}: Spatial join completed. Output saved at: {output_path}")
