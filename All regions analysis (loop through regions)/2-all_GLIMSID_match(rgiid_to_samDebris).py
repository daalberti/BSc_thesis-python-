import os
import geopandas as gpd

# Define regions and projections
regions = [
    #"01Alaska", 
    #"10NorthAsia", 
    #"11CentralEurope",
    #"12CaucasusMiddleEast", 
    #"14SouthAsiaWest",
    "15SouthAsiaEast"
    #"18NewZealand"
]

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

for region in regions:
    crs_aea = region_crs[region]

    # Build the full path
    out_dir = os.path.join(dir, region)

    # Define SAM file paths
    glacier_path = f"{dir}/{region}/{region}_sam_outlines_with_rgiid.gpkg"
    debris_path = f"{dir}/SAM_{region}_minGl2km2_combinedDebrisCover.geojson"
    output_path = f"{out_dir}/{region}_SAM_Debris_with_rgiid.gpkg"

    glacier = gpd.read_file(glacier_path)
    debris = gpd.read_file(debris_path)

    # Reproject
    glacier = glacier.to_crs(crs_aea)
    debris = debris.to_crs(crs_aea)

    debris_with_rgi = debris.merge(
        glacier[['GLIMSId', 'rgi_id']],
        on='GLIMSId',
        how='left'
    )

    # Check for any debris that did not find a match
    missing = debris_with_rgi['rgi_id'].isna().sum()
    print(f"{region}: Debris features without a matching rgi_id: {missing}")
    print(f"{region}: Total debris features with RGI ID: {len(debris_with_rgi)}")

    # Write out as GeoPackage
    debris_with_rgi.to_file(output_path, driver="GPKG")
    print(f"{region}: Exported to: {output_path}")
