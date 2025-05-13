# checks the number of entries of the various datasets before and after matching


from pathlib import Path
import geopandas as gpd
import pandas as pd



r1 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/01Alaska/01Alaska_debris_cover_per_elevation_band.gpkg")
r10 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/10NorthAsia/10NorthAsia_debris_cover_per_elevation_band.gpkg")
r11 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/11CentralEurope/11CentralEurope_debris_cover_per_elevation_band.gpkg")
r12 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/12CaucasusMiddleEast/12CaucasusMiddleEast_debris_cover_per_elevation_band.gpkg")
                   
r14 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/14SouthAsiaWest/14SouthAsiaWest_debris_cover_per_elevation_band.gpkg")
r15 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/15SouthAsiaEast/15SouthAsiaEast_debris_cover_per_elevation_band.gpkg")
                   
r18 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/18NewZealand/18NewZealand_debris_cover_per_elevation_band.gpkg")
gdf = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/all_together/all_regions_combined.gpkg")

# Group by 'region' and 'source', then count the number of entries in each group
counts = gdf.groupby(['region', 'source']).size().reset_index(name='count')

print(len(gdf)/3)

# Print the result
print(counts)

print("Matching Count")
print(len(r1)/3)
print(len(r10)/3)
print(len(r11)/3)
print(len(r12)/3)
print(len(r14)/3)
print(len(r15)/3)
print(len(r18)/3)



r1 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/SAM_01Alaska_minGl2km2_combinedDebrisCover.geojson")
r10 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/SAM_10NorthAsia_minGl2km2_combinedDebrisCover.geojson")
r11 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/SAM_11CentralEurope_minGl2km2_combinedDebrisCover.geojson")
r12 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/SAM_12CaucasusMiddleEast_minGl2km2_combinedDebrisCover.geojson")
                   
r14 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/SAM_14SouthAsiaWest_minGl2km2_combinedDebrisCover.geojson")
r15 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/SAM_15SouthAsiaEast_minGl2km2_combinedDebrisCover.geojson")
                   
r18 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/SAM_18NewZealand_minGl2km2_combinedDebrisCover.geojson")

print("SAM")
print(len(r1))
print(len(r10))
print(len(r11))
print(len(r12))
print(len(r14))
print(len(r15))
print(len(r18))



r1 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/S2_RGI_v7_01_alaska_debris_cover.gpkg")
r10 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/S2_RGI_v7_10_north_asia_debris_cover.gpkg")
r11 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/S2_RGI_v7_11_central_europe_debris_cover.gpkg")
r12 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/S2_RGI_v7_12_caucasus_middle_east_debris_cover.gpkg")
                   
r14 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/S2_RGI_v7_14_south_asia_west_debris_cover.gpkg")
r15 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/S2_RGI_v7_15_south_asia_east_debris_cover.gpkg")
                   
r18 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/S2_RGI_v7_18_new_zealand_debris_cover.gpkg")



r1 = r1.loc[r1["area_km2"] >= 2].copy()
r10 = r10.loc[r10["area_km2"] >= 2].copy()
r11 = r11.loc[r11["area_km2"] >= 2].copy()
r12 = r12.loc[r12["area_km2"] >= 2].copy()
r14 = r14.loc[r14["area_km2"] >= 2].copy()
r15 = r15.loc[r15["area_km2"] >= 2].copy()
r18 = r18.loc[r18["area_km2"] >= 2].copy()

print("s2")

print(len(r1))
print(len(r10))
print(len(r11))
print(len(r12))
print(len(r14))
print(len(r15))
print(len(r18))




r1 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/LX-C02-TOA_RGI-v7-01.geojson")
r10 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/LX-C02-TOA_RGI-v7-10.geojson")
r11 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/LX-C02-TOA_RGI-v7-11.geojson")
r12 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/LX-C02-TOA_RGI-v7-12.geojson")
                   
r14 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/LX-C02-TOA_RGI-v7-14.geojson")
r15 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/LX-C02-TOA_RGI-v7-15.geojson")
                   
r18 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/LX-C02-TOA_RGI-v7-18.geojson")



r1 = r1.loc[r1["area_km2"] >= 2].copy()
r10 = r10.loc[r10["area_km2"] >= 2].copy()
r11 = r11.loc[r11["area_km2"] >= 2].copy()
r12 = r12.loc[r12["area_km2"] >= 2].copy()
r14 = r14.loc[r14["area_km2"] >= 2].copy()
r15 = r15.loc[r15["area_km2"] >= 2].copy()
r18 = r18.loc[r18["area_km2"] >= 2].copy()
print()
print("LX")

print(len(r1))
print(len(r10))
print(len(r11))
print(len(r12))
print(len(r14))
print(len(r15))
print(len(r18))


r10 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis\Switzerland_Analysis/new/Switzerland_debris_ratio_per_elevation_band.gpkg")
r11 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis\Switzerland_Analysis/new/SGI_2016_glaciers.shp")
r2 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis\Switzerland_Analysis/new/SAM_11CentralEurope_minGl2km2_combinedDebrisCover.geojson")

r12 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis\Switzerland_Analysis/new/LX-C02-TOA_RGI-v7-11.geojson")
r14 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis\Switzerland_Analysis/new/S2_RGI_v7_11_central_europe_debris_cover.gpkg")

#r10 = r10.loc[r10["area_km2"] >= 2].copy()
r11 = r11.loc[r11["area_km2"] >= 2].copy()
r12 = r12.loc[r12["area_km2"] >= 2].copy()
r14 = r14.loc[r14["area_km2"] >= 2].copy()

print("Swiss Alpsss")
print("matching")
print(len(r10)/4)
print("sgi")
print(len(r11))
print("sam")
print(len(r2))
print("lx")
print(len(r12))
print("s2")
print(len(r14))