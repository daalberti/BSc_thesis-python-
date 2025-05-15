import os
import numpy as np
import rasterio
import subprocess
import shutil



regions = [
    #"01Alaska", 
    #"10NorthAsia", 
    #"11CentralEurope",
    #"12CaucasusMiddleEast", 
    #"14SouthAsiaWest",
    "15SouthAsiaEast", 
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

base_dir = r"C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/DEM/global"
gdalbuildvrt_path = r"C:/Users/david/anaconda3/Library/bin/gdalbuildvrt.exe"
gdal_translate_path = r"C:/Users/david/anaconda3/Library/bin/gdal_translate.exe"
gdalwarp_path = r"C:/Users/david/anaconda3/Library/bin/gdalwarp.exe"

for region in regions:
    target_crs = region_crs[region]

    input_dir = os.path.join(base_dir, region)
    masked_dir = os.path.join(base_dir, f"{region}_masked")
    reprojected_dir = os.path.join(base_dir, f"{region}_reprojected")
    vrt_path = os.path.join(base_dir, f"DEM_{region}.vrt")
    output_path = os.path.join(base_dir, f"DEM_{region}_merged_FINAL.tif")

    os.makedirs(masked_dir, exist_ok=True)
    os.makedirs(reprojected_dir, exist_ok=True)
    if os.path.exists(reprojected_dir):
        shutil.rmtree(reprojected_dir)
        os.makedirs(reprojected_dir)

    # Step 1: Replace 0 with NaN and save to _masked
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(".tif"):
            in_fp = os.path.join(input_dir, fname)
            out_fp = os.path.join(masked_dir, fname)

            with rasterio.open(in_fp) as src:
                data = src.read(1).astype('float32')
                data[data == 0] = np.nan

                meta = src.meta.copy()
                meta.update(dtype='float32', nodata=np.nan, count=1)

                with rasterio.open(out_fp, 'w', **meta) as dst:
                    dst.write(data, 1)

    print(f"✅ Masked input DEMs for {region} (0 ➔ NaN) to: {masked_dir}")

    # Step 2: Reproject masked files to region-specific CRS
    for fname in os.listdir(masked_dir):
        if fname.lower().endswith(".tif"):
            in_fp = os.path.join(masked_dir, fname)
            out_fp = os.path.join(reprojected_dir, fname)

            subprocess.run([
                gdalwarp_path,
                "-t_srs", target_crs,
                "-r", "bilinear",
                "-overwrite",
                "-dstnodata", "nan",
                in_fp,
                out_fp
            ], check=True)

    print(f"✅ Reprojected masked DEMs for {region} to {target_crs}: {reprojected_dir}")

    # Step 3: Build VRT from reprojected files
    input_files = [
        os.path.join(reprojected_dir, f)
        for f in os.listdir(reprojected_dir)
        if f.lower().endswith(".tif")
    ]

    input_files.sort()  # Ensures consistent order (alphabetical, first = priority)

    subprocess.run([
        gdalbuildvrt_path,
        "-resolution", "highest",  # use highest resolution from inputs
        "-overwrite",
        "-vrtnodata", "nan",
        vrt_path,
        *input_files
    ], check=True)


    print(f"✅ VRT created at: {vrt_path}")

    # Step 4: Delete old output if needed
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except PermissionError:
            print(f"❌ Cannot delete {output_path} – is it open in QGIS or Explorer?")
            continue

    # Step 5: Translate VRT to GeoTIFF
    subprocess.run([
        gdal_translate_path,
        "-co", "BIGTIFF=YES",
        "-co", "COMPRESS=DEFLATE",
        vrt_path,
        output_path
    ], check=True)

    print(f"✅ Final merged DEM for {region} saved to: {output_path}")
