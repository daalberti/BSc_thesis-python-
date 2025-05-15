import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
from rasterio.mask import mask
from rasterio.features import rasterize
import rasterio
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches 
from rasterio.warp import calculate_default_transform, reproject, Resampling


# Parameters
num_bands = 20

# Load data
glacier_rgi7 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/matching_files/matching_glaciers_Switzerland_RGI7.gpkg")
glacier_sam = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/matching_files/sam_matching_glacier_outlines.gpkg")
glacier_sgi = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/matching_files/sgi_matching_glacier_outlines.gpkg")
debris = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/matching_files/Merged_no_duplicates_final.gpkg")
dem_path = "C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/DEM/DEM_Subregion_11-01.tif" 
src_dem = rasterio.open(dem_path)

print(src_dem.crs)

# ensure crs is epsg 2056
glacier_rgi7 = glacier_rgi7.to_crs(epsg=2056)
glacier_sam = glacier_sam.to_crs(epsg=2056)
glacier_sgi = glacier_sgi.to_crs(epsg=2056)
debris = debris.to_crs(epsg=2056)

# Prepare list to store results
all_results = []
glacier_pixel_weights = {}



# Get unique glacier IDs to plot
rgi_ids_to_plot = ["RGI2000-v7.0-G-11-02216","RGI2000-v7.0-G-11-02596","RGI2000-v7.0-G-11-00743","RGI2000-v7.0-G-11-01189","RGI2000-v7.0-G-11-00978","RGI2000-v7.0-G-11-01457","RGI2000-v7.0-G-11-02611"]  # Add the list of rgi_ids here if you want to plot more

for idx, rgi_id in enumerate(rgi_ids_to_plot):
    # Prepare to plot side by side in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))  # 2 rows, 2 columns
    axes = axes.flatten()  # Flatten to make it easier to access each subplot
    # Loop through each source for the given rgi_id
    sources = debris[debris['rgi_id'] == rgi_id]['source'].unique()

    for source_idx, source in enumerate(sources):
        rgi7glacier = glacier_rgi7
        # Choose glacier dataset based on the source of the debris
        if 'lx' in source or 's2' in source:
            glacier = glacier_rgi7
            debris_subset = debris[(debris['rgi_id'] == rgi_id) & (debris['source'] == source)].copy()
            year = debris_subset['img_year'].iloc[0]
        
        elif 'sam' in source:
            glacier = glacier_sam
            debris_subset = debris[(debris['rgi_id'] == rgi_id) & (debris['source'] == source)].copy()
            
            year = debris_subset['img_year'].iloc[0]
        elif 'sgi' in source:
            glacier = glacier_sgi
            debris_subset = debris[(debris['rgi_id'] == rgi_id) & (debris['source'] == source)].copy()
            sgi_name = debris_subset['name_sgi'].iloc[0]
            year = debris_subset['img_year'].iloc[0]
        else:
            continue  # Skip if no valid source
        

        # Filter glacier geometries
        glacier_geom_mapping = [mapping(geom) for geom in glacier[glacier['rgi_id'] == rgi_id].geometry]
        rgi7glacier_geom_mapping = [mapping(geom) for geom in rgi7glacier[rgi7glacier['rgi_id'] == rgi_id].geometry]
        debris_subset = debris[debris['rgi_id'] == rgi_id][debris['source'] == source].copy()

        try:
            # Mask DEM to glacier extent
            glacier_dem, glacier_transform = mask(src_dem, glacier_geom_mapping, crop=True)
            rgi7glacier_dem, rgi7glacier_transform = mask(src_dem, rgi7glacier_geom_mapping, crop=True)
        except Exception as e:
            print(f"Skipping {rgi_id} - {source} due to masking error: {e}")
            continue

        glacier_dem = glacier_dem[0]
        glacier_dem = np.where(glacier_dem == src_dem.nodata, np.nan, glacier_dem)
        glacier_dem[glacier_dem == 0] = np.nan

        rgi7glacier_dem = rgi7glacier_dem[0]
        rgi7glacier_dem = np.where(rgi7glacier_dem == src_dem.nodata, np.nan, rgi7glacier_dem)
        rgi7glacier_dem[rgi7glacier_dem == 0] = np.nan

        if np.all(np.isnan(glacier_dem)):
            print(f"Skipping {rgi_id} - {source} due to empty DEM")
            continue
        if np.all(np.isnan(rgi7glacier_dem)):
            print(f"Skipping {rgi_id} - {source} due to empty DEM")
            continue

        # Use RGI7 glacier to define elevation bins
        glacier_rgi_geom = [mapping(geom) for geom in glacier_rgi7[glacier_rgi7['rgi_id'] == rgi_id].geometry]

        try:
            rgi7_dem, rgi7_transform = mask(src_dem, glacier_rgi_geom, crop=True)

        except Exception as e:
            print(f"Skipping {rgi_id} - {source} due to RGI7 DEM masking error: {e}")
            continue

        rgi7_dem = rgi7_dem[0]
        rgi7_dem = np.where(rgi7_dem == src_dem.nodata, np.nan, rgi7_dem)
        rgi7_dem[rgi7_dem == 0] = np.nan

        if np.all(np.isnan(rgi7_dem)):
            print(f"Skipping {rgi_id} - {source} due to empty RGI7 DEM")
            continue

        elev_min, elev_max = np.nanmin(rgi7_dem), np.nanmax(rgi7_dem)
        bin_size = (elev_max - elev_min) / num_bands

        
        normalized = (glacier_dem - elev_min) / bin_size
        band_indices = np.where(np.isnan(normalized), np.nan, normalized.astype(int) + 1)
        band_indices = np.clip(band_indices, 1, num_bands)

        # Rasterize the debris polygons into a mask (same shape as glacier_dem)
        debris_geom = [mapping(geom) for geom in debris_subset.geometry]  # Convert geometry to mapping
        debris_mask = rasterio.features.rasterize(
            [(geom, 1) for geom in debris_geom],  # Mark debris pixels with value 1
            out_shape=glacier_dem.shape,  # Shape of the output array (same as glacier DEM)
            transform=glacier_transform,  # Transform to align with the DEM
            fill=0,  # Fill non-debris areas with 0
            dtype='uint8'  # Integer type for mask
        )
        debris_mask = np.where(debris_mask == 0, np.nan, debris_mask)
        # Define a custom colormap that assigns brown to 1 and transparent to 0
        # Define a custom colormap: one color only for value 1
        cmap = ListedColormap(['#8c715a'])
        norm = BoundaryNorm([0.5, 1.5], cmap.N)
        

        # Plot the DEM with elevation band contours and debris cover for each source
        ax = axes[source_idx]  # Assign subplot based on the source index

        # === Add RGI7 glacier background with elevation bands ===

        # Use the same rgi7_dem and bin edges as before
        normalized_rgi7 = (rgi7_dem - elev_min) / bin_size
        band_indices_rgi7 = np.where(
            np.isnan(normalized_rgi7),
            np.nan,
            normalized_rgi7.astype(int) + 1
        )
        band_indices_rgi7 = np.clip(band_indices_rgi7, 1, num_bands)
        
        #plot rgi7 background
        rgi7_gray_mask = np.where(np.isnan(rgi7glacier_dem), np.nan, 1)

        extent = [
            rgi7glacier_transform[2],
            rgi7glacier_transform[2] + rgi7glacier_transform[0] * rgi7glacier_dem.shape[1],
            rgi7glacier_transform[5] + rgi7glacier_transform[4] * rgi7glacier_dem.shape[0],
            rgi7glacier_transform[5]
        ]

        extent_contours = [
            rgi7glacier_transform[2],  # xmin (left)
            rgi7glacier_transform[2] + rgi7glacier_transform[0] * rgi7glacier_dem.shape[1],  # xmax (right)
            rgi7glacier_transform[5],  # ymax (top)
            rgi7glacier_transform[5] + rgi7glacier_transform[4] * rgi7glacier_dem.shape[0]  # ymin (bottom)
        ]

        ax.imshow(rgi7_gray_mask, cmap=ListedColormap(['#febf4a']), alpha=0.9, zorder=0, extent = extent)

        im_glacier = ax.imshow(glacier_dem, cmap='Blues', alpha=1, extent = extent)  # Only the glacier with hues of blue

        # Overlay the elevation band contours
        for b in range(1, num_bands + 1):
            band_mask = (band_indices == b)
            # Generate contours for the current band
            ax.contour(band_mask, levels=[0.5], colors='black', linewidths=0.3, extent = extent_contours)

        # Overlay the debris mask with custom colormap and no background
        im_debris = ax.imshow(debris_mask, cmap = cmap, alpha=1, extent = extent)  # Red for debris with no background

        # Overlay RGI7 glacier outline (dotted line)
        # Overlay RGI7 glacier outline as a dotted black line (no fill)



        # Set title and labels for each subplot
        
        ax.set_title(f"{source} ({year})", fontsize = 18)
        ax.axis('off')  # Hide axes for cleaner display




    # Elevation Colorbar (Horizontal)
    fig.colorbar(im_glacier, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1, label="Elevation (m)")
    fig.suptitle(f"Glacier DEM with Elevation Bins overlaid with Debris Cover - {sgi_name} ({rgi_id})", font = 'serif', fontsize=19)
    # Debris Colorbar with a Rectangle (Horizontal)
    debris_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='#8c715a', edgecolor='black', lw=1)
    rgi7_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='#febf4a', edgecolor='black', lw=1)


    # Place the legend for debris cover in the lower left outside the subplots
    axes[0].legend(
        [debris_patch, rgi7_patch],
        ['Debris Cover', 'RGI 7.0 Glacier Outline'],
        loc='center left',
        bbox_to_anchor=(-1.1, -1.44  ),
        fontsize = 18
    )

    plt.subplots_adjust(bottom=0.2) 
    #plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()
