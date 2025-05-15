# code structure and syntax was created with the help of ChatGPT.

# read RGI 7.0 (combined) file for glacier attributes
# read all_regions_combined file for the calculation of  fractional debris cover change per glaciers.
# inspection of variables used in linear regression -> tranformations
# perform multiple linear regression on chosen attributes
# visualize in plots.

import statsmodels.formula.api as smf
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stargazer.stargazer import Stargazer
import seaborn as sns

# Load data
gdf = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/all_together/all_regions_combined.gpkg")
rgi7 = gpd.read_file("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/all_together/RGI7_all_combined.gpkg")

# Select relevant attributes from rgi7
rgi_attributes = rgi7[['rgi_id', 'slope_deg', 'aspect_deg', 'zmed_m', 'area_km2', 'cenlat']]

gdf = gdf.drop(columns=['area_km2'], errors='ignore')
# Merge into gdf (applies to all three source entries per glacier)
gdf = gdf.merge(rgi_attributes, on='rgi_id', how='left')


# Pivot: one row per rgi_id, one column per source
pivoted = gdf.pivot_table(index='rgi_id', columns='source', values='debris_cover_ratio')



def max_s2_difference(row):
    val_lx = row.get('lx')
    val_s2 = row.get('s2')
    val_sam = row.get('sam')

    diffs = []
    if pd.notna(val_lx) and pd.notna(val_s2):
        diffs.append(val_s2 - val_lx)
    if pd.notna(val_sam) and pd.notna(val_s2):
        diffs.append(val_s2 - val_sam)

    if diffs:
        return max(diffs)  
    else:
        return np.nan

pivoted['FDC_change'] = pivoted.apply(max_s2_difference, axis=1)




# Reattach attributes and region (take one row per rgi_id from gdf)
attributes = gdf.drop_duplicates(subset='rgi_id')[
    ['rgi_id', 'slope_deg', 'aspect_deg', 'zmed_m', 'area_km2', 'cenlat', 'region']
]

# Merge 
result = pivoted[['FDC_change']].merge(attributes, left_index=True, right_on='rgi_id')



#check distribution of predictors:
print("area_km2: check distribution")
sns.histplot(result['area_km2'], bins=60, kde=True)
plt.title('Distribution of area_km2')
plt.xlabel('slope_deg')
plt.ylabel('Frequency')
plt.show()

print("Skewness:", result['area_km2'].skew())





# Transformations
result["logArea"] = np.log10(result['area_km2'])
result["logSlope"] = np.log10(result['slope_deg'])
result["logZmed"] = np.log10(result['zmed_m'])
result['abs_cenlat'] = result['cenlat'].abs()
result['sqrt_max_diff'] = np.sqrt(result['FDC_change'])


# Drop NA for regression
regression_data = result.dropna(subset=['sqrt_max_diff', 'logSlope', 'logArea', 'zmed_m', 'abs_cenlat'])


# Linear regression
model_linear = smf.ols('sqrt_max_diff ~ zmed_m + logSlope + area_km2 + abs_cenlat + region', data=regression_data).fit()
print(model_linear.summary())
model_log = smf.ols('sqrt_max_diff ~ zmed_m + logSlope + logArea + abs_cenlat + region', data=regression_data).fit()
print(model_log.summary())

stargazer = Stargazer([model_log])
latex_output = stargazer.render_latex()

with open("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/all_together/regression_table_stargazer.tex", "w") as f:
    f.write(latex_output)


table = model_log.summary().as_latex()

#Save to .tex file
with open("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Global analysis/all_together/regression_table.tex", "w") as f:
    f.write(table)




# Residuals from linear model
resid_linear = model_linear.resid
fitted_linear = model_linear.fittedvalues

# Residuals from log-transformed model
resid_log = model_log.resid
fitted_log = model_log.fittedvalues

# Plot both
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(x=fitted_linear, y=resid_linear, ax=axes[0])
axes[0].set_title("Residuals: slope_deg")
axes[0].axhline(0, color='black', linestyle='--')

sns.scatterplot(x=fitted_log, y=resid_log, ax=axes[1])
axes[1].set_title("Residuals: logSlope")
axes[1].axhline(0, color='black', linestyle='--')

plt.tight_layout()
#plt.show()

# Plotting
attributes_to_plot = ['slope_deg','logSlope', 'aspect_deg', 'zmed_m','logZmed', 'area_km2', 'logArea', 'abs_cenlat']

for attr in attributes_to_plot:
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        regression_data[attr],
        regression_data['sqrt_max_diff'],
        c=regression_data['region'].astype('category').cat.codes,  # Convert region to categorical colors
        cmap='tab20',
        alpha=0.7,
        edgecolor='black'
    )
    plt.xlabel(attr.replace('_', ' ').capitalize())
    plt.ylabel('√(Max Absolute Difference in Debris Cover Ratio)')
    plt.title(f'Max Difference vs {attr.replace("_", " ").capitalize()}')
    plt.colorbar(scatter, label='Region Code')
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
