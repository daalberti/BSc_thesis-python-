# code structure and syntax was created with the aid of ChatGPT


# what this code does:
# - calculates fractional debris cover change between S2 and LX from combined file
# - performs linear regression on selected attributes (extracted from SGI 2016)


import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
# Load the files 
sgi = gpd.read_file("c:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/SGI_matching_glacier_outlines.gpkg")
combined = gpd.read_file("c:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/Switzerland_combined.gpkg")

pivoted = combined.pivot_table(
    index='rgi_id',
    columns='source',
    values='debris_cover_ratio',
    aggfunc='first'
)
def s2_minus_lx(row):
    val_lx = row.get('lx')
    val_s2 = row.get('s2')

    if pd.notna(val_lx) and pd.notna(val_s2):
        return val_s2 - val_lx
    else:
        return np.nan

pivoted['FDC_change'] = pivoted.apply(s2_minus_lx, axis=1)

# Reset index to bring rgi_id back as a column
pivoted = pivoted.reset_index()

# Merge max_change to sgi based on rgi_id
sgi_with_change = sgi.merge(pivoted[['rgi_id', 'FDC_change']], on='rgi_id', how='left')


sgi_with_change.to_file("c:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/sgi_outlines_with_max_FDC_change.gpkg", driver="GPKG")



# Drop rows with missing data to avoid plotting errors
plot_data = sgi_with_change.dropna(subset=['area_km2', 'FDC_change'])
plot_data['logArea'] = np.log10(plot_data['area_km2'])
plot_data['logSlope'] = np.log10(plot_data['slope_deg'])
plot_data['sqrt_max_change'] = np.sqrt(plot_data['FDC_change'])





# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(plot_data['logArea'], plot_data['sqrt_max_change'], alpha=0.7, edgecolor='black')
plt.xlabel('Glacier Area (km²)')
plt.ylabel('Max Change in Debris Cover Ratio')
plt.title('Debris Cover Change vs Glacier Area')
plt.grid(True)
plt.tight_layout()
plt.show()

model = smf.ols('sqrt_max_change ~ logArea + masl_med + logSlope', data=plot_data).fit()
print(model.summary())


stargazer = Stargazer([model])
latex_output = stargazer.render_latex()

with open("C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/FINAL/CH_regression_table_stargazer.tex", "w") as f:
    f.write(latex_output)