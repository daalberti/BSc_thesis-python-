
# code structure and syntax was created with the aid of ChatGPT


# what this code does:
# - reads the Switzerland_combined file and determines modal img_year of datasets
# - plots the change of fractional debris cover between datasets in histograms (upward in time only)


import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10
#  Load combined data
combined = gpd.read_file(
    "C:/Users/david/OneDrive - Universität Zürich UZH/Jahr 3/BSc Thesis/Switzerland_Analysis/new/Switzerland_combined.gpkg"
)

# Extract the modal acquisition year for each source (for titles)
modal_year = (
    combined
      .groupby('source')['img_year']
      .agg(lambda x: x.mode().iloc[0])
)
year_lx, year_sam, year_s2, year_sgi = (
    modal_year['lx'],
    modal_year['sam'],
    modal_year['s2'],
    modal_year['sgi']
)

# Pivot from long (4 rows per glacier) to wide (1 row per glacier),
#    taking the *first* (and only) debris_cover_ratio for each glacier+source.
df_wide = (
    combined
      .pivot_table(
         index='rgi_id',
         columns='source',
         values='debris_cover_ratio',
         aggfunc='first'
      )
      .reset_index()
      .rename(columns={
         'lx':  'debris_lx',
         'sam': 'debris_sam',
         's2':  'debris_s2',
         'sgi': 'debris_sgi'
      })
)

# Define every pair of sources you want to compare
transitions = [
    ('lx',  'sam'),
    ('lx',  'sgi'),
    ('lx',  's2'),
    ('sam', 'sgi'),
    ('sam', 's2'),
    ('sgi', 's2'),
]

#  Compute the per‑glacier change columns
for src, dst in transitions:
    df_wide[f'change_{src}_{dst}'] = (
        df_wide[f'debris_{dst}'] - df_wide[f'debris_{src}']
    )

#Plot a 2×3 grid of histograms; each bar = change for one glacier
fig, axes = plt.subplots(2, 3, figsize=(18, 10), tight_layout=True)
axes = axes.ravel()

year_map = {'lx': year_lx, 'sam': year_sam, 's2': year_s2, 'sgi': year_sgi}

ncols = 3  
nrows = int(np.ceil(len(axes) / ncols))
for idx, (ax, (src, dst)) in enumerate(zip(axes.flat, transitions)):
    col = f'change_{src}_{dst}'
    data = df_wide[col].dropna()
    μ, σ = data.mean(), data.std()
    ax.axvline(0, color='grey', linewidth=2, linestyle='-', zorder=1)
    ax.hist(data, bins=30, edgecolor='black')
    ax.axvline(μ, linestyle='--', label=f'μ={μ:.3f}', color="red")
    ax.axvline(μ + σ, linestyle=':', label=f'+1σ={σ:.3f}', color="red")
    ax.axvline(μ - σ, linestyle=':', color="red")
    ax.set_title(f"{src.upper()} ({year_map[src]}) → {dst.upper()} ({year_map[dst]})",
                 font='serif', fontsize=17)

    row_idx, col_idx = divmod(idx, ncols)
    
    # Only left columns gets y-axis label
    if col_idx == 0:
        ax.set_ylabel("Frequency", fontsize=16.5, font='serif')
    else:
        ax.set_ylabel("")

    # Only bottom row gets x-axis label
    if row_idx == nrows - 1:
        ax.set_xlabel("Fractional Debris Cover Change", fontsize=16.5, font='serif')
    else:
        ax.set_xlabel("")

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 12)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(fontsize=16.5, loc='upper left')

plt.suptitle("Difference in fractional Debris Cover (Swiss Alps)", fontsize=26, font='serif')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
