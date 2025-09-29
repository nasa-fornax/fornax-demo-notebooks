---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# This notebook will do the following:

* Queries metadata for all Chandra observations using HEASARC VO services.

* Extracts their sky coordinates.

* Plots them on an all-sky map using matplotlib Aitoff projection.

* Queries all sources along the Galactic plane from the Chandra Source Catalog and the Fermi Source Catalog

* Performs a simple cross-match of the sources, saved to a new table

* Assesses the properties of the new sample of sources that have a CXC and Fermi counterpart


# Dependencies

* astropy
* pyvo
* matplotlib
* healpy
* os
* astroquery
* numpy
* tqdm
* concurrent


## Step 1: Imports

```python
#%pip install -r requirements.txt --quiet
```

```python
import pyvo as vo
import healpy
import os
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.heasarc import Heasarc
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
import numpy as np
from astropy.table import unique, vstack
from astropy.wcs import WCS
%matplotlib inline
plt.style.use(astropy_mpl_style)
```

```python
import astropy
import matplotlib
print("Astropy version:", astropy.__version__)
print("Matplotlib version:", matplotlib.__version__)
```

## Step 2: Use HEASARC VO Service to access Chandra observation metdata

```python
heasarc = vo.dal.TAPService("https://heasarc.gsfc.nasa.gov/xamin/vo/tap")
```

## Step 3: Run a ADQL query to get all observation positions

```python
query = """
SELECT TOP 99999999 obsid, ra, dec, exposure, detector, time, name
FROM chanmaster
WHERE ra IS NOT NULL AND dec IS NOT NULL
"""


results = heasarc.search(query)
table = results.to_table()
table[:5]
```

## Step 4: Define the coordinates using SkyCoord

```python
coords = SkyCoord(ra=table['ra'],
                  dec=table['dec'])
```

## Step 5: Plot an all-sky map in Aitoff projection

```python
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="aitoff")

mask = table['exposure'] != 0
exp_log = np.log10(table['exposure'][mask]/1000)

# Convert to radians and flip for Aitoff
ra_rad = coords.ra.wrap_at(180*u.deg).radian[mask]
dec_rad = coords.dec.radian[mask]

sc = ax.scatter(ra_rad, dec_rad, c=exp_log,s=1, alpha=0.5, cmap='plasma')
cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
cb.set_label("log10([exposure time]) (ks)",fontsize=12)
ax.grid(True)
plt.title("Chandra Observations (All-Sky)",fontsize=12,pad=20)
plt.show()
```

## Step 6: Find Chandra source locations and fluxes for all-sky using the CSC

### Find those that lie along the Galactic plane and plot the sky map.

```python
default_cols = ['name', 'ra', 'dec', 'b_flux_ap']
columns_str = ', '.join(default_cols)
# construct a query for all entries; use TOP with a large number greater than the server's 1e5 LIMIT
query = f"SELECT TOP 9999999 {columns_str} FROM csc"
sample = heasarc.search(query).to_table()
```

```python
print(sample)
```

There are a total of 407806 sources in the CSC. Now let's grab the relevant properties we need.

```python
query = """
SELECT TOP 9999999 name, ra, dec, b_flux_ap
FROM csc
WHERE b_flux_ap > 0
"""

sample = heasarc.search(query)
sample_table = sample.to_table()
sample_table[:5]
```

```python
print(len(sample_table))
print(f"{sample_table['b_flux_ap'].min():.2e}")
print(sample_table['b_flux_ap'].max())
print(np.sum(sample_table['b_flux_ap'])/len(sample_table))
```

We can check the sample properties in the cell above to make sure we are getting the same output each time. See also [GitHub issue #3387 in Astroquery](https://github.com/astropy/astroquery/issues/3387).

In principle, one could try:

``sample = heasarc.query_region(
SkyCoord(0, 0, unit='deg',frame='galactic'),
spatial='all-sky',
catalog='csc',
maxrec=None
)``

Once the following bugs are resolved:
* fields parameter errors out every time.
* maxrec messes with the query, sending a random sample that matches the criteria for a maximum of 1e5 sources. 

Next we mask sources that meet the Galactic plane criteria: |b|< 10&deg;.

```python
# Create SkyCoord from the sample table
coords = SkyCoord(ra=sample_table['ra'], dec=sample_table['dec'], unit='deg')
gal_coords = coords.galactic

# Apply Galactic latitude filter: |b| <= 10°
mask = np.abs(gal_coords.b.deg) <= 10

# Filter coordinates and data
filtered_coords = gal_coords[mask]
filtered_flux = sample_table['b_flux_ap'][mask]
filtered_names = sample_table['name'][mask]

print(f"Total sources in catalog: {len(coords)}")
print(f"Number of filtered sources: {len(filtered_coords)}")
```

Next, we will wrap the longitude coordinates for Aitoff projection. We did this above for Chandra observations, too.

```python
print("Longitude range (unwrapped):", filtered_coords.l.min().deg, "to", filtered_coords.l.max().deg)
print("Longitude range (wrapped):", filtered_coords.l.wrap_at(180*u.deg).min().deg, "to", filtered_coords.l.wrap_at(180*u.deg).max().deg)
```

```python
flux_log = np.log10(filtered_flux)

l_rad = filtered_coords.l.wrap_at(180*u.deg).radian
b_rad = filtered_coords.b.radian

plt.figure(figsize=(10, 6))
ax = plt.subplot(111, projection='aitoff')

sc = ax.scatter(l_rad, b_rad, c=flux_log, cmap='plasma', s=5, alpha=0.7)
cb = plt.colorbar(sc, orientation='horizontal', pad=0.05)
cb.set_label("log10([b_flux_ap]) (erg/cm2/s)",fontsize=12)

ax.grid(True)
plt.title(r"Chandra Source Catalog Sources within $\pm$ 10$\circ$ of Galactic Plane",fontsize=12,pad=20)
plt.show()
```

We can plot a histogram of source count by longitude, too. Naturally the Galactic Center sees the largest number of sources. 

```python
l_deg = filtered_coords.l.deg

plt.hist(l_deg, bins=36, range=(0, 360))
plt.xlabel('Galactic Longitude (deg)',fontsize=10)
plt.ylabel('Number of sources',fontsize=10)
plt.title('Distribution of sources near Galactic plane by longitude',fontsize=12)
plt.show()
```

The code below plots the sources along the Galactic plane again, but using the Astropy WCS package so we can display only the Galactic plane.

```python
# 1. Build a WCS header manually for Galactic coordinates
wcs = WCS(naxis=2)
wcs.wcs.ctype = ['GLON-TAN', 'GLAT-TAN']  # Galactic longitude/latitude
wcs.wcs.crval = [0, 0]  # Center of the plot in l and b
wcs.wcs.crpix = [900, 50]  # Reference pixel
wcs.wcs.cdelt = [-0.2, 0.2]  # Degrees per pixel
wcs.wcs.cunit = ['deg', 'deg']

# 2. Prepare your data (Galactic longitudes/latitudes in degrees)
l_deg = filtered_coords.l.wrap_at(180 * u.deg).deg
b_deg = filtered_coords.b.deg
flux_log = np.log10(filtered_flux)

# 3. Convert world coords (deg) to pixel coords
x_pix, y_pix = wcs.wcs_world2pix(l_deg, b_deg, 0)

# 4. Plot
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(111, projection=wcs)

sc = ax.scatter(x_pix, y_pix, c=flux_log, cmap='cool', s=5, alpha=0.7)

cb = plt.colorbar(sc, orientation='vertical', pad=0.05)
cb.set_label('log10(Broadband Flux)')

# 5. Set axis limits in pixel coords to match lon/lat range
ax.set_xlim(0, 1850)  # 360 deg / 0.2 deg per pixel = 1800 pixels
ax.set_ylim(0, 100)   # 20 deg / 0.2 deg per pixel = 100 pixels

# 6. Axis labels, ticks, grid (in world coordinates)
ax.coords.grid(True, color='grey', ls='-')
ax.coords[0].set_axislabel('Galactic Longitude (deg)')
ax.coords[1].set_axislabel('Galactic Latitude (deg)')
ax.coords[0].set_ticks(spacing=20 * u.deg)
ax.coords[1].set_ticks(spacing=5 * u.deg)

plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
plt.show()
```

## Step 7: Perform a cross-match of the Fermi and CSC Catalogs using ADQL

What the query will filter:
* $ |b| < 10\,$&deg; 
* CSC flux > 0
* A CSC source must fall within the error circle of the Fermi source defined by ``circle('ICRS', fsc.ra, fsc.dec, fsc.semi_minor_axis_95)``

```python
query = """
    select fsc.name,fsc.lii,fsc.bii, fsc.semi_major_axis_95, fsc.energy_flux, fsc.source_type, csc.name, csc.lii, csc.bii, csc.b_flux_ap 
    from fermilpsc as fsc, csc 
    where ( fsc.bii between -10 and 10 and csc.b_flux_ap > 0)
    and contains(
        point('ICRS', csc.ra, csc.dec),
        circle('ICRS', fsc.ra, fsc.dec, fsc.semi_major_axis_95)
      ) = 1"""
result = heasarc.search(query)
```

```python
table = result.to_table()
new_names = [
    "fermi_name", "fermi_l", "fermi_b", "fermi_r95", "fermi_flux", "fermi_class",
    "csc_name", "csc_l", "csc_b", "csc_flux"]
table.rename_columns(table.colnames,new_names)
print(table)
```

There are 9162 sources that have a CSC counterpart within the Fermi error circle. Note: This includes several of the same Fermi source overlapping with multiple CSC sources. We will filter this further below. 

```python
csc_coords = SkyCoord(l=table['csc_l'], b=table['csc_b'], frame='galactic')
fermi_coords = SkyCoord(l=table['fermi_l'], b=table['fermi_b'], frame='galactic')

# Compute angular separation
sep2d = csc_coords.separation(fermi_coords)

# Add the information as a new column to the table
table['sep_deg'] = sep2d.deg
```

### Plot a sky map of the matched table, plotting each source only once even if it appears more than once in the table just above. 

```python
# Unique CSC and Fermi source rows
unique_csc = unique(table, keys='csc_name')
unique_fermi = unique(table, keys='fermi_name')

# CSC SkyCoords
csc_coords = SkyCoord(l=unique_csc['csc_l'], b=unique_csc['csc_b'], unit='deg', frame='galactic')
csc_lons = csc_coords.l.wrap_at(180 * u.deg).radian
csc_lats = csc_coords.b.radian

# Fermi SkyCoords
fermi_coords = SkyCoord(l=unique_fermi['fermi_l'], b=unique_fermi['fermi_b'], unit='deg', frame='galactic')
fermi_lons = fermi_coords.l.wrap_at(180 * u.deg).radian
fermi_lats = fermi_coords.b.radian

# Fermi R95 sizes
marker_scale = 100
fermi_r95_deg = unique_fermi['fermi_r95']
fermi_sizes = (fermi_r95_deg * marker_scale)**2

# Plot
plt.figure(figsize=(12, 6))
ax = plt.subplot(111, projection='aitoff')
ax.grid(True)

ax.scatter(csc_lons, csc_lats, s=5, color='blue', alpha=0.6, label='CSC source')
ax.scatter(fermi_lons, fermi_lats, s=fermi_sizes, edgecolor='red', facecolor='none', alpha=0.7)
ax.scatter(0, 0, s=5, edgecolor='red', facecolor='none', alpha=0.7, label='Fermi R95')

ax.set_title("CSC sources with 4FGL counterparts (Galactic Coordinates)", pad=20, fontsize=12)
ax.legend(loc='best')
plt.subplots_adjust(bottom=0.1, top=0.95)
plt.show()
```

The code below plots the matched table along the Galactic plane again, but using the Astropy WCS package so we can display only the Galactic plane.

```python
# Log flux ratio
flux_ratio = np.log10(table['csc_flux'] / table['fermi_flux'])

# Build a WCS header manually for Galactic coordinates
wcs = WCS(naxis=2)
wcs.wcs.ctype = ['GLON-TAN', 'GLAT-TAN']  # Galactic longitude/latitude
wcs.wcs.crval = [0, 0]  # Center of the plot in l and b
wcs.wcs.crpix = [900, 50]  # Reference pixel
wcs.wcs.cdelt = [-0.2, 0.2]  # Degrees per pixel
wcs.wcs.cunit = ['deg', 'deg']

# CSC SkyCoords
csc_coords = SkyCoord(l=table['csc_l'], b=table['csc_b'], unit='deg', frame='galactic')
csc_lons = csc_coords.l.wrap_at(180 * u.deg).deg
csc_lats = csc_coords.b.deg

# Fermi SkyCoords
fermi_coords = SkyCoord(l=unique_fermi['fermi_l'], b=unique_fermi['fermi_b'], unit='deg', frame='galactic')
fermi_lons = fermi_coords.l.wrap_at(180 * u.deg).deg
fermi_lats = fermi_coords.b.deg

# Fermi R95 sizes
marker_scale = 100
fermi_r95_deg = unique_fermi['fermi_r95']
fermi_sizes = (fermi_r95_deg * marker_scale)**2

# Convert to pixel coords
csc_x_pix, csc_y_pix = wcs.wcs_world2pix(csc_coords.l.wrap_at(180 * u.deg).deg, csc_coords.b.deg, 0)
fermi_x_pix, fermi_y_pix = wcs.wcs_world2pix(fermi_coords.l.wrap_at(180 * u.deg).deg, fermi_coords.b.deg, 0)

# Plot
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111, projection=wcs)

sc = ax.scatter(csc_x_pix, csc_y_pix, c=flux_ratio, cmap='coolwarm', s=8, alpha=0.8, label='CSC')

ax.scatter(fermi_x_pix, fermi_y_pix, s=fermi_sizes, edgecolor='red', facecolor='none', alpha=0.7)
ax.scatter(0, 0, s=5, edgecolor='red', facecolor='none', alpha=0.7, label='Fermi R95')

cb = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.05)
cb.set_label(r'Log$_{10}$(X-ray to Gamma-ray Flux Ratio)',fontsize=10)

ax.set_xlim(0, 1800)
ax.set_ylim(0, 100)

ax.coords.grid(True, color='gray', ls='--')
ax.coords[0].set_axislabel('Galactic Longitude (deg)',fontsize=10)
ax.coords[1].set_axislabel('Galactic Latitude (deg)',fontsize=10)
ax.coords[0].set_ticks(spacing=30 * u.deg)
ax.coords[1].set_ticks(spacing=5 * u.deg)

ax.set_title("Associated Sources by Flux Ratio", pad=10,fontsize=12)
ax.legend(loc='upper right')

plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)
plt.show()
```

## Step 8: Build a final table consolidating repeating entries

Get rid of repeating Fermi entries. However, we want to retain the information that several CSC sources can overlap with the same Fermi source, so we build a nested table.

```python
# Step 1: Unique Fermi sources
unique_pairs = unique(table, keys=['csc_name', 'fermi_name'])
unique_fermi_names = list(set(unique_pairs['fermi_name']))

# Step 2: Prepare row containers
rows = []

for fname in unique_fermi_names:
    # Filter for this Fermi source
    mask = unique_pairs['fermi_name'] == fname
    group = unique_pairs[mask]

    # Use first row for Fermi info
    f_row = group[0]

    # Extract CSC info as Python lists
    csc_names = list(group['csc_name'])
    csc_fluxes = list(group['csc_flux'])
    csc_lons = list(group['csc_l'])
    csc_b_lats = list(group['csc_b'])

    n_matches = len(csc_names)

    # Append row
    rows.append((
        fname,
        f_row['fermi_l'],
        f_row['fermi_b'],
        f_row['fermi_flux'],
        f_row['fermi_r95'],
        f_row['fermi_class'],
        n_matches,
        csc_names,
        csc_fluxes,
        csc_lons,
        csc_b_lats
    ))

# Step 3: Build final table
grouped_display_table = Table(rows=rows, names=[
    'fermi_name', 'fermi_l', 'fermi_b', 'fermi_flux', 'fermi_r95','fermi_class',
    'num_csc_matches', 'csc_names', 'csc_fluxes', 'csc_glons', 'csc_glats'
])

print(len(rows))
```

There are 340 Fermi sources with CSC counterparts. Save the final table as a CSV. 

```python
export_table = grouped_display_table.copy()
export_table.sort(['fermi_l', 'fermi_b'])

# Convert all list-like columns to strings
for col in export_table.colnames:
    if isinstance(export_table[col][0], (list, tuple)):
        export_table[col] = [', '.join(str(x) for x in row) for row in export_table[col]]

# Write to CSV
export_table.write("grouped_csc_fermi_pairs.csv", format="csv", overwrite=True)
```

## Step 9: Let's plot some of the properties of the final table

How many CSC matches per Fermi source as a function of longitude. Again we see the Galactic Center have the largest number of matches. 

```python
# Count CSC matches per Fermi source using a dict
fermi_counts = {}
for fname in table['fermi_name']:
    fermi_counts[fname] = fermi_counts.get(fname, 0) + 1

fermi_lons = []
num_csc_matches = []

for fname, count in fermi_counts.items():
    # Find fermi longitude in unique_fermi
    idx = (unique_fermi['fermi_name'] == fname).nonzero()[0]
    if len(idx) == 0:
        continue
    lon = unique_fermi['fermi_l'][idx[0]]
    fermi_lons.append(lon)
    num_csc_matches.append(count)

plt.figure(figsize=(8,5))
plt.bar(fermi_lons, num_csc_matches, 100*unique_fermi['fermi_r95'], alpha=0.7,lw=2)
plt.xlabel('Fermi Galactic Longitude (deg)',fontsize=10)
plt.ylabel('Number of CSC Matches',fontsize=10)
plt.title('Number of CSC Matches vs Fermi Longitude', fontsize=12)
plt.ylim(0,100)
plt.grid(True)
plt.show()
```

What is more interesting are properties like the X- to gamma-ray flux ratios, which can help identify a specific source class or subclass. 

```python
log_flux_ratio = np.log10(table['csc_flux'] / table['fermi_flux'])

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # 1 row, 2 columns

# First plot
axes[0].scatter(table['fermi_l'], log_flux_ratio, alpha=0.6, color='C1')
axes[0].set_xlabel('Fermi Galactic Longitude (deg)', fontsize=10)
axes[0].set_ylabel('Log10(CSC Flux / Fermi Flux)', fontsize=10)
axes[0].grid(True)

# Second plot
axes[1].scatter(table['fermi_b'], log_flux_ratio, alpha=0.6, color='C2')
axes[1].set_xlabel('Fermi Galactic Latitude (deg)', fontsize=10)
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

Finally, we can plot the flux ratio as a function of source class (from the Fermi catalog). 

```python
fermi_classes_clean = [' '.join(c.split()) for c in table['fermi_class']]

class_counts = {cls: fermi_classes_clean.count(cls) for cls in set(fermi_classes_clean)}

# Define desired order explicitly - galactic to extragalactic
desired_order = ['PSR', 'psr', 'PWN', 'pwn', 'SNR', 'snr', 'SPP', 'spp','HMB', 'hmb', '', 'UNK', 'unk', 'SFR', 'MSP', 'msp',
                'glc', 'GC', 'FSRQ', 'fsrq', 'NOV', 'sey', 'bcu', 'BIN', 'bin', 'rdg', 'bll']

nonzero_classes = [cls for cls in desired_order if class_counts.get(cls, 0) > 0]
# Extract unique classes and map to numbers for plotting
class_to_num = {cls: i for i, cls in enumerate(nonzero_classes)}

x_vals = []
y_vals = []
for c, y in zip(fermi_classes_clean, log_flux_ratio):
    if c in class_to_num:
        x_vals.append(class_to_num[c])
        y_vals.append(y)

x_vals = np.array(x_vals)
y_vals = np.array(y_vals)

jitter = np.random.normal(0, 0.1, size=len(x_vals))
x_jittered = x_vals + jitter

plt.figure(figsize=(12,6))
plt.scatter(x_jittered, y_vals, alpha=0.6)
plt.xticks(range(len(nonzero_classes)), nonzero_classes, rotation=45)
plt.xlabel('Fermi Class', fontsize=10)
plt.ylabel('Log10(CSC Flux / Fermi Flux)', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
```

With this cross-matched set you can do lots of things:

* Do brighter gamma-ray sources have more X-ray counterparts? Make a scatter plot of CSC matches as a function of Fermi flux.

* Assess association reliability. Here we only do a superficial cross-match using Fermi's 95% semimajor axis of each source. Plot the difference between Fermi and CSC coordinates as a histogram. 

* What regions of the sky show a clustering of sources? Plot a sky map where the point size is proportional to the number of CSC matches. 

* How many sources that lie along the Galactic plane are classified as extragalactic source classes? From this, what is the probability any given source is Galactic in nature? 

* Create a new table of the subsample (maybe prioritizing one or more similar source classes) to explore further trends, etc. 


## Step 10: Cross-match the new table of sources with IRSA and MAST

It is interesting to note the extragalactic source classes falling within the Galactic plane. Can we cross match the table we just created with IRSA and MAST to gather IR and optical counterparts? IRSA includes 2MASS (IR) and MAST has GAIA (optical). 

```python
gaia_dr3 = vo.dal.TAPService("https://gea.esac.esa.int/tap-server/tap")
mass = vo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
```

```python
tables = gaia_dr3.tables 

gaia_table = tables['gaiadr3.gaia_source']

for col in gaia_table.columns:
    print(col.name, col.datatype, "-", col.description, col.unit)
```

```python
#for tname, tinfo in mass.tables.items():
#    print(tname, "-", tinfo.description)
```

```python
tables = mass.tables 

mass_table = tables['fp_psc']

for col in mass_table.columns:
    print(col.name, col.datatype, "-", col.description, col.unit)
```

Unfortunately adding DISTANCE() breaks the mass ADQL search. <a href="https://irsa.ipac.caltech.edu/docs/program_interface/TAP.html">IRSA TAP docs</a> 

Workaround:

``curl -o out.tbl "https://irsa.ipac.caltech.edu/TAP/sync?FORMAT=IPAC_TABLE&QUERY =SELECT+TOP+1+ra,dec,DISTANCE(POINT('ICRS',ra,dec),POINT('ICRS',210.80225,54.34894))+AS+dist_tap+FROM+fp_psc+WHERE+CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',210.80225,54.34894,0.006))=1+ORDER+BY+dist_tap+ASC"``

Or in Python:
```
from urllib.parse import quote

import requests

#2MASS query
    twomass_query = f"""
    SELECT TOP 3 designation, ra, dec, glon, glat, DISTANCE(POINT('ICRS', ra,dec),POINT('ICRS', {f_coord.ra.deg}, {f_coord.dec.deg})) AS dist
    FROM fp_psc
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {f_coord.ra.deg}, {f_coord.dec.deg}, {radius})
    )
    ORDER BY dist ASC
    """
    
encoded_query = quote(twomass_query)

url = f"https://irsa.ipac.caltech.edu/TAP/sync?FORMAT=IPAC_TABLE&QUERY={encoded_query}"

response = requests.get(url)

response.raise_for_status()

twomass_results = Table.read(BytesIO(response.content), format='ipac')
print(twomass_results)
```

I have not tested this explicitly yet! Anyway, this alternative is really only good for a small sample of sources, not 340 Fermi sources! For now we perform a query without DISTANCE()

```python
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


unit_row = {
        "fermi_name": "",
        "fermi_l" : "deg",
        "fermi_b" : "deg",
        "fermi_flux" : "erg/cm2/s",
        "fermi_r95" : "deg",
        "fermi_class" : "",
        "num_csc_matches": "",
        "csc_names" : "",
        "csc_glons" : "deg",
        "csc_glats" : "deg",
        "csc_fluxes" : "erg/cm2/s",
        "gaia_names": "",
        "gaia_classes": "gaia class probabilities (quasar/galaxy/star)",
        "gaia_glons": "deg",
        "gaia_glats": "deg",
        "gaia_mean_g_flux": "e-/s",
        "gaia_mean_g_mag": "mag",
        "gaia_variability": "flag",
        "sep_from_fermi_gaia": "arcsec",
        "sep_from_csc_gaia": "arcsec",
        "twomass_names": "",
        "twomass_glons": "deg",
        "twomass_glats": "deg",
        "sep_from_fermi_2mass": "arcsec",
        "sep_from_csc_2mass": "arcsec"
}

radius = 0.03

def process_source(row):
    fname = row['fermi_name']
    f_l, f_b = row['fermi_l'], row['fermi_b']
    f_coord = SkyCoord(l=f_l*u.deg, b=f_b*u.deg, frame="galactic").icrs
    
    #GAIA-DR3 query
    query = f"""
    SELECT TOP 3 source_id, ra, dec, l, b, phot_g_mean_flux, phot_g_mean_mag, phot_variable_flag, classprob_dsc_combmod_quasar, classprob_dsc_combmod_galaxy, classprob_dsc_combmod_star, DISTANCE(POINT('ICRS', ra,dec),POINT('ICRS', {f_coord.ra.deg}, {f_coord.dec.deg})) AS dist
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra,dec),
        CIRCLE('ICRS', {f_coord.ra.deg}, {f_coord.dec.deg}, {radius})
    )
    ORDER BY dist ASC
    """
    gaia_results = gaia_dr3.run_async(query).to_table()
    
    #2MASS query
    twomass_query = f"""
    SELECT TOP 3 designation, ra, dec, glon, glat
    FROM fp_psc
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {f_coord.ra.deg}, {f_coord.dec.deg}, {radius})
    )
    """
    twomass_results = mass.run_async(twomass_query).to_table()

    
    gaia_names, gaia_glons, gaia_glats = [], [], []
    gaia_mean_g_flux, gaia_mean_g_mag, gaia_variability = [], [], []
    gaia_classes, sep_fermi, sep_csc = [], [], []
    
    for g in gaia_results:
        g_coord = SkyCoord(l=g['l']*u.deg, b=g['b']*u.deg, frame="galactic")
        g_gal = g_coord.galactic
        gaia_names.append(str(g['source_id']))
        gaia_glons.append(g['l'])
        gaia_glats.append(g['b'])
        gaia_mean_g_flux.append(g['phot_g_mean_flux'])
        gaia_mean_g_mag.append(g['phot_g_mean_mag'])
        gaia_variability.append(g['phot_variable_flag'])
        gaia_classes.append((
            g['classprob_dsc_combmod_quasar'],
            g['classprob_dsc_combmod_galaxy'],
            g['classprob_dsc_combmod_star']
        ))
        
        #separation from Fermi centroid
        sep_fermi.append(g_coord.separation(f_coord).arcsec)
        
        #separation from closest CSC source
        csc_coords = None
        if row['num_csc_matches'] > 0:
            csc_coords = SkyCoord(l=row['csc_glons']*u.deg, b=row['csc_glats']*u.deg, frame="galactic")
            sep_csc.append(g_coord.separation(csc_coords).arcsec.min())
        else:
            sep_csc.append(np.nan)
    
    twomass_names, twomass_glons, twomass_glats, sep_fermi_mass, sep_csc_mass = [], [], [], [], []
    
    for m in twomass_results:
        m_coord = SkyCoord(l=m['glon']*u.deg, b=m['glat']*u.deg, frame="galactic")
        twomass_names.append(m['designation'])
        twomass_glons.append(m_coord.l.deg)
        twomass_glats.append(m_coord.b.deg)
        sep_fermi_mass.append(m_coord.separation(f_coord).arcsec)
        sep_csc_mass.append(np.min(m_coord.separation(csc_coords).arcsec))

        
    new_data = {
        "gaia_names": gaia_names,
        "gaia_classes": gaia_classes,
        "gaia_glons": gaia_glons,
        "gaia_glats": gaia_glats,
        "gaia_mean_g_flux": gaia_mean_g_flux,
        "gaia_mean_g_mag": gaia_mean_g_mag,
        "gaia_variability": gaia_variability,
        "sep_from_fermi_gaia": sep_fermi,
        "sep_from_csc_gaia": sep_csc,
        "twomass_names": twomass_names,
        "twomass_glons": twomass_glons,
        "twomass_glats": twomass_glats,
        "sep_from_fermi_2mass": sep_fermi_mass,
        "sep_from_csc_2mass": sep_csc_mass
    }
    
    merged_row = {**row, **new_data}  # Merge original + cross-match
    return merged_row
    
results = []
with ThreadPoolExecutor(max_workers=4) as executor:
    for r in tqdm(executor.map(process_source, grouped_display_table),
                  total=len(grouped_display_table),
                  desc="Cross-matching"):
        results.append(r)
        
final = [unit_row] + results
```

```python
col_order = [
    "fermi_name",
    "fermi_l",
    "fermi_b",
    "fermi_flux",
    "fermi_r95",
    "fermi_class",
    "num_csc_matches",
    "csc_names",
    "csc_fluxes",
    "csc_glons",
    "csc_glats",
    "gaia_names",
    "gaia_classes",
    "gaia_glons",
    "gaia_glats",
    "gaia_mean_g_flux",
    "gaia_mean_g_mag",
    "gaia_variability",
    "sep_from_fermi_gaia",
    "sep_from_csc_gaia",
    "twomass_names",
    "twomass_glons",
    "twomass_glats",
    "sep_from_fermi_2mass",
    "sep_from_csc_2mass"
]

# final is your list of dicts (unit row + data rows)
export_table = Table(rows=final, names=col_order)

# Convert list/tuple columns to strings
for col in export_table.colnames:
    new_col = []
    for val in export_table[col]:
        if isinstance(val, (list, tuple, np.ndarray)):
            if isinstance(val, np.ndarray) and val.shape == ():  # 0-d array
                new_col.append(str(val.item()))
            else:
                new_col.append(', '.join(str(x) for x in val))
        else:
            new_col.append(str(val))
    export_table[col] = new_col

# Write to CSV
export_table.write("grouped_csc_fermi_gaia_2mass_pairs.csv", format="csv", overwrite=True)
```

Next lets plot some interesting things about our new dataset! 

```python
fermi_classes = list(export_table['fermi_class'])

# Define desired order explicitly - galactic to extragalactic
desired_order = ['PSR', 'psr', 'PWN', 'pwn', 'SNR', 'snr', 'SPP', 'spp','HMB', 'hmb', '', 'UNK', 'unk', 'SFR', 'MSP', 'msp',
                'glc', 'GC', 'FSRQ', 'fsrq', 'NOV', 'sey', 'bcu', 'BIN', 'bin', 'rdg', 'bll']

nonzero_classes = [cls for cls in desired_order if class_counts.get(cls, 0) > 0]
# Extract unique classes and map to numbers for plotting

gaia_vals = []
twomass_vals = []

for c in nonzero_classes:
    indices = [i for i, fc in enumerate(fermi_classes) if fc == c]
    gaia_count = sum(len(export_table['gaia_names'][i]) > 0 for i in indices)
    twomass_count = sum(len(export_table['twomass_names'][i]) > 0 for i in indices)
    
    gaia_vals.append(gaia_count)
    twomass_vals.append(twomass_count)
    
width = 1.0
x = np.arange(len(nonzero_classes))

fig, ax = plt.subplots(figsize=(10,6))
ax.bar(x-width/2,gaia_vals, width=width, label='GAIA', color='skyblue')
ax.bar(x+width/2, twomass_vals, width=width, label='2MASS', color='orange')

ax.set_xticks(x)
ax.set_xticklabels(nonzero_classes,rotation=45, ha='right')
ax.set_ylabel("Number of Fermi Sources",fontsize=10)
ax.set_xlabel("Fermi Class", fontsize=10)
plt.title('Number of Fermi sources with GAIA/2MASS Counterparts by Class',fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()
```

```python
fermi_classes = list(export_table['fermi_class'])

# Define desired order explicitly - galactic to extragalactic
desired_order = ['PSR', 'psr', 'PWN', 'pwn', 'SNR', 'snr', 'SPP', 'spp','HMB', 'hmb', '', 'UNK', 'unk', 'SFR', 'MSP', 'msp',
                'glc', 'GC', 'FSRQ', 'fsrq', 'NOV', 'sey', 'bcu', 'BIN', 'bin', 'rdg', 'bll']

nonzero_classes = [cls for cls in desired_order]  # keep all, counts may be zero

# Count variability per class
variable_counts = [0]*len(nonzero_classes)
constant_counts = [0]*len(nonzero_classes)

for i, f_class in enumerate(fermi_classes):
    if f_class not in nonzero_classes:
        f_class = ''  # fallback for unknown/empty
    idx = class_to_num[f_class]
    
    gaia_var_str = export_table['gaia_variability'][i]
    if not gaia_var_str:
        continue
    
    gaia_var_list = [v.strip().upper() for v in gaia_var_str.split(',')]
    for var_flag in gaia_var_list:
        if var_flag == 'VARIABLE':
            variable_counts[idx] += 1
        else:
            constant_counts[idx] += 1

# Plot stacked histogram
x = np.arange(len(nonzero_classes))
width = 1.0

plt.figure(figsize=(10,6))
plt.bar(x, constant_counts, width=width, label='Not Variable', color='skyblue')
plt.bar(x, variable_counts, width=width, bottom=constant_counts, label='Variable', color='orange')

plt.xticks(x, nonzero_classes, rotation=45, ha='right')
plt.yscale('log')
plt.xlabel('Fermi Class', fontsize=10)
plt.ylabel('Number of Gaia Counterparts', fontsize=10)
plt.title('Gaia Variability per Fermi Class',fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()
```

```python
fermi_classes = list(export_table['fermi_class'])

# Define order explicitly - galactic → extragalactic
desired_order = ['PSR', 'psr', 'PWN', 'pwn', 'SNR', 'snr', 'SPP', 'spp',
                 'HMB', 'hmb', '', 'UNK', 'unk', 'SFR', 'MSP', 'msp',
                 'glc', 'GC', 'FSRQ', 'fsrq', 'NOV', 'sey', 'bcu', 'BIN', 'bin', 'rdg', 'bll']

# Map Fermi class to index
class_to_num = {cls: i for i, cls in enumerate(desired_order)}

# Initialize counts
variable_counts = [0] * len(desired_order)
total_counts = [0] * len(desired_order)

for i, f_class in enumerate(fermi_classes):
    if f_class not in class_to_num:
        f_class = ''  # fallback for empty/unknown
    idx = class_to_num[f_class]
    
    gaia_var_str = export_table['gaia_variability'][i]
    if not gaia_var_str:
        continue
    gaia_var_list = [v.strip().upper() for v in gaia_var_str.split(',')]
    
    # Count total Gaia counterparts and variable ones
    total_counts[idx] += len(gaia_var_list)
    variable_counts[idx] += sum(1 for v in gaia_var_list if v == 'VARIABLE')

# Compute fraction of variable Gaia counterparts
fraction_variable = [v / t if t > 0 else 0 for v, t in zip(variable_counts, total_counts)]

# Plot
x = np.arange(len(desired_order))
plt.figure(figsize=(10,6))
plt.bar(x, fraction_variable, color='orange', alpha=0.8)
plt.xticks(x, desired_order, rotation=45, ha='right')
plt.ylabel('Fraction of Variable Gaia Counterparts',fontsize=10)
plt.xlabel('Fermi Class',fontsize=10)
plt.title('Fraction of Variable Gaia Counterparts per Fermi Class',fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```
