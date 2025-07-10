---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import os
import sys

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

sys.path.append('code_src/')
from data_structures_spec import MultiIndexDFObject

from plot_functions import create_figures
from sample_selection import clean_sample

```

```python
coords = []
labels = []
coords.append(SkyCoord("{} {}".format("+53.15508", "-27.80178"), unit=(u.deg, u.deg)))
labels.append("JADESGS-z7-01-QU")

coords.append(SkyCoord("{} {}".format("+53.15398", "-27.80095"), unit=(u.deg, u.deg)))
labels.append("TestJWST")


sample_table = clean_sample(coords, labels, precision=2.0 * u.arcsecond, verbose=1)
```

```python
df_spec = MultiIndexDFObject()
```

```python
%%time
# Get Euclid Spectra
from Euclid_functions import Euclid_get_spec
df_spec_Euclid = Euclid_get_spec(sample_table=sample_table, search_radius_arcsec=10)
df_spec.append(df_spec_Euclid)
```

```python
from astropy.io import fits
from astropy.table import QTable
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astroquery.ipac.irsa import Irsa

table_1dspectra = 'euclid.objectid_spectrafile_association_q1'
obj_id = 2689918641685825137
adql_query = f"""
SELECT * FROM {table_1dspectra}
WHERE objectid = {obj_id}
"""
result = Irsa.query_tap(adql_query).to_table()

# Inputs
uri = result[0]["uri"]
hdu_index = result[0]["hdu"]
file_uri = f"https://irsa.ipac.caltech.edu/{uri}"  # fixed missing slash

# Open the FITS file and read the spectrum table
with fits.open(file_uri, ignore_missing_simple=True) as hdul:
    spectrum = QTable.read(hdul[hdu_index], format="fits")
    spec_header = hdul[hdu_index].header

# Extract FSCALE from header
fscale = spec_header.get("FSCALE", 1.0)

# Extract columns and apply scaling
wave = spectrum["WAVELENGTH"] # already in Angstroms
signal_scaled = spectrum["SIGNAL"] * fscale * u.erg / u.second / (u.centimeter**2) / u.angstrom
error_scaled = np.sqrt(spectrum["VAR"]) * fscale * signal_scaled.unit

# Convert MASK to plain array and define good vs bad bins
mask = np.array(spectrum["MASK"])
bad_mask = (mask % 2 == 1) | (mask >= 64)
valid_mask = ~bad_mask

# Plot the scaled flux with mask filtering
plt.figure(figsize=(12, 6))
plt.plot(wave[valid_mask].to(u.micron), signal_scaled[valid_mask], color='black', label='Flux')
plt.plot(wave[valid_mask].to(u.micron), error_scaled[valid_mask], color='gray', alpha=0.5, label='Error')

plt.xlabel("Wavelength [μm]")
plt.ylabel("Flux [erg / s / cm² / Å]")
plt.title("Euclid NISP 1D Spectrum")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

```python

```
