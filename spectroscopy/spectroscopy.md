(spectroscopy)=
# Spectroscopy

These tutorials explore collecting and analyzing spectroscopic data from multiple archival resources.

## Extract Multi-Wavelength Spectroscopy from Archival Data

This tutorial collects fully reduced spectra from multiple NASA and non-NASA archives (SDSS, DESI, Keck, HST, JWST, Spitzer, Herschel) for a user-supplied list of sources.
Spectra are loaded using `specutils` and stored in a Pandas multiindex dataframe for convenient multi-wavelength analysis.

## Analytical Data Search in the Cloud: Finding Jets in JWST Spectral Cubes

This tutorial searches thousands of JWST spectral image cubes from MAST for Fe II emission lines associated with jets from young stellar objects.
It demonstrates how to parallelize data access, load files directly from cloud storage into memory, and perform cone searches on large target lists.
