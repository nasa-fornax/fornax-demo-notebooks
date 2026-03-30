# Data - Forced Photometry

This directory holds data downloaded or generated during the forced-photometry workflow. Users may place additional data here as needed. Output produced by the notebook is stored in a separate directory called `../output`.

Instrument-specific PRF/PSF files are also stored here; these are required by the code and should not be removed.

Two pre-generated files are included for convenience:

- COSMOS_chandra.fits – A formatted Chandra catalog (RA/Dec, uncertainties, and basic X-ray fluxes) used for nway probabilistic cross-matching.

- multiband_phot.fits – The Tractor-derived multiband photometry catalog used as input for the cross-matching and final color–color analysis.


No new data should be committed to the repository. This directory is listed in .gitignore to prevent accidental commits.
