<!-- #region -->
# Explore Euclid Data
***

## Learning Goals    
By the end of this tutorial, you will be able to:

 &bull; work with data in the cloud

 &bull; download and visualize Euclid images

 &bull; download and visualize Euclid spectra

 
## Introduction
<!-- #endregion -->

## 1.  Define the Sample

Start with the MOSDEF sample Kriek et al 2015
- We think these galaxies will be included the May 2024 early data release
- Redshift range of this sample is good match to Euclid

*TODO: need coords of this sample
-  https://mosdef.astro.berkeley.edu/for-scientists/data-releases/ links arent working
-  Kriek et al., 2015 doesn't have it
-  Du et al., 2021 coords are on Vizier.  Is this an ok sample to start with?  
   - 157 Star-forming Galaxies at z~2, some fraction of which are in COSMOS




## 2. Query the Euclid catalog for the sample
MER is the merged catalog of Euclid imaging & spectroscopy as well as external photometry sources

Return a data structure of targets with Euclid data

*TODO: how do we access MER?\
*TODO: define this data structure (astropy table?) and what columns are interesting to keep




## 3. Grab the images, 1D coadded spectrum, & individual spectra 
*TODO: Define the data structure to hold this data
 - Is the data structure here similar to the spectroscopy notebook where some rows have single band photometry and some have spectroscopy and hold arrays of flux, unc?

*TODO: How do we access the imaging?
 - images might be available via SIAv2

*TODO: How do we access the spectra?
 - spectra will be available as datalinks from the MER catalog


## 4. Visualize Euclid data

```python
# plot object-scale cutouts of the MER images

```

```python
# plot coadded 1D spectrum
```

```python
# plot individual dithers to get a sense of the quality of the data
```

```python

```
