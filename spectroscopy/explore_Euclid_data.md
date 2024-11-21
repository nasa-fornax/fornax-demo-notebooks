<!-- #region -->
# Explore Euclid Data
***

## Learning Goals    
By the end of this tutorial, you will be able to:

 &bull; work with data in the cloud

 &bull; get and visualize Euclid images

 &bull; get and visualize Euclid spectra

 
## Introduction
This notebook is written to be used in Fornax which is a cloud based computing platform using AWS.  It will access Euclid data stored in the cloud. The user does not need to know where the actual data are stored.  We will want to make sure the data access methods work from various locations.
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
*TODO: define this data structure (astropy table?) and what columns are interesting to keep\
*TODO: May data release will be [early release observations](https://www.cosmos.esa.int/web/euclid/ero-data-release), what fraction of this notebook can we do with that dataset, and what fraction will we need to test on other datasets 





## 3. Grab the images, 1D coadded spectrum, & individual spectra 
*TODO: Define the data structure to hold this data
 - Is the data structure here similar to the spectroscopy notebook where some rows have single band photometry and some have spectroscopy and hold arrays of flux, unc?

*TODO: How do we access the imaging?
 - images might be available via SIAv2

*TODO: How do we access the spectra?
 - From Anastasia, there will be 2 ways to access Euclid spectra
 - 1. Do a TAP search on the MER photometry catalog. The results will include a service descriptor to get the spectrum for rows of interest.
   2. Query the Euclid datalink service directly for spectra.

4/19 update: 
- IRSA datalink service is not yet ready, maybe in a couple weeks
- Alternatively, we could explore CADC
- CADC runs almost entirely off datalink - their SIA and ObsTAP services return datalink tables that you have to follow to get to any of their products.



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

```python
#### IMPORTS ###
!pip install -r requirements_euclid.txt

import os, sys
import numpy as np
import glob
from tqdm import tqdm

from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits, ascii
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.table import Table, vstack, hstack
from astropy.stats import sigma_clipped_stats
#from astropy.visualization import LogStretch, SqrtStretch
#from astropy.visualization.mpl_normalize import ImageNormalize
#astroquery.__version__

import matplotlib.pyplot as plt
import matplotlib as mpl


## Plotting stuff
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelpad'] = 7
mpl.rcParams['xtick.major.pad'] = 7
mpl.rcParams['ytick.major.pad'] = 7
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.minor.top'] = True
mpl.rcParams['xtick.minor.bottom'] = True
mpl.rcParams['ytick.minor.left'] = True
mpl.rcParams['ytick.minor.right'] = True
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
#mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['hatch.linewidth'] = 1

def_cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
```

```python
### LIST OF COORDINATES ###

## NGC 6254
#ra = 254.2923478
#dec = -4.1020057

## NGC 6397
ra = 265.1764034
dec = -53.6746141
```

```python
### SEARCH FOR IMAGES ###
## DISCLAIMER: This only works for combined images (either extended or point source stacks). This
## would not work if there are multiple, let's say, H-band images of Euclid at a given position.
## Therefore, no time domain studies here (which is anyway not one of the main goals of Euclid).

## Create coordinate object and retrieve image table from IRSA.
coord = SkyCoord(ra, dec, unit=(u.deg,u.deg), frame='icrs')
image_tab = Irsa.query_sia(pos=(coord, 1.5 * u.arcmin), collection='euclid_ero').to_table()
image_tab.sort('em_min') # sort by wavelength.
```

```python
### INSPECT TABLE ###

## Make basic selection
# For example, we only want the "Flattened" images (= compact sources stack and not extended emission stack)
#sel_basic = np.where( ["Flattened" in tt["access_url"] for tt in image_tab] )[0]
sel_basic = np.where( ["LSB" in tt["access_url"] for tt in image_tab] )[0]
image_tab = image_tab[sel_basic]

## We now inspect the table and group the images.
# We do this so complicated because we don't want np.unique() to sort the
# filters alphabetically. We want to keep the wavelength sorting that we had
# before.
idxs = np.unique(image_tab["energy_bandpassname"], return_index=True)[1]
filters = [image_tab["energy_bandpassname"][idx] for idx in sorted(idxs)]
print("Filters: {}".format(filters))

## Create dictionary with all the necessary information (science, weight, noise, mask)
summary_table = Table(names=["filter","products","facility_name","instrument_name"] , dtype=[str,str,str,str])
for filt in filters:
    sel = np.where(image_tab['energy_bandpassname'] == filt)[0]
    products = list( np.unique(image_tab["dataproduct_subtype"][sel].value) )
    if "science" in products: # sort so that science is the first in the list. This is the order we create the hdu extensions
        products.remove("science")
        products.insert(0,"science")
    else:
        print("WARNING: no science image found!")
    print(products)

    summary_table.add_row( [filt , ";".join(products), str(np.unique(image_tab["facility_name"][sel].value)[0]), str(np.unique(image_tab["instrument_name"][sel].value)[0])] )

```

```python
%%time
### CREATE CUTOUTS ###
# Now we can create the cutouts and construct the HDUs. We can create
# images for each band. If there are multiple bands, we combine them
# as a list.

for ii,filt in tqdm(enumerate(filters)):
    products = summary_table[summary_table["filter"] == filt]["products"][0].split(";")

    for product in products:
        sel = np.where( (image_tab["energy_bandpassname"] == filt) & (image_tab["dataproduct_subtype"] == product) )[0]
        ## TODO: I noted that sometimes there seem to be two images (noise) with the same access link in the table... why is that??
        ## For here, just take the first one.

        with fits.open(image_tab['access_url'][sel[0]], use_fsspec=True) as hdul:
            tmp = Cutout2D(hdul[0].section, position=coord, size=0.5 * u.arcmin, wcs=WCS(hdul[0].header)) # create cutout
            

            if (product == "science") & (ii == 0): # if science image, then create a new hdu.
                hdu0 = fits.PrimaryHDU(data = tmp.data, header=hdul[0].header)
                hdu0.header.update(tmp.wcs.to_header()) # update header with WCS info
                hdu0.header["EXTNAME"] = "{}_{}".format(filt,product.upper())
                hdu0.header["PRODUCT"] = product.upper()
                hdu0.header["FILTER"] = filt.upper()
                hdulcutout = fits.HDUList([hdu0])
            else:
                hdu = fits.ImageHDU(data = tmp.data, header=hdul[0].header)
                hdu.header.update(tmp.wcs.to_header()) # update header with WCS info
                hdu.header["EXTNAME"] = "{}_{}".format(filt,product.upper())
                hdu.header["PRODUCT"] = product.upper()
                hdu.header["FILTER"] = filt.upper()
                hdulcutout.append(hdu)

## Save the HDUL cube:
hdulcutout.writeto("./data/euclid_images_test.fits", overwrite=True)
```

```python
hdulcutout.info()
```

```python
### PLOT IMAGES ####

# what filters?
#filters = np.unique(np.asarray([hdul.header["FILTER"] for hdul in hdulcutout]))
#print(filters)

# who many images?
nimages = len(filters)
print(nimages)

# set up plot
ncols = int(4)
nrows = int( nimages // ncols )
fig = plt.figure(figsize = (5*ncols,5*nrows) )
axs = [fig.add_subplot(nrows,ncols,ii+1) for ii in range(nimages)]


# plot
for ii,filt in enumerate(filters):
    print(filters)
    img = hdulcutout["{}_SCIENCE".format(filt)].data
    axs[ii].imshow(img , origin="lower")
    axs[ii].text(0.05,0.05 , "{} ({}/{})".format(summary_table["facility_name"][ii],summary_table["instrument_name"][ii],filt) , fontsize=14 , color="white",
                 va="bottom", ha="left" , transform=axs[ii].transAxes)

plt.show()
```

```python
#### PHOTOMETRY PIPELINE ####
import sep
from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry, IterativePSFPhotometry, IntegratedGaussianPRF, make_psf_model_image
from photutils.background import LocalBackground, MMMBackground


## Definitions (need to change based on image)
psf_fwhm = 0.16 # arcsec
pixscale = 0.1 # arcsec/px
#psf_fwhm = 0.3 # arcsec
#pixscale = 0.3 # arcsec/px

## Get Data (this will be replaced later)
img = hdulcutout["VIS_SCIENCE"].data
hdr = hdulcutout["VIS_SCIENCE"].header
#img = img[100:200,100:200].copy(order='C')
img[img == 0] = np.nan

## Create mask
mask = np.isnan(img)

## Get statistics
mean, median, std = sigma_clipped_stats(img, sigma=3.0)  
print(np.array((mean, median, std))) 

## Object detection (using SEP - found that this is better compared to PHOTUTILS source detection)
objects = sep.extract(img-median, thresh=1.2, err=std, minarea=3, mask=mask, deblend_cont=0.0002, deblend_nthresh=64 )
print("Number of sources extracted: ", len(objects))

## Aperture photometry (just for testing)
flux, fluxerr, flag = sep.sum_circle(img-median, objects['x'], objects['y'],r=3.0, err=std, gain=1.0)

## Do photometry fitting (using PHOTUTILS)
init_params = Table([objects["x"],objects["y"]] , names=["x","y"]) # initial positions
psf_model = IntegratedGaussianPRF(flux=1, sigma=psf_fwhm/pixscale / 2.35)
psf_model.x_0.fixed = True
psf_model.y_0.fixed = True
psf_model.sigma.fixed = False
fit_shape = (5, 5)
psfphot = PSFPhotometry(psf_model,
                        fit_shape,
                        finder = DAOStarFinder(fwhm=0.1, threshold=3.*std, exclude_border=True), # not really needed because we are using fixed initial positions.
                        aperture_radius = 4,
                        progress_bar = True)
phot = psfphot(img-median, error=None, mask=mask, init_params=init_params)
resimage = psfphot.make_residual_image(data = img-median, psf_shape = (9, 9))
#resimage = psfphot.make_residual_image(img-median)

## Test figure
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.imshow(np.log10(img), cmap="Greys", origin="lower")
ax1.plot(phot["x_fit"], phot["y_fit"] , "s", markersize=10 , markeredgecolor="red", fillstyle="none")

ax2.imshow(resimage,vmin=-5*std, vmax=5*std, cmap="RdBu", origin="lower")
ax2.plot(phot["x_fit"], phot["y_fit"] , "s", markersize=10 , markeredgecolor="red", fillstyle="none")

plt.show()
```

```python
## TEST THE DIFFERENCE BETWEEN APERTURE AND PSF PHOTOMETRY ##
x = objects["flux"]
y = phot["flux_fit"]

plt.plot(x , y , "o", markersize=2)
minlim = np.nanmin(np.concatenate((x,y)))
maxlim = np.nanmax(np.concatenate((x,y)))

plt.fill_between(np.asarray([minlim,maxlim]),np.asarray([minlim,maxlim])/1.5,np.asarray([minlim,maxlim])*1.5, color="gray", alpha=0.2, linewidth=0)
plt.fill_between(np.asarray([minlim,maxlim]),np.asarray([minlim,maxlim])/1.2,np.asarray([minlim,maxlim])*1.2, color="gray", alpha=0.4, linewidth=0)
plt.plot(np.asarray([minlim,maxlim]),np.asarray([minlim,maxlim]), ":", color="gray")

plt.xlabel("Aperture Photometry [flux]")
plt.ylabel("PSF forced-photometry [flux]")
plt.xscale('log')
plt.yscale('log')
plt.show()
```

```python
## FORCE PHOTOMETER THE NISP IMAGE ##
## Use the position priors from the VIS image. Need to convert to NISP x/y using the WCS.

## Definitions (need to change based on image)
#psf_fwhm = 0.16 # arcsec
#pixscale = 0.1 # arcsec/px
psf_fwhm = 0.3 # arcsec
pixscale = 0.3 # arcsec/px

## Get Data (this will be replaced later)
img2 = hdulcutout["Y_SCIENCE"].data
hdr2 = hdulcutout["Y_SCIENCE"].header
img2[img2 == 0] = np.nan

## Get statistics (not really needed)
mean2, median2, std2 = sigma_clipped_stats(img2, sigma=3.0)  
print(np.array((mean2, median2, std2))) 

## Create mask
mask2 = np.isnan(img2)

## Get prior positions
wcs = WCS(hdr) # VIS
wcs2 = WCS(hdr2) # NISP
radec = wcs.all_pix2world(objects["x"],objects["y"], 0)
xy = wcs2.all_world2pix(radec[0],radec[1],0)


## Do photometry fitting (using PHOTUTILS using VIS positions!)
init_params = Table([xy[0],xy[1]] , names=["x","y"]) # initial positions
psf_model = IntegratedGaussianPRF(flux=1, sigma=psf_fwhm/pixscale / 2.35)
psf_model.x_0.fixed = True
psf_model.y_0.fixed = True
psf_model.sigma.fixed = False
fit_shape = (3, 3)
psfphot2 = PSFPhotometry(psf_model,
                        fit_shape,
                        finder = DAOStarFinder(fwhm=0.1, threshold=3.*std2, exclude_border=True), # not really needed because we are using fixed initial positions.
                        aperture_radius = 4,
                        progress_bar = True)
phot2 = psfphot2(img2-median2, error=None, mask=mask2, init_params=init_params)
resimage2 = psfphot2.make_residual_image(data = img2-median2, psf_shape = (3, 3))


## Test figure
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.imshow(np.log10(img2), cmap="Greys", origin="lower")
ax1.plot(phot2["x_fit"], phot2["y_fit"] , "s", markersize=10 , markeredgecolor="red", fillstyle="none")

ax2.imshow(resimage2,vmin=-20*std2, vmax=20*std2, cmap="RdBu", origin="lower")
ax2.plot(phot2["x_fit"], phot2["y_fit"] , "s", markersize=10 , markeredgecolor="red", fillstyle="none")

plt.show()
```

```python
END
```

```python
### Catalogs ###

#Irsa.list_catalogs()
```
