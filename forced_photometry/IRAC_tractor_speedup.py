#!/usr/bin/env python
"""
IRAC forced photometry following Kristina Nylands script using tractor.

modified notebook for speedup
"""

import math
import time
import warnings
import concurrent.futures

import sys
import os
from contextlib import contextmanager

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
import astropy.wcs as wcs
import astropy.io.fits as fits

from reproject import reproject_interp

from skimage.transform import rotate

from tractor import (Tractor, PointSource, PixPos, Flux, PixelizedPSF, NullWCS,
                     NullPhotoCal, ConstantSky, Image)

# set up clean catalog with fiducial band fluxes, ra, dec, shape parameters,
# probability that it is a star


# read in the catalog I generated from IRSA website
# COSMOS 2015 with only some of the columns and a few rows
# Type: 0 = galaxy, 1 = star, 2 = X-ray source
# I think the center of this catalog is roughy 149.955, 3.5375
df = pd.read_csv('table_irsa_catalog_search_results.csv')

# set default cutout width = 10"
cutout_width = 10

ra_0 = df.ra[0]
dec_0 = df.dec[0]


# are there missing values
print('Number of missing values:')
print(df.isna().sum())
print('')

# We don't mind that there are missing values for IRAC fluxes or for photzs.
# The rest of the rows are complete


# out of curiosity how many stars vs. galaxies vs. x ray sources
print('Numbers of stars, galaxies and x-ray sources:')
print(df.type.value_counts())
print('')


# initialize columns in data frame for photometry results
df[["ch1flux", "ch1flux_unc", "ch2flux", "ch2flux_unc", "ch3flux",
    "ch3flux_unc", "ch4flux", "ch4flux_unc"]] = np.nan


# function to determine what type of source it is from catalog
def determine_source_type(ra, dec, df_type, fid_flux, x1, y1):
    """Determine source type."""
    # make all sources point sources for now
    # use fiducial flux as first guess of source flux in different bands

    src = PointSource(PixPos(x1, y1), Flux(fid_flux))
    return src


# function to extract cutout image
def extract_cutout(infile, ra, dec, cutout_width, mosaic_pix_scale):
    """
    Extract cutout.

    infile: mosaic containing catalog source
    outfile: cutout fits file of source
    ra: RA of source being modeled
    dec: DEC of source being modeled
    cutout_width: desired width of cutout in arcseconds
    """
    if not os.path.isfile(infile):
        error_message = ('ERROR: FITS FILE {} NOT FOUND - ABORTING SCRIPT'
                         .format(infile))
        sys.exit(error_message)
    else:
        try:
            # open the mosaic file
            hdulist = fits.open(infile)[0]
            hdr = hdulist.header
            wcs_info = wcs.WCS(hdulist)

            # convert ra and dec into x, y
            x0, y0 = wcs_info.all_world2pix(ra, dec, 1)
            position = (x0, y0)
            # subimage = hdulist[0].data

            # make size array in pixels
            # how many pixels are in cutout_width
            size = (cutout_width / mosaic_pix_scale)
            size = int(math.ceil(size))  # round up the nearest integer

            # make the cutout
            cutout = Cutout2D(hdulist.data, position, size, copy=True,
                              mode="trim", wcs=wcs_info)
            subimage = cutout.data.copy()
            subimage_wcs = cutout.wcs.copy()

            # now need to set the values of x1, y1 at the location of the
            # target *in the cutout*
            x1, y1 = subimage_wcs.all_world2pix(ra, dec, 1)
            # print('x1, y1', x1, y1)

            # hdulist.close()
            nodata_param = False

        except:
            nodata_param = True
            subimage = np.empty([10, 10])
            subimage[:] = 0.0
            x1 = 5
            y1 = 5
            subimage_wcs = "extract cutout didn't work"
    return subimage.data, hdr, nodata_param, x1, y1, subimage_wcs


# function to normalize the PRF with the same pixel scale as the mosaic images
# tractor expects this, and will give bad results if the pixel scale or
# normalization is anything else
def prepare_prf(prf_fitsname, ra_0, dec_0, rotate_angle):
    """Prepare the PRF."""
    # read in the PRF fits file
    ext_prf = fits.open(prf_fitsname)[0]

    # fix a type on the header that crashes reproject_interp
    ext_prf.header['CTYPE1'] = 'RA---TAN'

    # ok, need to fake it and make the ra and dec of the center of the prf
    # be the same as the center of the cutout
    # just using a random cutout here to make this work since we need an image
    # for reproject_interp to work
    ext_prf.header['CRVAL1'] = ra_0
    ext_prf.header['CRVAL2'] = dec_0

    cutout = (fits.open(
        '0001_149.96582000_2.53160000_irac_ch1_go2_sci_10.fits')[0])

    prf_resample, footprint = reproject_interp(ext_prf, cutout.header)

    # ugg, ok, and check if it is an odd size
    # tractor crashes if the PRF has an even number of pixels
    if (len(prf_resample.data) % 2) < 1:
        prf_resample = Cutout2D(prf_resample, (9, 9), (17, 17))

        # and because cutout2D changes data types
        prf_resample = prf_resample.data

    # renormalize the PRF so that the sum of all pixels = 1.0
    # tractor gives anomolous results if the PRF is normalized any other way
    prf_resample_norm = prf_resample / prf_resample.sum()

    # looks like a rotation might help
    # still working to figure this out, but setting up to let it happen here
    prf_resample_norm_rotate = rotate(prf_resample_norm, rotate_angle)

    return prf_resample_norm_rotate


# function to figure out how many sources are in cutout
# and set up necessary tractor input for those sources
def find_nconfsources(raval, decval, gal_type, fluxval, x1, y1, cutout_width,
                      subimage_wcs):
    """Find number of confusing sources."""
    # setup to collect sources
    objsrc = []

    # keep the main source
    objsrc.append(determine_source_type(raval, decval, gal_type, fluxval,
                                        x1, y1))

    # find confusing sources with real fluxes
    radiff = (df.ra - raval) * np.cos(decval)
    decdiff = df.dec - decval
    posdiff = np.sqrt(radiff**2 + decdiff**2) * 3600.
    det = df.ks_flux_aper2 > 0  # make sure they have fluxes

    # make an index into the dataframe for those objects within the same cutout
    good = ((abs(radiff * 3600.) < cutout_width / 2) &
            (abs(decdiff * 3600.) < cutout_width / 2) &
            (posdiff > 0.2) &
            det)
    nconfsrcs = np.size(posdiff[good])

    # add confusing sources
    # if there are any confusing sources
    if nconfsrcs > 0:
        ra_conf = df.ra[good].values
        dec_conf = df.dec[good].values
        flux_conf = df.ks_flux_aper2[good].values  # should all be real fluxes
        type_conf = df.type[good].values

        for n in range(nconfsrcs):
            # now need to set the values of x1, y1 at the location
            # of the target *in the cutout*
            xn, yn = subimage_wcs.all_world2pix(ra_conf[n], dec_conf[n], 1)
            objsrc.append(determine_source_type(ra_conf[n], dec_conf[n],
                                                type_conf[n], flux_conf[n],
                                                xn, yn))

    return objsrc, nconfsrcs


# setup to supress output of tractor
# seems to be the only way to not output every step of optimization
# https://stackoverflow.com/questions/2125702/
#         how-to-suppress-console-output-in-python

@contextmanager
def suppress_stdout():
    """Suppress standard output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# function to display original, model, and residual images
# for individual targets
def display_images(mod, chi, subimage):
    """Display images."""
    # make the residual image
    diff = subimage - mod

    # setup plotting
    fig = plt.figure(figsize=(7, 2))

    ax1 = fig.add_subplot(131, autoscale_on=False, xlim=(0, 17), ylim=(0, 17))
    ax2 = fig.add_subplot(132, autoscale_on=False, xlim=(0, 17), ylim=(0, 17))
    ax3 = fig.add_subplot(133, autoscale_on=False, xlim=(0, 17), ylim=(0, 17))

    ax1.set(xticks=[], yticks=[])
    ax2.set(xticks=[], yticks=[])
    ax3.set(xticks=[], yticks=[])

    # display the images
    im1 = ax1.imshow(subimage, cmap='gray')  # , vmin = 0.01, vmax = 0.20
    ax1.set_title('Original Image')
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(mod, cmap='gray')  # , vmin = 0.01, vmax = 0.20
    ax2.set_title('Model')
    fig.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(diff, cmap='gray')  # , vmin = 0.01, vmax = 0.20
    ax3.set_title('Residual')
    fig.colorbar(im3, ax=ax3)

    return

# calling sequence display_images(tractor.getModelImage(0),
#   tractor.getChiImage(0), subimage )


# function to display an SED
def plot_sed(obj):
    """Plot a spectral energy distribution."""
    # make super simple plot
    wavelength = [3.6, 4.5, 5.8, 8.0]
    # FUV ~1500Angstroms, NUV ~2300Angstroms or 0.15 & 0.23 microns
    flux = [df.ch1flux[obj], df.ch2flux[obj], df.ch3flux[obj],
            df.ch4flux[obj]]
    fluxerr = [df.ch1flux_unc[obj], df.ch2flux_unc[obj], df.ch3flux_unc[obj],
               df.ch4flux_unc[obj]]

    # fudge the uncertainties higher until the uncertainty function is working
    fluxerr = [i * 5 for i in fluxerr]

    # plot as log wavenlength vs. log flux to eventually include Galex
    fig, ax = plt.subplots()
    ax.set_xscale("log", nonpositive='clip')
    ax.set_yscale("log", nonpositive='clip')
    ax.errorbar(wavelength, flux, yerr=fluxerr)

    # set up labels
    ax.set(xlabel='Wavelength (microns)', ylabel="Flux ($\mu$Jy)", title='SED')
    plt.show()

    return

# Set up variables for the next function to use
rotate_angle = 0

irac_fluxconversion = (1E12) / (4.254517E10) * (0.6) * (0.6)
# convert tractor result to microjanskies
flux_conv = irac_fluxconversion
mosaic_pix_scale = 0.6

# set up prfs for each channel
prfs = [prepare_prf('IRAC.1.EXTPRF.5X.fits', ra_0, dec_0, rotate_angle),
        prepare_prf('IRAC.2.EXTPRF.5X.fits', ra_0, dec_0, rotate_angle),
        prepare_prf('IRAC.3.EXTPRF.5X.fits', ra_0, dec_0, rotate_angle),
        prepare_prf('IRAC.4.EXTPRF.5X.fits', ra_0, dec_0, rotate_angle)]

# set up mosaics for each channel
infiles = ['COSMOS_irac_ch1_mosaic_test.fits',
           'COSMOS_irac_ch2_mosaic_test.fits',
           'COSMOS_irac_ch3_mosaic_test.fits',
           'COSMOS_irac_ch4_mosaic_test.fits']


def calc_instrflux(band, ra, dec, stype, ks_flux_aper2):
    """
    Calculate instrumental fluxes and uncertainties for four IRAC bands.

    Parameters:
    -----------
    index: pandas Index
        Index to the dataframe
    band: int
        integer in [0, 1, 2, 3] for the four IRAC bands
    ra, dec: float or double
        celestial coordinates for measuring photometry
    stype: int
        0, 1, 2 for star, galaxy, x-ray source
    ks_flux_aper_2: float
        flux in aperture 2

    Returns:
    --------
    outband: int
        reflects input band for identification purposes
    flux: float
        measured flux in microJansky, NaN if unmeasurable
    unc: float
        measured uncertainty in microJansky, NaN if not able to estimate
    """
    prf = prfs[band]
    infile = infiles[band]

    # make a cutout
    subimage, hdr, nodata_param, x1, y1, subimage_wcs = extract_cutout(
        infile, ra, dec, cutout_width, mosaic_pix_scale)

    # catch errors in making the cutouts
    if not nodata_param:  # meaning we have data in the cutout

        # set up the source list
        # src = determine_source_type(df.ra[i], df.dec[i], df.type[i],
        # df.ks_flux_aper2[i], x1,y1)
        objsrc, nconfsrcs = find_nconfsources(ra, dec, stype,
                                              ks_flux_aper2, x1, y1,
                                              cutout_width, subimage_wcs)

        # measure sky noise and mean level
        skymean, skymedian, skynoise = sigma_clipped_stats(subimage, sigma=3.0)

        # make the tractor image
        tim = Image(data=subimage, invvar=np.ones_like(subimage) / skynoise**2,
                    psf=PixelizedPSF(prf),
                    wcs=NullWCS(), photocal=NullPhotoCal(),
                    sky=ConstantSky(skymean))

        # make tractor object
        tractor = Tractor([tim], objsrc)  # [src]

        # freeze the parameters we don't want tractor fitting
        tractor.freezeParam('images')  # now fits 2 positions and flux
        # only fit for flux
        # src.freezeAllRecursive()
        # src.thawPathsTo('brightness')

        # run the tractor optimization (do forced photometry)
        # Take several linearized least squares steps
        fit_fail = False
        try:
            tr = 0
            with suppress_stdout():
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', '.*divide by zero.*')
                    # warnings.simplefilter('ignore')
                    for tr in range(20):
                        dlnp, X, alpha, flux_var = (
                            tractor.optimize(variance=True))
                        # print('dlnp',dlnp)
                        if dlnp < 1e-3:
                            break
        # catch exceptions and bad fits
        except:
            fit_fail = True

        # record the photometry results
        if fit_fail:
            # tractor fit failed
            # set flux and uncertainty as nan and move on
            return(band, np.nan, np.nan)
        elif flux_var is None:
            # fit worked, but flux variance did not get reported
            params_list = objsrc[0].getParamNames()
            bindex = params_list.index('brightness.Flux')
            flux = objsrc[0].getParams()[bindex]
            # convert to microjanskies
            microjy_flux = flux * flux_conv
            return(band, microjy_flux, np.nan)
        else:
            # fit and variance worked
            params_list = objsrc[0].getParamNames()
            bindex = params_list.index('brightness.Flux')
            flux = objsrc[0].getParams()[bindex]

            # determine flux uncertainty
            # which value of flux_var is for the flux variance?
            # Assumes we are fitting positions and flux
            fv = ((nconfsrcs + 1) * 3) - 1
            tractor_std = np.sqrt(flux_var[fv])

            # convert to microjanskies
            microjy_flux = flux * flux_conv
            microjy_unc = tractor_std * flux_conv
            return(band, microjy_flux, microjy_unc)

    else:
        return(band, np.nan, np.nan)

# Save a copy of the dataframe for the parallel calculation
pl_df = df.copy(deep=True)


# Run it on everything

t0 = time.time()

for row in df.itertuples():
    for band in range(4):
            outband, flux, unc = calc_instrflux(band, row.ra, row.dec,
                                                row.type, row.ks_flux_aper2)
            df.loc[row.Index, 'ch{:d}flux'.format(outband + 1)] = flux
            df.loc[row.Index, 'ch{:d}flux_unc'.format(outband + 1)] = unc

t1 = time.time()

print('It took {:.2f} seconds to run all sources'.format((t1 - t0)))

# how many rows did get filled in?  = 225
print('Serial calculation: number of ch1 fluxes filled in',
      np.sum(df.ch1flux > 0))

# Parallelization: we can either interate over the rows of the dataframe and
# run the four bands in parallel; or we could zip together the row index,
# band, ra, dec, type, ks_flux_aper2

paramlist = []
for row in df.itertuples():
    for band in range(4):
        paramlist.append([row.Index, band, row.ra, row.dec, row.type,
                          row.ks_flux_aper2])


print('length of paramlist is', len(paramlist))


print('tests of first calculation:')
print(calc_instrflux(paramlist[0][1], paramlist[0][2],
                     paramlist[0][3], paramlist[0][4], paramlist[0][5]))


print(calc_instrflux(*paramlist[0][1:]))


def calculate_flux(args):
    """Calculate flux."""
    f = calc_instrflux
    val = f(*args[1:])
    return(args[0], val)

t2 = time.time()
outputs = []
with concurrent.futures.ProcessPoolExecutor(24) as executor:
    for result in executor.map(calculate_flux, paramlist):
        # print(result)
        pl_df.loc[result[0],
                  'ch{:d}flux'.format(result[1][0] + 1)] = result[1][1]
        pl_df.loc[result[0],
                  'ch{:d}flux_unc'.format(result[1][0] + 1)] = result[1][1]
        outputs.append(result)
t3 = time.time()

print(outputs[0])
print('')


print('Parallel calculation took {:.2f} seconds'.format((t3 - t2)))
print('Speedup is {:.2f}'.format((t1 - t0) / (t3 - t2)))
print('Parallel calculation: number of ch1 fluxes filled in',
      np.sum(pl_df.ch1flux > 0))
