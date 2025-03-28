import numpy as np
from acstools import acszpt
from astropy.time import Time


def convert_wise_flux_to_millijansky(flux_nanomaggy, *, band=None):
    """unWISE light curves flux is stored in nanomaggy. Convert to millijansky.

    See https://www.sdss3.org/dr8/algorithms/magnitudes.php and Meisner et al. (2023, https://iopscience.iop.org/article/10.3847/1538-3881/aca2ab/pdf).
    For Vega to AB conversion for WISE see https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html

    Parameters
    ----------
    flux_nanomaggy : float or iterable of floats
        Flux in nanomaggy.
    band : str
        WISE band name corresponding to flux_nanomaggy. One of "W1", "W2", "W3", "W4", or None.
        If None, band will be set to the name attribute of flux_nanomaggy. Useful when applying this
        transform to a DataFrame grouped by band.

    Returns
    -------
    flux_mjy : float or iterable of floats
        flux_nanomaggy converted to millijansky.
    """
    if band is None:
        band = flux_nanomaggy.name

    # Vega to AB magnitude conversions
    vega_to_ab_conv = {"W1": 2.699, "W2": 3.339, "W3": 5.174, "W4": 6.620}

    # get Vega magnitude from nanomaggy flux as described in Meisner et al. (2023)
    mag_vega = 22.5 - 2.5*np.log10(flux_nanomaggy)

    # convert Vega magnitude to AB magnitude
    mag_ab = mag_vega + vega_to_ab_conv[band]

    # convert AB magnitude to mJy
    flux_mjy = 10**(-0.4*(mag_ab - 23.9)) / 1e3

    return flux_mjy


def convertACSmagtoflux(date, filterstring, mag, magerr):
    """converts HST ACS magnitudes into flux units of Janskies

    Parameters
    ----------
    date : float
        date of observation in units of MJD
    filterstring : str {'F435W', 'F475W','F502N','F550M','F555W','F606W','F625W',/
    'F658N','F660N','F775W','F814W','F850LP','F892N'}
        name of the ACS band to be converted
    mag : array-like
        array of ACS magnitudes
    magerr : array-like
        array of ACS uncertainties on the magnitudes

    Returns
    -------
    flux: array
        flux in janskies corresponding to the input magnitudes
    flux uncertaintiy: array
        uncertainty on the returned flux corresponding to input magerr
    """

    # date is currently in MJD and needs to be in ISO Format (YYYY-MM-DD)
    # use astropy to handle this properly
    t = Time(date, format='mjd')

    # t.iso has this, but also the hrs, min, sec, so need to truncate those
    tiso = t.iso[0:10]
    q = acszpt.Query(date=tiso, detector='WFC', filt=filterstring)
    zpt_table = q.fetch()
    print(zpt_table)
    zpt = zpt_table['VEGAmag'].value

    # ACS provides conversion to erg/s/cm^2/angstrom
    flux = 10**((mag - zpt)/(-2.5))  # now in erg/s/cm^2/angstrom
    # calculate the error
    magupper = mag + magerr
    maglower = mag - magerr
    flux_upper = abs(flux - (10**((magupper - zpt)/(-2.5))))
    flux_lower = abs(flux - (10**((maglower - zpt)/(-2.5))))

    fluxerr = (flux_upper + flux_lower) / 2.0

    # now convert from erg/s/cm^2/angstrom to Jy
    # first convert angstrom to Hz using c
    c = 3E8
    flux = flux * c / (1E-10)  # now in erg/s/cm^2/Hz
    fluxerr = fluxerr * c / (1E-10)  # now in erg/s/cm^2/Hz

    flux = flux * (1E-23)  # now in Jy

    fluxerr = fluxerr * (1E-23)  # now in Jy

    return flux, fluxerr


def mjd_to_jd(mjd):
    """
    Convert Modified Julian Day to Julian Day.

    Parameters
    ----------
    mjd : float
        Modified Julian Day

    Returns
    -------
    jd : float
        Julian Day


    """
    return mjd + 2400000.5
