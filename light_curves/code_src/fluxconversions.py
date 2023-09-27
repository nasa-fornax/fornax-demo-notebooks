from acstools import acszpt
from astropy.time import Time


def convert_wise_flux_to_millijansky(nanomaggy_flux):
    """unWISE light curves flux is stored in nanomaggy. Convert to millijansky.
    
    See https://www.sdss3.org/dr8/algorithms/magnitudes.php
    """
    millijansky_per_nanomaggy = 3.631e-3
    return nanomaggy_flux * millijansky_per_nanomaggy


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
 
    #date is currently in MJD and needs to be in ISO Format (YYYY-MM-DD)
    #use astropy to handle this properly
    t = Time(date, format = 'mjd')
    
    #t.iso has this, but also the hrs, min, sec, so need to truncate those
    tiso = t.iso[0:10]
    q = acszpt.Query(date=tiso, detector='WFC', filt=filterstring)
    zpt_table = q.fetch()
    print(zpt_table)
    zpt = zpt_table['VEGAmag'].value
    
    #ACS provides conversion to erg/s/cm^2/angstrom
    flux = 10**((mag - zpt)/(-2.5))  #now in erg/s/cm^2/angstrom
    #calculate the error
    magupper = mag + magerr
    maglower = mag - magerr
    flux_upper = abs(flux - (10**((magupper - zpt)/(-2.5))))
    flux_lower =  abs(flux - (10**((maglower - zpt)/(-2.5))))
    
    fluxerr = (flux_upper + flux_lower) / 2.0

    #now convert from erg/s/cm^2/angstrom to Jy
    #first convert angstrom to Hz using c
    c = 3E8
    flux = flux * c /(1E-10) #now in erg/s/cm^2/Hz
    fluxerr = fluxerr * c /(1E-10) #now in erg/s/cm^2/Hz

    flux = flux * (1E-23) # now in Jy
    
    fluxerr = fluxerr * (1E-23) # now in Jy

    return flux, fluxerr
            