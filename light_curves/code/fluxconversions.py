import numpy as np
from acstools import acszpt
from astropy.time import Time


def convert_wise_flux_to_mag(flux, dflux):
    """Convert WISE fluxes to magnitudes.

    This follows the conversions done for the original Meisner et al., 2023 catalog
    (see https://github.com/fkiwy/unTimely_Catalog_explorer/blob/main/unTimely_Catalog_tools.py),
    except that here we support vectorization.
    """
    def calculate_magnitude(myflux):
        # need to allow myflux to be an array, so use np.log10 not math.log10
        mymag = 22.5 - 2.5 * np.log10(myflux)
        # if myflux < 0, np.log10(myflux) == nan
        # if myflux = 0, np.log10(myflux) == -inf. replace with nan to match unTimely_Catalog_tools
        mymag[np.isinf(mymag)] = np.nan
        return mymag

    mag = calculate_magnitude(flux)
    mag_upper = calculate_magnitude(flux - dflux)
    mag_lower = calculate_magnitude(flux + dflux)
    dmag = (mag_upper - mag_lower) / 2

    return mag, dmag


#need to convert those magnitudes into mJy to be consistent in data structure.
#using zeropoints from here: https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
def convert_WISEtoJanskies(mag, magerr, band):
    """converts WISE telescope magnitudes into flux units of Jansies in mJy
    
    Parameters
    ----------
    mag : array-like
        array of WISE magnitudes
    magerr : array-like
        array of WISE uncertainties on the magnitudes
    band : str {'w1', 'w2'}
        name of the WISE band to be converted
        
    Returns
    -------
    flux: array
        flux in mJy corresponding to the input magnitudes
    flux uncertaintiy: array
        uncertainty on the returned flux corresponding to input magerr
    """
    if band == 'w1':
        zpt = 309.54
    elif band == 'w2':
        zpt = 171.787
            
    flux_Jy = zpt*(10**(-mag/2.5))
    
    #calculate the error
    magupper = mag + magerr
    maglower = mag - magerr
    flux_upper = abs(flux_Jy - (zpt*(10**(-magupper/2.5))))
    flux_lower = abs(flux_Jy - (zpt*(10**(-maglower/2.5))))
    
    fluxerr_Jy = (flux_upper + flux_lower) / 2.0
    
    return flux_Jy*1E3, fluxerr_Jy*1E3  #now in mJy

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
            