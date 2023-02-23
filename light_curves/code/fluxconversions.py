#need to convert those magnitudes into mJy to be consistent in data structure.
#using zeropoints from here: https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
def convert_WISEtoJanskies(mag, magerr, band):
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
            