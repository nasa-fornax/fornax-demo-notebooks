import numpy as np
from determine_source_type import determine_source_type

#function to figure out how many sources are in cutout
#and set up necessary tractor input for those sources
def find_nconfsources(raval, decval, gal_type, fluxval, x1, y1, cutout_width, subimage_wcs, df):
    
    #setup to collect sources
    objsrc = []
    
    #keep the main source
    objsrc.append(determine_source_type(raval, decval, gal_type, fluxval, x1, y1))
    
    #find confusing sources with real fluxes
    radiff = (df.ra-raval)*np.cos(decval)
    decdiff= df.dec-decval
    posdiff= np.sqrt(radiff**2+decdiff**2)*3600.
    det = df.ks_flux_aper2 > 0  #make sure they have fluxes
    
    #make an index into the dataframe for those objects within the same cutout
    good = (abs(radiff*3600.) < cutout_width/2) & (abs(decdiff*3600.) < cutout_width/2) & (posdiff > 0.2) & det
    nconfsrcs = np.size(posdiff[good])

    #add confusing sources
    #if there are any confusing sources
    if nconfsrcs > 0:
        ra_conf = df.ra[good].values
        dec_conf = df.dec[good].values
        flux_conf = df.ks_flux_aper2[good].values #should all be real fluxes
        type_conf = df.type[good].values

        for n in range(nconfsrcs):
            #now need to set the values of x1, y1 at the location of the target *in the cutout*          
            xn, yn = subimage_wcs.all_world2pix(ra_conf[n], dec_conf[n],1)
            objsrc.append(determine_source_type(ra_conf[n], dec_conf[n], type_conf[n], flux_conf[n], xn, yn))
                
            
    return objsrc, nconfsrcs
    

