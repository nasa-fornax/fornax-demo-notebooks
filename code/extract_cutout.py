##function to extract cutout image
import math
import numpy as np
import os
from astropy.nddata import Cutout2D
import astropy.wcs as wcs
import astropy.io.fits as fits


def extract_cutout(ra, dec, cutout_width, mosaic_pix_scale, hdulist, wcs_info):
    '''
    outfile: cutout fits file of source
    ra: RA of source being modeled  
    dec: DEC of source being modeled  
    cutout_width: desired width of cutout in arcseconds
    '''
    try:
 
        #convert ra and dec into x, y
        x0, y0 = wcs_info.all_world2pix(ra, dec,1)
        position=(x0, y0)
        #subimage = hdulist[0].data

        #make size array in pixels
        #how many pixels are in cutout_width
        size = (cutout_width / mosaic_pix_scale)
        size = int(math.ceil(size)) #round up the nearest integer

        #make the cutout
        cutout = Cutout2D(hdulist.data, position, size, copy = True, mode = "trim", wcs = wcs_info)
        subimage = cutout.data.copy()
        subimage_wcs = cutout.wcs.copy()

        #now need to set the values of x1, y1 at the location of the target *in the cutout*
        x1, y1 = subimage_wcs.all_world2pix(ra, dec,1)
        #print('x1, y1', x1, y1)

        #hdulist.close()
        nodata_param = False

    except:
        nodata_param = True
        subimage = np.empty([10,10])
        subimage[:] = 0.0
        x1= 5
        y1= 5
        subimage_wcs = "extract cutout didn't work"
            
    return subimage.data, nodata_param, x1, y1, subimage_wcs


                                                                                                    
