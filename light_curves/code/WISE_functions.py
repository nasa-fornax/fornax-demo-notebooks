from tqdm import tqdm
import pandas as pd
from astropy.coordinates import SkyCoord
from unTimely_Catalog_tools import unTimelyCatalogExplorer
import os

from .data_structures import MultiIndexDFObject
from .fluxconversions import convert_WISEtoJanskies


#WISE
def WISE_get_lightcurves(coords_list, labels_list, radius, bandlist):
    """Searches WISE catalog from meisner et al., 2023 for light curves from a list of input coordinates
    
    Parameters
    ----------
    coords_list : list of astropy skycoords
        the coordinates of the targets for which a user wants light curves
    labels_list: list of strings
        journal articles associated with the target coordinates
    radius : float
        search radius, how far from the source should the archives return results
    bandlist: list of strings
        which WISE bands to search for, example: ['w1', 'w2']
        
    Returns
    -------
    df_lc : pandas dataframe
        the main data structure to store all light curves
    """

    #setup the explorer
    ucx = unTimelyCatalogExplorer(directory=os.getcwd(), cache=True, show_progress=True, timeout=300,
                                  catalog_base_url='http://unwise.me/data/neo7/untimely-catalog/',
                                  catalog_index_file='untimely_index-neo7.fits')

    df_lc = MultiIndexDFObject()
    #for ccount in range(1):#enumerate(coords_list):
    for ccount, coord in enumerate(tqdm(coords_list)):
        #doesn't take SkyCoord, convert to floats
        ra = coord.ra.deg 
        dec = coord.dec.deg 
        lab = labels_list[ccount]
        try:
            #search the untimely catalog
            result_table = ucx.search_by_coordinates(ra, dec, box_size=100, cone_radius=radius,
                                                     show_result_table_in_browser=False, save_result_table=False)#, suppress_console_output=True)

            if (len(result_table) > 0):
                #got a live one
                for bcount, band in enumerate(bandlist):
                    #sort by band
                    mask = result_table['band'] == (bcount + 1)
                    result_table_band = result_table[mask]
    
                    mag = result_table_band['mag']
                    magerr = result_table_band['dmag']
    
                    wiseflux, wisefluxerr = convert_WISEtoJanskies(mag,magerr ,band)

                    time_mjd = result_table_band['mjdmean']
            
                    #plt.figure(figsize=(8, 4))
                    #plt.errorbar(time_mjd, wiseflux, wisefluxerr)
                    dfsingle = pd.DataFrame(dict(flux=wiseflux, err=wisefluxerr, time=time_mjd, objectid=ccount + 1, band=band,label=lab)).set_index(["objectid","label", "band", "time"])

                    #then concatenate each individual df together
                    df_lc.append(dfsingle)
            else:        
                print("There is no WISE light curve for this object")
            
        except:
            pass
        
    return(df_lc)

