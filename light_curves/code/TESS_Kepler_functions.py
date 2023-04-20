from tqdm import tqdm
import pandas as pd
from astropy.coordinates import SkyCoord
import lightkurve as lk

from clean_filternames import clean_filternames
from data_structures import MultiIndexDFObject


def TESS_Kepler_get_lightcurves(coords_list, labels_list, radius):
    """Searches TESS, Kepler, and K2 for light curves from a list of input coordinates
    
    Parameters
    ----------
    coords_list : list of astropy skycoords
        the coordinates of the targets for which a user wants light curves
    labels_list: list of strings
        journal articles associated with the target coordinates
    radius : float
        search radius, how far from the source should the archives return results

    Returns
    -------
    df_lc : pandas dataframe
        the main data structure to store all light curves
    """
    
    df_lc = MultiIndexDFObject()
    #for all objects
    for ccount, coord in enumerate(tqdm(coords_list)):
    #for testing, this has 79 light curves between the three missions.
    #for ccount in range(1):
    #    coord = '19:02:43.1 +50:14:28.7'
        try:
        #use lightkurve to search TESS, Kepler and K2
            search_result = lk.search_lightcurve(coord, radius = radius)
            lab = labels_list[ccount]

            #figure out what to do with the results
            if len(search_result) >= 1:
                #https://docs.lightkurve.org/tutorials/1-getting-started/searching-for-data-products.html
                print(ccount, 'got a live one')
                #download all of the returned light curves from TESS, Kepler, and K2
                lc_collection = search_result.download_all()

                #can't get the whole collection directly into pandas multiindex
                #pull out inidividual light curves, convert to uniform units, and put them in pandas
                for numlc in range(len(search_result)):
            
                    lc = lc_collection[numlc]  #for testing 0 is Kepler, #69 is TESS

                    #convert to Pandas
                    lcdf = lc.to_pandas().reset_index()
        
                    #these light curves are too highly sampled for our AGN use case, so reduce their size
                    #by choosing only to keep every nth sample
                    nsample = 30
                    lcdf_small = lcdf[lcdf.index % nsample ==0]  #selects every nth row starting with row 0
            
                    #convert time to mjd
                    time_lc = lcdf_small.time #in units of time - 2457000 BTJD days
                    time_lc= time_lc + 2457000 - 2400000.5 #now in MJD days within a few minutes (except for the barycenter correction)

                    #TESS, Kepler, and K2 report flux in units of electrons/s 
                     #- there is no good way to convert this to anything more useful because the bandpasses are very wide and nonstandard
                     #- really we don't care about absolute scale, but want to scale the light curve to be on the same plot as other light curves
                     #- save as electron/s here and scale when plotting
                    flux_lc = lcdf_small.flux #in electron/s
                    fluxerr_lc = lcdf_small.flux_err #in electron/s

                    #record band name
                    filtername = clean_filternames(search_result, numlc)

                    #put this single object light curves into a pandas multiindex dataframe
                    # fluxes are in units of electrons/s and will be scaled to fit the other fluxes when plotting
                    dfsingle = pd.DataFrame(dict(flux=flux_lc, err=fluxerr_lc, time=time_lc, objectid=ccount + 1, band=filtername,label=lab)).set_index(["objectid", "label", "band", "time"])
    
                    #then concatenate each individual df together
                    df_lc.append(dfsingle)
        except:
            pass
        
    return(df_lc)
