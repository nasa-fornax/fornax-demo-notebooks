import pandas as pd
from astroquery.heasarc import Heasarc
from tqdm import tqdm

from data_structures import MultiIndexDFObject


def HEASARC_get_lightcurves(coords_list,labels_list,radius, mission_list ):
    """Searches HEASARC archive for light curves from a specific list of mission catalogs
    
    Parameters
    ----------
    coords_list : list of astropy skycoords
        the coordinates of the targets for which a user wants light curves
    labels_list: list of strings
        journal articles associated with the target coordinates
    radius : astropy.units.quantity.Quantity
        search radius, how far from the source should the archives return results
    mission_list : str list
        list of catalogs within HEASARC to search for light curves.  Must be one of the catalogs listed here: 
            https://astroquery.readthedocs.io/en/latest/heasarc/heasarc.html#getting-list-of-available-missions
    Returns
    -------
    df_lc : pandas dataframe
        the main data structure to store all light curves
    """
    
    #for the yang sample, no results are returned, so this is an example that will return a result for testing
    #for ccount in range(1):
        #To get a fermigtrig source
        #coord = SkyCoord('03h41m21.2s -89d00m33.0s', frame='icrs')

        #to get a bepposax source
        #coord = SkyCoord('14h32m00.0s -88d00m00.0s', frame='icrs')

    df_lc = MultiIndexDFObject()
    for objectid, coord in tqdm(coords_list):
        #use astroquery to search that position for either a Fermi or Beppo Sax trigger
        for mcount, mission in enumerate(mission_list):
            try:
                results = Heasarc.query_region(coord, mission = mission, radius = radius)#, sortvar = 'SEARCH_OFFSET_')
                #really just need to save the one time of the Gamma ray detection
                #time is already in MJD for both catalogs
                if mission == 'FERMIGTRIG':
                    time_mjd = results['TRIGGER_TIME'][0].astype(float)
                else:
                    time_mjd = results['TIME'][0].astype(float)
                
                type(time_mjd)
                lab = labels_list[objectid]

                #really just need to mark this spot with a vertical line in the plot
                dfsingle = pd.DataFrame(dict(flux=[0.1], err=[0.1], time=[time_mjd], objectid=[objectid], band=[mission], label=lab)).set_index(["objectid", "label", "band", "time"])

                # Append to existing MultiIndex light curve object
                df_lc.append(dfsingle)

            except AttributeError:
            #print("no results at that location for ", mission)
                pass
            
    return df_lc

#**** These HEASARC searches are returning an attribute error because of an astroquery bug
# bug submitted to astroquery Oct 18, waiting for a fix.
# if that gets fixed, can probably change this cell 
