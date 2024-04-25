import os

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

import pandas as pd

from astroquery.mast import Observations

from data_structures_spec import MultiIndexDFObject

from specutils import Spectrum1D

def HST_get_spec(sample_table, search_radius_arcsec, datadir):
    '''
    Retrieves HST spectra for a list of sources.

    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    search_radius_arcsec : `float`
        Search radius in arcseconds.
    datadir : `str`
        Data directory where to store the data. Each function will create a
        separate data directory (for example "[datadir]/HST/" for HST data).

    Returns
    -------
    df_lc : MultiIndexDFObject
        The main data structure to store all spectra
        
    '''

    ## Create directory
    this_data_dir = os.path.join(datadir , "HST/")
    
    
    ## Initialize multi-index object:
    df_spec = MultiIndexDFObject()
    
    for stab in sample_table:

        print("Processing source {}".format(stab["label"]))

        ## Query results
        search_coords = stab["coord"]
        query_results = Observations.query_criteria(coordinates = search_coords, radius = search_radius_arcsec * u.arcsec,
                                                dataproduct_type=["spectrum"], obs_collection=["HST"], intentType="science", calib_level=[3,4],
                                               )
        print("Number of search results: {}".format(len(query_results)))

        if len(query_results) > 0: # found some spectra
            
            
            ## Retrieve spectra
            data_products_list = Observations.get_product_list(query_results)
            
            ## Filter
            data_products_list_filter = Observations.filter_products(data_products_list,
                                                    productType=["SCIENCE"],
                                                    extension="fits",
                                                    calib_level=[3,4], # only fully reduced or contributed
                                                    productSubGroupDescription=["SX1"] # only 1D spectra
                                                                    )
            print("Number of files to download: {}".format(len(data_products_list_filter)))

            if len(data_products_list_filter) > 0:
                
                ## Download
                download_results = Observations.download_products(data_products_list_filter, download_dir=this_data_dir)
            
                
                ## Create table
                keys = ["filters","obs_collection","instrument_name","calib_level","t_obs_release","proposal_id","obsid","objID","distance"]
                tab = Table(names=keys , dtype=[str,str,str,int,float,int,int,int,float])
                for jj in range(len(download_results)):
                    tmp = query_results[query_results["obsid"] == data_products_list_filter["obsID"][jj]][keys]
                    tab.add_row( list(tmp[0]) )
                
                ## Create multi-index object
                for jj in range(len(tab)):
                
                    # open spectrum
                    filepath = download_results[jj]["Local Path"]
                    print(filepath)
                    spec1d = Spectrum1D.read(filepath)  
                    
                    dfsingle = pd.DataFrame(dict(wave=[spec1d.spectral_axis] , flux=[spec1d.flux], err=[np.repeat(0,len(spec1d.flux))],
                                                 label=[stab["label"]],
                                                 objectid=[stab["objectid"]],
                                                 #objID=[tab["objID"][jj]],
                                                 #obsid=[tab["obsid"][jj]],
                                                 mission=[tab["obs_collection"][jj]],
                                                 instrument=[tab["instrument_name"][jj]],
                                                 filter=[tab["filters"][jj]],
                                                )).set_index(["objectid", "label", "filter", "mission"])
                    df_spec.append(dfsingle)
            
            else:
                print("Nothing to download for source {}.".format(stab["label"]))
        else:
            print("Source {} could not be found".format(stab["label"]))
        

    return(df_spec)