import sys
import time
import warnings
from math import ceil
from multiprocessing import Pool

import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy import stats
import pandas as pd

warnings.filterwarnings('ignore')

# local code imports
sys.path.append('code/')
from panstarrs import panstarrs_get_lightcurves
from gaia_functions import Gaia_get_lightcurve
from HCV_functions import HCV_get_lightcurves
from icecube_functions import icecube_get_lightcurve
from sample_selection import get_lamassa_sample, get_macleod16_sample, get_ruan_sample, get_macleod19_sample, get_sheng_sample, get_green_sample, get_lyu_sample, get_lopeznavas_sample, get_hon_sample, get_yang_sample,get_SDSS_sample, get_paper_sample, clean_sample,TDE_id2coord
from data_structures import MultiIndexDFObject
from heasarc_functions import HEASARC_get_lightcurves
from TESS_Kepler_functions import TESS_Kepler_get_lightcurves
from WISE_functions import WISE_get_lightcurves
from ztf_functions_old import ZTF_get_lightcurve


def build_sample():
    '''Putting together a sample of SDSS quasars, WISE variable AGNs,
    TDEs, Changing look AGNs, .. coordinates from different 
    papers.'''
    
    coords =[]
    labels = []

    get_lamassa_sample(coords, labels)  #2015ApJ...800..144L
    get_macleod16_sample(coords, labels) #2016MNRAS.457..389M
    get_ruan_sample(coords, labels) #2016ApJ...826..188R
    get_macleod19_sample(coords, labels)  #2019ApJ...874....8M
    get_sheng_sample(coords, labels)  #2020ApJ...889...46S
    get_green_sample(coords, labels)  #2022ApJ...933..180G
    get_lyu_sample(coords, labels)  #z32022ApJ...927..227L
    get_lopeznavas_sample(coords, labels)  #2022MNRAS.513L..57L
    get_hon_sample(coords, labels)  #2022MNRAS.511...54H
    get_yang_sample(coords, labels)   #2018ApJ...862..109Y

    # Variable AGN sample from Ranga/Andreas:
    VAGN = pd.read_csv('data/WISE_MIR_variable_AGN_with_PS1_photometry_and_SDSS_redshift.csv')
    vagn_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(VAGN['SDSS_RA'], VAGN['SDSS_Dec'])]
    vagn_labels = ['WISE-Variable' for ra in VAGN['SDSS_RA']]
    coords.extend(vagn_coords)
    labels.extend(vagn_labels)    

    #now get some "normal" QSOs for use in the classifier
    #there are ~500K of these, so choose the number based on
    #a balance between speed of running the light curves and whatever 
    #the ML algorithms would like to have
    num_normal_QSO = 1000 
    get_SDSS_sample(coords, labels, num_normal_QSO)

    ## ADD TDEs to the sample, manually copied the TDE ZTF names from Hammerstein et al. 2023
    tde_names = ['ZTF18aabtxvd','ZTF18aahqkbt','ZTF18abxftqm','ZTF18acaqdaa','ZTF18acpdvos','ZTF18actaqdw','ZTF19aabbnzo',
                 'ZTF18acnbpmd','ZTF19aakiwze','ZTF19aakswrb','ZTF17aaazdba','ZTF19aapreis','ZTF19aarioci','ZTF19abhejal',
                 'ZTF19abhhjcc','ZTF19abidbya','ZTF19abzrhgq','ZTF19accmaxo','ZTF20aabqihu','ZTF19acspeuw','ZTF20aamqmfk',
                 'ZTF18aakelin','ZTF20abjwvae','ZTF20abfcszi','ZTF20abefeab','ZTF20abowque','ZTF20abrnwfc','ZTF20acitpfz',
                 'ZTF20acqoiyt', 'ZTF20abnorit']
    TDE_id2coord(tde_names,coords,labels)
    
    get_paper_sample('2019A&A...627A..33D','Cicco19',coords,labels)
    get_paper_sample('2022ApJ...933...37W','Galex variable 22',coords,labels)
    get_paper_sample('2020ApJ...896...10B','Palomar variable 20',coords,labels)

    #remove duplicates from the list if combining multiple references
    coords_list, labels_list = clean_sample(coords, labels)
    print('final sample: ',len(coords_list))
    return coords_list,labels_list

def parallel_lc(coords_list,labels_list):
    ''' Check all the archives for the light curve data of the 
    list of coordinates given in input in parallel and return a 
    muldidimensional lightcurve dataframe.'''
    
    mission_list = ["FERMIGTRIG", "SAXGRBMGRB"]
    heasarc_radius = 0.1 * u.degree
    bandlist = ["w1", "w2"]
    wise_radius = 1.0
    panstarrs_radius = 1.0 / 3600.0  # search radius = 1 arcsec
    lk_radius = 1.0  # arcseconds
    hcv_radius = 1.0 / 3600.0  # radius = 1 arcsec
    n_single_archives, n_multiple_archives = 6, 2  # must sum to total number of archives called
    n_chunks_per_archive = 5  # will make one api call per chunk per 'multiple' archive
    n_workers = n_single_archives + n_multiple_archives * n_chunks_per_archive
    
    parallel_starttime = time.time()

    # start a multiprocessing pool and run all the archive queries
    parallel_df_lc = MultiIndexDFObject()  # to collect the results
    callback = parallel_df_lc.append  # will be called once on the result returned by each archive
    with Pool(processes=n_workers) as pool:

        # start the processes that call the fast archives
        pool.apply_async(
            Gaia_get_lightcurve, (coords_list, labels_list, 1), callback=callback
        )
        pool.apply_async(
            HEASARC_get_lightcurves, (coords_list, labels_list, heasarc_radius, mission_list), callback=callback
        )
        pool.apply_async(
            HCV_get_lightcurves, (coords_list, labels_list, hcv_radius), callback=callback
        )
        pool.apply_async(
            icecube_get_lightcurve, (coords_list, labels_list, 3, "./data/", 1), callback=callback
        )
        pool.apply_async(
            panstarrs_get_lightcurves, (coords_list, labels_list, panstarrs_radius), callback=callback
        )
        pool.apply_async(
            TESS_Kepler_get_lightcurves, (coords_list, labels_list, lk_radius), callback=callback
        )

        # split coords_list into smaller chunks and call remaining archives
        chunksize = ceil(len(coords_list) / n_chunks_per_archive)  # num coords per api call
        for n in range(0, len(coords_list), chunksize):
            coords = coords_list[n : n + chunksize]

            # start the processes that call the slow archives
            pool.apply_async(
                WISE_get_lightcurves, (coords, labels_list, wise_radius, bandlist), callback=callback
            )
            pool.apply_async(
                ZTF_get_lightcurve, (coords, labels_list, 0), callback=callback
            )

        pool.close()  # signal that no more jobs will be submitted to the pool
        pool.join()  # wait for all jobs to complete, including the callback

    parallel_endtime = time.time()
    print('parallel processing took', parallel_endtime - parallel_starttime, 's')
    
    # # Save the data for future use with ML notebook
    parquet_savename = 'data/df_lc_.parquet.gzip'
    parallel_df_lc.data.to_parquet(parquet_savename)
    print("file saved!")
    return parallel_df_lc

def main():
    c,l = build_sample()
    dflc = parallel_lc(c,l)
    # Unify for ML and save
    
if __name__ == "__main__":
    main()