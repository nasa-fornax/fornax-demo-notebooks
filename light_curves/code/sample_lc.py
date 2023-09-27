import sys
import time
import warnings
from math import ceil
import multiprocessing as mp
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy import stats
import pandas as pd
from panstarrs import panstarrs_get_lightcurves
from gaia_functions import Gaia_get_lightcurve
from HCV_functions import HCV_get_lightcurves
from icecube_functions import icecube_get_lightcurve
from sample_selection import get_lamassa_sample, get_macleod16_sample, get_ruan_sample, get_macleod19_sample, get_sheng_sample, get_green_sample, get_lyu_sample, get_lopeznavas_sample, get_hon_sample, get_yang_sample,get_SDSS_sample, get_paper_sample, clean_sample,nonunique_sample,TDE_id2coord
from data_structures import MultiIndexDFObject
from heasarc_functions import HEASARC_get_lightcurves
from TESS_Kepler_functions import TESS_Kepler_get_lightcurves
from WISE_functions import WISE_get_lightcurves
from ztf_functions import ZTF_get_lightcurve
from ML_utils import unify_lc, stat_bands, autopct_format, combine_bands,\
mean_fractional_variation, normalize_mean_objects, normalize_max_objects, \
normalize_clipmax_objects, shuffle_datalabel, dtw_distance, stretch_small_values_arctan


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
    num_normal_QSO = 2000
    get_SDSS_sample(coords, labels, num_normal_QSO)

    ## ADD TDEs to the sample, manually copied the TDE ZTF names from Hammerstein et al. 2023
    tde_names = ['ZTF18aabtxvd','ZTF18aahqkbt','ZTF18abxftqm','ZTF18acaqdaa','ZTF18acpdvos','ZTF18actaqdw','ZTF19aabbnzo','ZTF18acnbpmd','ZTF19aakiwze','ZTF19aakswrb','ZTF17aaazdba','ZTF19aapreis','ZTF19aarioci','ZTF19abhejal','ZTF19abhhjcc','ZTF19abidbya','ZTF19abzrhgq','ZTF19accmaxo','ZTF20aabqihu','ZTF19acspeuw','ZTF20aamqmfk','ZTF18aakelin','ZTF20abjwvae','ZTF20abfcszi','ZTF20abefeab','ZTF20abowque','ZTF20abrnwfc','ZTF20acitpfz','ZTF20acqoiyt', 'ZTF20abnorit']
    TDE_id2coord(tde_names,coords,labels)
    

    get_paper_sample('2015ApJ...810...14A','FermiBL',coords,labels)
    get_paper_sample('2019A&A...627A..33D','Cicco19',coords,labels)
    get_paper_sample('2022ApJ...933...37W','Galex variable 22',coords,labels)
    get_paper_sample('2020ApJ...896...10B','Palomar variable 20',coords,labels)

    #To remove duplicates from the list if combining multiple references clean_sample can be used 
    # the call below with nonunique_sample just changes the structure to mimic the output of clean sample
    coords_list, labels_list = nonunique_sample(coords, labels) 
    print('final sample: ',len(coords))
    return coords_list,labels_list
 

def parallel_lc(coords_list,labels_list,parquet_savename = 'data/df_lc_.parquet.gzip'):
    ''' Check all the archives for the light curve data of the 
    list of coordinates given in input in parallel and return a 
    muldidimensional lightcurve dataframe.'''
    
    max_fermi_error_radius = str(1.0)  
    max_sax_error_radius = str(3.0)
    heasarc_cat = ["FERMIGTRIG", "SAXGRBMGRB"]
    error_radius = [max_fermi_error_radius , max_sax_error_radius]
    bandlist = ["W1", "W2"]
    wise_radius = 1.0 * u.arcsec
    panstarrs_radius = 1.0 / 3600.0  # search radius = 1 arcsec
    lk_radius = 1.0  # arcseconds
    hcv_radius = 1.0 / 3600.0  # radius = 1 arcsec
    n_workers = 8  # this should equal the total number of archives called
    ztf_nworkers = None

    parallel_starttime = time.time()

    # start a multiprocessing pool and run all the archive queries
    # "spawn" new processes because it uses less memory and is thread safe
    # in particular, this is required for pd.read_parquet (used by ZTF_get_lightcurve)
    # https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn
    mp.set_start_method("spawn", force=True)
    parallel_df_lc = MultiIndexDFObject()  # to collect the results
    callback = parallel_df_lc.append  # will be called once on the result returned by each archive
    with mp.Pool(processes=n_workers) as pool:

        # start the processes that call the archives
        pool.apply_async(
            Gaia_get_lightcurve, (coords_list,  labels_list , 1), callback=callback
        )
        pool.apply_async(
            HEASARC_get_lightcurves, (coords_list, labels_list, heasarc_cat, error_radius), callback=callback
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
        pool.apply_async(
            WISE_get_lightcurves, (coords_list, labels_list, wise_radius, bandlist), callback=callback
        )
        pool.apply_async(
            ZTF_get_lightcurve, (coords_list, labels_list, ztf_nworkers), callback=callback
        )

        pool.close()  # signal that no more jobs will be submitted to the pool
        pool.join()  # wait for all jobs to complete, including the callback

    parallel_endtime = time.time()
    print('parallel processing took', parallel_endtime - parallel_starttime, 's')
    
    # # Save the data for future use with ML notebook
    parallel_df_lc.data.to_parquet(parquet_savename)
    print("file saved!")
    return parallel_df_lc

def main():
    c,l = build_sample()
    dflc = parallel_lc(c,l,parquet_savename = 'data/df_lc_smalltest.parquet.gzip')
    # Unify for ML and save
    
if __name__ == "__main__":
    main()