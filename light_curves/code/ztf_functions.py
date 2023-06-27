import re

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvo
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from data_structures import MultiIndexDFObject
from tqdm import tqdm


URLBASE = "https://irsa.ipac.caltech.edu/data/ZTF/lc/lc_dr17"
# get a list of files in the dataset using the checksums file
DSFILES = pd.read_table(
        f"{URLBASE}/checksum.md5", sep="\s+", names=["md5", "path"], usecols=["path"]
    ).squeeze().str.removeprefix("./").to_list()


def find_files(coord, ztf_radius, service):
    # get the field, ccdid, and quadrant id using tap
    result = service.run_sync(
        # not really using oid right now but will return it anyway
        f"SELECT {', '.join(['oid', 'field', 'ccdid', 'qid'])} "
        "FROM ztf_objects "
        "WHERE CONTAINS("
        # must be one of 'J2000', 'ICRS', and 'GALACTIC'
        # guessing icrs, but need to check
        f"POINT('ICRS',ra, dec), CIRCLE('ICRS',{coord.ra.deg},{coord.dec.deg},{ztf_radius.value})"
        ")=1"
    )
    field_df = result.to_table().to_pandas().set_index("oid").drop_duplicates()

    # now find the files
    files = []
    for field, ccd, quad in list(field_df.to_records(index=False)):
        fre = re.compile(f"[01]/field{field}/ztf_{field}_z[gri]_c{ccd}_q{quad}_dr17.parquet")
        files.extend([f"{URLBASE}/{f}" for f in filter(fre.match, DSFILES)])

    return files


def ZTF_get_lightcurve(coords_list, labels_list, plotprint=1, ztf_radius=0.000278*u.deg):
    """ Function to add the ZTF lightcurves in all three bands to a multiframe data structure 
     
    Parameters
    ----------
    coords_list : list of astropy skycoords
        the coordinates of the targets for which a user wants light curves
    labels_list: list of strings
        journal articles associated with the target coordinates
    plotprint: int
        print out plots (1) or not(0)
    ztf_radius : float
        search radius, how far from the source should the archives return results
    
    Returns
    -------
    df_lc : MultiIndexDFObject
        the main data structure to store all light curves
    """

    count_nomatch = 0 # counter for non matched objects
    count_plots = 0 
    df_lc = MultiIndexDFObject()

    service = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
    for oid, coord in tqdm(coords_list):
        lab = labels_list[oid]

        # find which files this object is in
        files = find_files(coord, ztf_radius, service)  # ~30 sec~

        # load everything in the files and use astropy to find the object
        # these 3 lines take about 1 sec total
        ztf_lc = vstack([Table.read(file) for file in files])  # 1 sec
        ztf_coords = SkyCoord(ra=ztf_lc["objra"] * u.deg, dec=ztf_lc["objdec"] * u.deg)  # 8 ms
        ztf_lc = ztf_lc[coord.separation(ztf_coords) < ztf_radius]  # 20 ms
        # remove data flagged as bad. 
        # this doesn't work because of the nested structure, but the structure causes more 
        # problems later on and we'll need to think through a solution. punting for now.
        # ztf_lc = ztf_lc[ztf_lc["catflags"] != 32768]

        # add some columns that are used below
        ztf_lc["oid"] = ztf_lc["objectid"]
        ztf_lc["mjd"] = ztf_lc["hmjd"]  # this isn't right but leaving it for now
        filter_dict = {1: "zg", 2: "zr", 3: "zi"}
        ztf_lc["filtercode"] = [filter_dict[fid] for fid in ztf_lc["filterid"]]
        
        # the rest of this doesn't work yet because this new ztf_lc has nested columns
        # and the code below expects a different shape

        if len(np.unique(ztf_lc['oid']))>0:
            # reading in indecies of unique IDs to do coordinate match and find dif magnitudes of same objects 
            idu,inds = np.unique(ztf_lc['oid'],return_index=True)
            if count_plots<plotprint:
                print('object '+str(oid)+' , unique ztf IDs:'+str(len(np.unique(ztf_lc['oid'])))+',in '+str(len(np.unique(ztf_lc['filtercode'])))+' filters')
                plt.figure(figsize=(6,4))
            
            ## Neeeded to remove repeated lightcurves in each band. Will keep only the longest following (Sanchez-Saez et al., 2021)
            filternames = ['zg','zr','zi']
            filtercounter = [0,0,0] #[g,r,i]
            filterflux = [[],[],[]]

            for nn,idd in enumerate(idu):
                sel = (ztf_lc['oid']==idd)

                flux = 10**((ztf_lc['mag'][sel] - 23.9)/(-2.5))  #now in uJy [Based on ztf paper https://arxiv.org/pdf/1902.01872.pdf zeropoint corrections already applied]
                magupper = ztf_lc['mag'][sel] + ztf_lc['magerr'][sel]
                maglower = ztf_lc['mag'][sel] - ztf_lc['magerr'][sel]
                flux_upper = abs(flux - (10**((magupper - 23.9)/(-2.5))))
                flux_lower =  abs(flux - (10**((maglower - 23.9)/(-2.5))))
                fluxerr = (flux_upper + flux_lower) / 2.0
                flux = flux * (1E-3) # now in mJy
                fluxerr = fluxerr * (1E-3) # now in mJy
                if count_plots<plotprint:
                    plt.errorbar(np.round(ztf_lc['mjd'][sel],0),flux,yerr=fluxerr,marker='.',linestyle='',label=ztf_lc['filtercode'][sel][0])
                    plt.legend()
                # if a band is observed before and this has longer lightcurve, remove previous and add this one
                if (filtercounter[filternames.index(ztf_lc['filtercode'][sel][0])]>=1) and (len(flux)<=len(filterflux[filternames.index(ztf_lc['filtercode'][sel][0])])):
                    #print('1st loop, filter'+str(ztf_lc['filtercode'][sel][0])+' len:'+str(len(flux)))
                    filtercounter[filternames.index(ztf_lc['filtercode'][sel][0])]+=1
                elif (filtercounter[filternames.index(ztf_lc['filtercode'][sel][0])]>=1) and (len(flux)>len(filterflux[filternames.index(ztf_lc['filtercode'][sel][0])])):
                    #print('2nd loop, filter'+str(ztf_lc['filtercode'][sel][0])+' len:'+str(len(flux)))
                    df_lc.remove((oid, lab, ztf_lc["filtercode"][sel][0]))
                    dfsingle = pd.DataFrame(dict(flux=flux, err=fluxerr, time=ztf_lc['mjd'][sel], oid=oid, band=ztf_lc['filtercode'][sel][0], label=lab)).set_index(["oid","label", "band", "time"])
                    df_lc.append(dfsingle)
                else:
                    #print('3rd filter'+str(ztf_lc['filtercode'][sel][0])+' len:'+str(len(flux)))
                    dfsingle = pd.DataFrame(dict(flux=flux, err=fluxerr, time=ztf_lc['mjd'][sel], oid=oid, band=ztf_lc['filtercode'][sel][0], label=lab)).set_index(["oid" ,"label","band", "time"])
                    df_lc.append(dfsingle)
                    filtercounter[filternames.index(ztf_lc['filtercode'][sel][0])]+=1
                    filterflux[filternames.index(ztf_lc['filtercode'][sel][0])]=flux
                
            if count_plots<plotprint:
                plt.show()
                count_plots+=1
        else:
            count_nomatch+=1
    print(count_nomatch,' objects did not match to ztf')
    return df_lc
