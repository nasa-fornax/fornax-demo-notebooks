import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from alerce.core import Alerce
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from tqdm import tqdm

from data_structures import MultiIndexDFObject


def ZTF_id2coord(object_ids,coords,labels,verbose=1):
    ''' To find and append coordinates of objects with only ZTF obj name'''
    alerce = Alerce()
    objects = alerce.query_objects(oid=object_ids, format="pandas")
    ztf_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(objects['meanra'],objects['meandec'])]
    ztf_labels = ['ZTF-Objname' for ra in objects['meanra']]
    coords.extend(ztf_coords)
    labels.extend(ztf_labels)
    if verbose:
        print('number of ztf coords added by Objectname:',len(objects['meanra']))
    

def ZTF_get_lightcurve(coords_list, labels_list, plotprint=1, ztf_radius=0.000278):
    ''' Function to add the ZTF lightcurves in all three bands 
    to a multiframe data structure 
    input:
     --- list of coordinates
     --- list of labels
     --- ztf_radius default to 0.000278
     --- optional plotprint number of objects to plot
     
     output:
      --- the multi-data frame with ztf sources appended.
      '''

    count_nomatch = 0 # counter for non matched objects
    count_plots = 0 
    df_lc = MultiIndexDFObject()

    for objectid, coord in tqdm(coords_list):
        #doesn't take SkyCoord
        ra = coord.ra.deg 
        dec = coord.dec.deg 
        lab = labels_list[objectid]
        #make the string for the URL query ask for all three bands (g, r, i)
        #don't want data that is flagged as unusable by the ZTF pipeline
        urlstr = 'https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE %f %f %f&BANDNAME=g,r,i&FORMAT=ipac_table&BAD_CATFLAGS_MASK=32768'%(ra, dec,ztf_radius)
        response = requests.get(urlstr)
        if response.ok:
            ztf_lc = ascii.read(response.text, format='ipac')
            if len(np.unique(ztf_lc['oid']))>0:
                # reading in indecies of unique IDs to do coordinate match and find dif magnitudes of same objects 
                idu,inds = np.unique(ztf_lc['oid'],return_index=True)
                if count_plots<plotprint:
                    print('object '+str(objectid)+' , unique ztf IDs:'+str(len(np.unique(ztf_lc['oid'])))+',in '+str(len(np.unique(ztf_lc['filtercode'])))+' filters')
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
                        df_lc.remove((objectid, lab, ztf_lc["filtercode"][sel][0]))
                        dfsingle = pd.DataFrame(dict(flux=flux, err=fluxerr, time=ztf_lc['mjd'][sel], objectid=objectid, band=ztf_lc['filtercode'][sel][0], label=lab)).set_index(["objectid","label", "band", "time"])
                        df_lc.append(dfsingle)
                    else:
                        #print('3rd filter'+str(ztf_lc['filtercode'][sel][0])+' len:'+str(len(flux)))
                        dfsingle = pd.DataFrame(dict(flux=flux, err=fluxerr, time=ztf_lc['mjd'][sel], objectid=objectid, band=ztf_lc['filtercode'][sel][0], label=lab)).set_index(["objectid" ,"label","band", "time"])
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
