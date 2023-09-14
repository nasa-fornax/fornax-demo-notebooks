import numba
import numpy as np
import time
import pandas as pd
import os
import sys
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import json
from scipy import stats
from scipy import interpolate
from astropy.coordinates import SkyCoord, name_resolve
import astropy.units as u
from astropy.table import Table, vstack, hstack, unique
from astropy.io import fits,ascii
from astropy.time import Time
from astropy.timeseries import TimeSeries

from tqdm import tqdm
sys.path.append('code/')

def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{:.1f}%\n({v:d})'.format(pct, v=val)
        return my_format
    
def unify_lc(df_lc,bands_inlc=['zr','zi','zg'],xres=160,numplots=1):
    x_ztf = np.linspace(0,1600,xres) # X array for interpolation
    x_wise = np.linspace(0,4000,xres) # X array for interpolation

    printcounter = 0
    objects,dobjects,flabels = [],[],[]
    for obj in tqdm(df_lc.data.count(axis=0,level=0).index):

        singleobj = df_lc.data.loc[obj,:,:,:]    
        label = singleobj.index.unique('label')
        bands = singleobj.loc[obj,label[0],:,:].index.get_level_values('band')[:].unique()
        keepobj = 0
        if len(np.intersect1d(bands,bands_inlc))==len(bands_inlc):
            if printcounter<numplots:
                fig= plt.subplots(figsize=(15,5))
                    
            obj_newy = [ [] for _ in range(len(bands_inlc))]
            obj_newdy = [ [] for _ in range(len(bands_inlc))]

            keepobj = 1 # 
            for l,band in enumerate(bands_inlc):
                band_lc = singleobj.loc[obj,label[0], band, :]
                band_lc_clean = band_lc[band_lc.index.get_level_values('time') < 65000]
                x,y,dy = np.array(band_lc_clean.index.get_level_values('time')-band_lc_clean.index.get_level_values('time')[0]),np.array(band_lc_clean.flux),np.array(band_lc_clean.err)

                x2,y2,dy2 = x[np.argsort(x)],y[np.argsort(x)],dy[np.argsort(x)]
                if len(x2)>5:
                    n = np.sum(x2==0)
                    for b in range(1,n): # this is a hack of shifting time of different lightcurves by a bit so I can interpolate! 
                        x2[::b+1]=x2[::b+1]+1*0.001 
                
                    f = interpolate.interp1d(x2,y2,kind='previous',fill_value="extrapolate")
                    df = interpolate.interp1d(x2,dy2,kind='previous',fill_value="extrapolate")
                    
                    if printcounter<numplots:    
                        plt.errorbar(x2,y2,dy2 , capsize = 1.0,marker='.',linestyle='', label = label[0]+band)
                        if band=='w1' or band=='w2':
                            plt.plot(x_wise,f(x_wise),'--',label='nearest interpolation '+str(band))
                        else:
                            plt.plot(x_ztf,f(x_ztf),'--',label='nearest interpolation '+str(band))
                
                    if band =='w1' or band=='w2':
                        obj_newy[l] = f(x_wise)#/f(x_wise).max()
                        obj_newdy[l] = df(x_wise)
                    else:
                        obj_newy[l] = f(x_ztf)#/f(x_ztf).max()
                        obj_newdy[l] = df(x_ztf)#/f(x_ztf).max()
                        
                if len(obj_newy[l])<5: #don't keep objects which have less than x datapoints in any keeoping bands
                    keepobj = 0         
            
            if printcounter<numplots:
                plt.title('Object '+str(obj)+' from '+label[0]+' et al.')
                plt.xlabel('Time(MJD)')
                plt.ylabel('Flux(mJy)')
                plt.legend()
                plt.show()
                printcounter+=1


        if keepobj:
            objects.append(obj_newy)
            dobjects.append(obj_newdy)
            flabels.append(label[0])
    return np.array(objects),np.array(dobjects),flabels

def combine_bands(objects,bands):
    '''
    combine all lightcurves in individual bands of an object
    into one long array, by appending the indecies.
    '''
    dat = []
    for o,ob in enumerate(objects):
        obj = []
        for b in range(len(bands)):
            obj = np.append(obj,ob[b],axis=0)
        dat.append(obj)
    return np.array(dat)

def mean_fractional_variation(lc,dlc):
    '''A common way of defining variability'''
    meanf = np.mean(lc) #mean flux of all points
    varf = np.std(lc)**2
    deltaf = np.mean(dlc)**2
    if meanf<=0:
        meanf = 0.0001
    fvar = (np.sqrt(varf-deltaf))/meanf
    return fvar

def stat_bands(objects,dobjects,bands):
    '''
    returns arrays with maximum,mean,std flux in the 5sigma clipped lightcurves of each band .
    '''
    fvar,maxarray,meanarray = np.zeros((len(bands),len(objects))),np.zeros((len(bands),len(objects))),np.zeros((len(bands),len(objects)))
    for o,ob in enumerate(objects):
        for b in range(len(bands)):
            clipped_arr,l,u = stats.sigmaclip(ob[b], low=5.0, high=5.0)
            clipped_varr,l,u = stats.sigmaclip(dobjects[o,b,:], low=5.0, high=5.0)
            maxarray[b,o] = clipped_arr.max()
            meanarray[b,o] = clipped_arr.mean()
            fvar[b,o] = mean_fractional_variation(clipped_arr,clipped_varr)
    return fvar,maxarray,meanarray


def normalize_mean_objects(data):
    '''
    normalize objects in all bands together by mean value.
    '''
    # normalize each databand
    row_sums = data.mean(axis=1)
    return data / row_sums[:, np.newaxis]

def normalize_max_objects(data):
    '''
    normalize objects in all bands together by max value.
    '''
    # normalize each databand
    row_sums = data.max(axis=1)
    return data / row_sums[:, np.newaxis]

def normalize_clipmax_objects(data,maxarr,band =1):
    '''
    normalize combined data array by by max value after clipping the outliers in one band (second band here).
    '''
    d2 = np.zeros_like(data)
    for i,d in enumerate(data):
        if band<=np.shape(maxarr)[0]:
            d2[i] = (d/maxarr[band,i])
        else:
            d2[i] = (d/maxarr[0,i])
    return d2

# Shuffle before feeding to umap
def shuffle_datalabel(data,labels):
    """shuffles the data, labels and also returns the indecies """
    p = np.random.permutation(len(data))
    data2 = data[p,:]
    fzr=np.array(labels)[p.astype(int)]
    return data2,fzr,p

@numba.njit()
def dtw_distance(series1, series2):
    """
    Returns the DTW similarity distance between two 2-D
    timeseries numpy arrays.
    Arguments:
        series1, series2 : array of shape [n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared
    Returns:
        DTW distance between sequence 1 and 2
    """
    l1 = series1.shape[0]
    l2 = series2.shape[0]
    E = np.empty((l1, l2))

    # Fill First Cell
    E[0][0] = np.square(series1[0] - series2[0])

    # Fill First Column
    for i in range(1, l1):
        E[i][0] = E[i - 1][0] + np.square(series1[i] - series2[0])

    # Fill First Row
    for i in range(1, l2):
        E[0][i] = E[0][i - 1] + np.square(series1[0] - series2[i])

    for i in range(1, l1):
        for j in range(1, l2):
            v = np.square(series1[i] - series2[j])

            v1 = E[i - 1][j]
            v2 = E[i - 1][j - 1]
            v3 = E[i][j - 1]

            if v1 <= v2 and v1 <= v3:
                E[i][j] = v1 + v
            elif v2 <= v1 and v2 <= v3:
                E[i][j] = v2 + v
            else:
                E[i][j] = v3 + v

    return np.sqrt(E[-1][-1])

def stretch_small_values_arctan(data, factor=1.0):
    """
    Stretch small values in an array using the arctan function.

    Parameters:
    - data (numpy.ndarray): The input array.
    - factor (float): A factor to control the stretching. Larger values will stretch more.

    Returns:
    - numpy.ndarray: The stretched array.
    """
    stretched_data = np.arctan(data * factor)
    return stretched_data