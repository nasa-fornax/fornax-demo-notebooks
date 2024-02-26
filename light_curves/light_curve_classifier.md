---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Light Curve Classifier
***

## Learning Goals
By the end of this tutorial, you will be able to:
- do some basic data cleaning and filtering to prepare data for ML algorithms
- work with Pandas data frames as a way of storing time domain datasets
- use sktime & pyts algorithms to train a classifier and predict values on a test dataset

## Introduction
The science goal of this notebook is to find a classifier that can accurately discern changing look active galactic nuclei (CLAGN) from a broad sample of all Sloan Digital Sky Survey (SDSS) identified Quasars (QSOs) based solely on archival photometry in the form of multiwavelength light curves.  

CLAGN are astrophysically interesting objects because they appear to change state.  CLAGN are characterized by the appearance or disappearance of broad emission lines on timescales of order months.  Astronomers would like to understand the physical mechanism behind this apparent change of state.  However, only a few hundered CLAGN are known, and finding CLAGN is observationally expensive, traditionally requiring multiple epochs of spectroscopy.  Being able to identify CLAGN in existing large samples would allow us to identify a statisitcally significant sample from which we could better understand the underlying physics.

This notebook walks through an exercise in using multiwavelength photometry alone (no spectroscopy) to learn if we can identify CLAGN based on their light curves alone.  If we are able to find a classifier that can differentiate CLAGN from SDSS QSOs, we would then be able to run the entire sample of SDSS QSOs (~500,000) to find additional CLAGN candidates.

Input to this notebook is output of a previous demo notebook which generates multiwavelength light curves from archival data.  THis notebook starts with light curves, does data prep, and runs the light curves through multiple ML classification algorithms.  There are many ML algorthms to choose from; We choose to use [sktime](https://www.sktime.net/en/stable/index.html) algorithms for time domain classification beacuse it is a library of many algorithms specifically tailored to time series datasets.  It is based on the sklearn library so syntax is familiar to many users.

The challenges of this time-domain dataset for ML work are:
1. Multi-variate = There are multiple bands of observations per target (13+)
2. Unequal length = Each band has a light curve with different sampling than other bands
3. Missing data = Not each object has all observations in all bands




## Input
Light curve parquet file of multiwavelength light curves from the light_curve_generator.md demo notebook in this same repo.  The format of the light curves is a Pandas multiindex data frame.

We choose to use a Pandas multiindex dataframe to store and work with the data because it fulfills these requirements:
1. It can handle the above challenges of a dataset = multi-variate, unqueal length with missing data.
2. Multiple targets (multiple rows)
3. Pandas has some built in understanding of time units
4. Can be scaled up to big data numbers of rows (altough we don't push to out of memory structures in this use case)
5. Pandas is user friendly with a lot of existing functionality

A useful reference for what sktime expects as input to its ML algorithms: https://github.com/sktime/sktime/blob/main/examples/AA_datatypes_and_datasets.ipynb

## Output
Trained classifiers as well as estimates of their accuracy and plots of confusion matrices

## Imports
- `pandas` to work with light curve data structure
- `numpy` for numerical calculations
- `matplotlib` for plotting
- `sys` for paths
- `astropy` to work with coordinates/units and data structures
- `tqdm` for showing progress meter
- `sktime` ML algorithms specifically for time-domain data
- `sklearn` general use ML algorthims with easy to use interface
- `scipy` for statistical analysis
- `json` for storing intermediate files
- `google_drive_downloader` to access files stored in google drive
- `pyts` time series ML algorithms
  
## Authors
Jessica Krick, Shooby Hemmati, Troy Raen, Brigitta Sipocz, Andreas Faisst, Vandana Desai, Dave Shoop

## Acknowledgements

```{code-cell} ipython3
#insure all dependencies are installed
!pip install -r requirements-lc_classifier.txt
#need updated version of astropy, so this is here temporarily to make sure we grab that.
!pip install -U astropy
```

```{code-cell} ipython3
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.table import Table
from google_drive_downloader import GoogleDriveDownloader as gdd
from scipy.stats import sigmaclip
from tqdm import tqdm
import json

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics.cluster import completeness_score, homogeneity_score

from sktime.classification.deep_learning import CNNClassifier
from sktime.classification.dictionary_based import IndividualTDE
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.dummy import DummyClassifier
from sktime.classification.ensemble import WeightedEnsembleClassifier
from sktime.classification.feature_based import Catch22Classifier, RandomIntervalClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.kernel_based import Arsenal, RocketClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.registry import all_estimators, all_tags
from sktime.datatypes import check_is_mtype

from pyts.classification import KNeighborsClassifier
from pyts.classification import SAXVSM
from pyts.classification import BOSSVS
from pyts.classification import LearningShapelets
from pyts.classification import TimeSeriesForest

# local code imports
sys.path.append('code_src/')
from fluxconversions import mjd_to_jd
```

## 1. Read in a dataset of archival light curves

```{code-cell} ipython3
#access structure of light curves made in the light curve generator notebook
# has known CLAGN & random SDSS small sample of targets, all bands
savename_df_lc = './data/small_CLAGN_SDSS_df_lc.parquet'
gdd.download_file_from_google_drive(file_id='1DrB-CWdBBBYuO0WzNnMl5uQnnckL7MWH',
                                    dest_path=savename_df_lc,
                                    unzip=True)

df_lc = pd.read_parquet(savename_df_lc)


#access the sample_table made in the light curve generator notebook
#has information about the sample including ra & dec 
savename_sample = './data/small_CLAGN_SDSS_sample.ecsv'
gdd.download_file_from_google_drive(file_id='1pSEKVP4LbrdWQK9ws3CaI90m3Z_2fazL',
                                    dest_path=savename_sample,
                                    unzip=True)
sample_table = Table.read(savename_sample, format='ascii.ecsv')
```

```{code-cell} ipython3
#get rid of indices set in the light curve code and reset them as needed before sktime algorithms
df_lc = df_lc.reset_index()  

#what does the dataset look like at the start?
df_lc
```

## 2. Data Prep
The majority of work in all ML projects is preparing and cleaning the data.  As most do, this dataset needs significant work before it can be fed into a ML algorithm.  Data preparation includes everything from removing statistical outliers to putting it in the correct data format for the algorithms.

+++

### 2.1 Remove bands with not enough data
For this use case of CLAGN classification, we don't need to include some of the bands
that are known to be sparse.  Most ML algorithms cannot handle sparse data so one way to accomodate that 
is to remove the sparsest datasets.

```{code-cell} ipython3
##what are the unique set of bands included in our light curves
df_lc.band.unique()

# get rid of some of the bands that don't have enough data for all the sources
#CLAGN are generall fainter targets, and therefore mostly not found in datasets like TESS & K2

bands_to_drop = ["IceCube", "TESS", "FERMIGTRIG", "K2"]
df_lc = df_lc[~df_lc["band"].isin(bands_to_drop)]
```

### 2.2 Combine Labels for a Simpler Classification
All CLAGN start in the dataset as having labels based on their discovery paper.  Because we want one sample with all known CLAGN, we change those discovery names to be simply "CLAGN" for all CLAGN, regardless of origin.

```{code-cell} ipython3
df_lc['label'] = df_lc.label.str.replace('MacLeod 16', 'CLAGN')
df_lc['label'] = df_lc.label.str.replace('LaMassa 15', 'CLAGN')
df_lc['label'] = df_lc.label.str.replace('Yang 18', 'CLAGN')
df_lc['label'] = df_lc.label.str.replace('Lyu 22', 'CLAGN')
df_lc['label'] = df_lc.label.str.replace('Hon 22', 'CLAGN')
df_lc['label'] = df_lc.label.str.replace('Sheng 20', 'CLAGN')
df_lc['label'] = df_lc.label.str.replace('MacLeod 19', 'CLAGN')
df_lc['label'] = df_lc.label.str.replace('Green 22', 'CLAGN')
df_lc['label'] = df_lc.label.str.replace('Lopez-Navas 22', 'CLAGN')
```

```{code-cell} ipython3
print(df_lc.groupby(["objectid"]).ngroups, "n objects before removing missing band data")
```

### 2.3 Data Visualization
- can we see any trends by examining plots of a subset of the data?

```{code-cell} ipython3
#chhose your own adventure, the bands from which you can choose are:
df_lc.band.unique()
```

```{code-cell} ipython3
#plot a single band for all objects
band_of_interest = 'zr'
band_lc = df_lc[df_lc['band'] == band_of_interest]
#reset zero time to be start of that mission
band_lc["time"] = band_lc["time"] - band_lc["time"].min()
band_lc.time.min()

band_lc.set_index('time', inplace = True)  #helps with the plotting

#drop some objects to try to clear up plot
querystring1 = 'objectid < 162'
querystring2 = 'objectid > 200'
band_lc = band_lc.drop(band_lc.query(querystring1 ).index)
band_lc = band_lc.drop(band_lc.query(querystring2 ).index)

#quick normalization for plotting
#we normalize for real after cleaning the data
# make a new column with max_r_flux for each objectid
band_lc['mean_band'] = band_lc.groupby('objectid', sort=False)["flux"].transform('mean')
band_lc['sigma_band'] = band_lc.groupby('objectid', sort=False)["flux"].transform('std')

#choose to normalize (flux - mean) / sigma
band_lc['flux'] = (band_lc['flux'] - band_lc['mean_band']).div(band_lc['sigma_band'], axis=0)

#want to have two different sets so I can color code
clagn_df = band_lc[band_lc['label'] == 'CLAGN']
sdss_df = band_lc[band_lc['label'] == 'SDSS']
print(clagn_df.groupby(["objectid"]).ngroups, "n objects CLAGN ")
print(sdss_df.groupby(["objectid"]).ngroups, "n objects SDSS ")

#groupy objectid & plot flux vs. time
fig, ax = plt.subplots(figsize=(10,6))
lc_sdss = sdss_df.groupby(['objectid'])['flux'].plot(kind='line', ax=ax, color = 'gray', label = 'SDSS', linewidth = 0.3)
lc_clagn = clagn_df.groupby(['objectid'])['flux'].plot(kind='line', ax=ax, color = 'orange', label = 'CLAGN', linewidth = 1)

#add legend and labels/titles
legend_elements = [Line2D([0], [0], color='orange', lw=4, label='CLAGN'),
                   Line2D([0], [0], color='gray', lw=4, label='SDSS')]
ax.legend(handles=legend_elements, loc='best')

ax.set_ylabel('Normalized Flux')
ax.set_xlabel('Time in days since start of mission')
plt.title(f"{band_of_interest} light curves")

#tailored to ZTF r band with lots of data
ax.set_ylim([-2, 4])
ax.set_xlim([1000, 1250])
```

### 2.4 Clean the dataset of unwanted data
"unwanted" includes:
- NaNs
  - SKtime does not work with NaNs
- zero flux
  - there are a few flux measurements that come into our dataframe with zeros.  It is not clear what these are, and zero will be used to mean lack of observation in the rest of this notebook, so want to drop these rows at the outset.
- outliers in uncertainty
  - This is a tricky job because we want to keep astrophysical sources that are variable objects, but remove instrumental noise and CR (ground based).  The user will need to choose a sigma clipping threshold, and there is some plotting functionality available to help users make that decision
- objects with no measurements in WISE W1 band
  - Below we want to normalize all light curves by W1, so we neeed to remove those objects without W1 fluxes because there will be nothing to normalize those light curves with.  We don't want to have un-normalized data.
- objects with incomplete data
  - Incomplete is defined here as not enough flux measurements to make a good light curve.  Some bands in some objects have only a few datapoints. Three data points is not large enough for KNN interpolation, so we will consider any array with fewer than 4 photometry points to be incomplete data.  Another way of saying this is that we choose to remove those light curves with 3 or 
fewer data points.

```{code-cell} ipython3
def sigmaclip_lightcurves(df_lc, sigmaclip_value = 10.0, include_plot = False, verbose = False):
    """
    Sigmaclip to remove bad values from the light curves; optionally plots histograms of uncertainties
        to help determine sigmaclip_value from the data. 
    
    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info
   
    sigmaclip_value: float
        what value of sigma should be used to make the cuts

    include_plot: bool
        have the function plot histograms of uncertainties for each band
        
    Returns
    --------
    df_lc: MultiIndexDFObject with all  light curves
        
    """
    #keep track of how many rows this removes
    start_len = len(df_lc.index)

    #setup to collect the outlier thresholds per band to later reject
    nsigmaonmean= {}

    if include_plot:
        #create the figure and axes
        fig, axs = plt.subplots(5, 3, figsize = (12, 12))

        # unpack all the axes subplots
        axe = axs.ravel()

    #for each band
    for count, (bandname, singleband) in enumerate(df_lc.groupby("band")):
    
        #use scipy sigmaclip to iteratively determine sigma on the dataset
        clippedarr, lower, upper = sigmaclip(singleband.err, low = sigmaclip_value, high = sigmaclip_value)
    
        #store this value for later
        nsigmaonmean[bandname] = upper
    
        if include_plot:        
            #plot distributions and print stddev
            singleband.err.plot(kind = 'hist', bins = 30, subplots =True, ax = axe[count],label = bandname+' '+str(upper), legend=True)

    #remove data that are outside the sigmaclip_value
    #make one large querystring joined by "or" for all bands in df_lc
    querystring = " | ".join(f'(band == {bandname!r} & err > {cut})' for bandname, cut in nsigmaonmean.items())
    clipped_df_lc = df_lc.drop(df_lc.query(querystring).index)

    #how much data did we remove with this sigma clipping?
    #This should inform our choice of sigmaclip_value.

    if verbose:
        end_len = len(clipped_df_lc.index)
        fraction = ((start_len - end_len) / start_len) *100.
        print(f"This {sigmaclip_value} sigma clipping removed {fraction}% of the rows in df_lc")

    return clipped_df_lc
```

```{code-cell} ipython3
def remove_objects_without_band(df_lc, bandname_to_drop = "W1", verbose=False):
    """
    Get rid of the light curves which do not have data in the band we want to use for normalization.  
    
    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info

    bandname_to_drop: str "w1"

    verbose: should the function provide feedback
        
    Returns
    --------
    drop_df_lc: MultiIndexDFObject with all  light curves
        
    """

    #maka a copy so we can work with it
    drop_df_lc = df_lc
    
    #keep track of how many get dropped
    dropcount = 0

    #for each object
    for oid , singleoid in drop_df_lc.groupby("objectid"):
        #what bands does that object have
        bandname = singleoid.band.unique().tolist()
    
        #if it doesn't have bandname_to_drop:
        if bandname_to_drop not in bandname:
            #delete this oid from the dataframe of light curves
            indexoid = drop_df_lc[ (drop_df_lc['objectid'] == oid)].index
            drop_df_lc.drop(indexoid , inplace=True)
        
            #keep track of how many are being deleted
            dropcount = dropcount + 1
        
    if verbose:    
        print( dropcount, "objects without", bandname_to_drop, " were removed")

    return drop_df_lc
```

```{code-cell} ipython3
def remove_incomplete_data(df_lc, threshold_too_few = 3, verbose = True):
    """
    Remove those light curves that don't have enough data for classification.
       
    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info

    threshold_too_few: Int
        Define what the threshold is for too few datapoints.
        Default is 3
        
    Returns
    --------
    df_lc: MultiIndexDFObject with all  light curves
        
    """

    #how many groups do we have before we start
    if verbose:
        print(df_lc.groupby(["band", "objectid"]).ngroups, "n groups before removing incomplete data")

    #use pandas .filter to remove small groups
    complete_df_lc = df_lc.groupby(["band", "objectid"]).filter(lambda x: len(x) > threshold_too_few)

    #how many groups do we have after culling?
    if verbose:
        print(complete_df_lc.groupby(["band", "objectid"]).ngroups, "n groups after removing incomplete data")

    return complete_df_lc
    
```

```{code-cell} ipython3
#drop rows which have Nans
df_lc.dropna(inplace = True, axis = 0)

#drop rows with zero flux
querystring = 'flux < 0.000001'
df_lc = df_lc.drop(df_lc.query(querystring).index)

#remove outliers
sigmaclip_value = 10.0
df_lc = sigmaclip_lightcurves(df_lc, sigmaclip_value, include_plot = False)
print(df_lc.groupby(["objectid"]).ngroups, "n objects after sigma clipping")


#remove objects without W1 fluxes
df_lc = remove_objects_without_band(df_lc, 'W1', verbose=True)
print(df_lc.groupby(["objectid"]).ngroups, "n objects after removing objects without W1")

#remove incomplete data
threshold_too_few = 3
df_lc = remove_incomplete_data(df_lc, threshold_too_few)

print(df_lc.groupby(["objectid"]).ngroups, "n objects after removing incomplete data")
```

### 2.5 Missing Data
Some objects do not have light curves in all bands.  Some ML algorithms can handle mising data, but not all, so we try to do something intentional and sensible to handle this missing data up front.

There are two options here:
1) We will add light curves with zero flux and err values for the missing data.  SKtime does not like NaNs, so we choose zeros.  This option has the benefit of including more bands and therefore more information, but the drawback of having some objects have bands with entire arrays of zeros.
2) Remove bands which have less data from all objects so that there are no objects with missing data.  This has the benefit of less zeros, but the disadvantage of throwing away some information for the few objects which do have light curves in the bands which will be removed.  

Functions are inlcuded for both options.

```{code-cell} ipython3
def make_zero_light_curve(oid, band, label):
    """
    Make placeholder light curves with flux and err values = 0.0.
       
    Parameters
    ----------
    oid: Int
        objectid
    band: string
        photometric band name
    label: string
        value for ML algorithms which details what kind of object this is
        
    Returns
    --------
    zerosingle: dictionary with light curve info
        
    """
    #randomly choose some times during the WISE survey
    #these will all get fleshed out in the section on making uniform length time arrays
    #so the specifics are not important now
    timelist = [55230.0,57054.0, 57247.0, 57977.0, 58707.0]  
    
    #make a dictionary to hold the light curve
    zerosingle = {"objectid": oid, "label": label, "band": band, "time": timelist, 
                  "flux": np.zeros(len(timelist)), "err":np.zeros(len(timelist))}
    
    return zerosingle
```

```{code-cell} ipython3
def missingdata_to_zeros(df_lc):
    """
    Convert mising data into zero fluxes and uncertainties
       
    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info

    Returns
    --------
    df_lc: MultiIndexDFObject with all  light curves
        
    """
    #what is the full set of unique band names?
    full_bandname = df_lc.band.unique()

    #setup a list to store empty light curves
    zerosingle_list = [] 

    #for each object in each band
    for oid , singleoid in df_lc.groupby("objectid"):
                                  
        #this is the list of bandnames for that object                                
        oid_bandname = singleoid.band.unique()
    
        #figure out which bands are missing
        missing = list(set(full_bandname).difference(oid_bandname))
    
        #if it is not the complete list, ie some bandnames are missing:                            
        if len(missing) > 0:
    
            #make new dataframe for this object with zero flux and err values
            for band in missing:
                label = str(singleoid.label.unique().squeeze())
                zerosingle = make_zero_light_curve(oid, band, label)
                #keep track of these empty light curces in a list
                zerosingle_list.append(zerosingle)
    
    #turn the empty light curves into a dataframe
    df_empty = pd.DataFrame(zerosingle_list)
    # df_empty has one row per dict. time,flux, and err columns store arrays.
    # "explode" the dataframe to get one row per light curve point. time, flux, and err columns will now store floats.
    df_empty = df_empty.explode(["time", "flux","err"], ignore_index=True)
    df_empty = df_empty.astype({col: "float" for col in ["time", "flux", "err"]})


    #now put the empty light curves back together with the main light curve dataframe
    zeros_df_lc = pd.concat([df_lc, df_empty])

    return(zeros_df_lc)
```

```{code-cell} ipython3
#this function is not currently needed for the notebook, but I would like to keep it around for potential later testing

import itertools

def calc_nobjects_per_band_combo(df_lc):
    all_bands = df_lc.band.unique()
    object_bands = df_lc.groupby("objectid").band.aggregate(lambda x: set(x))

    band_combos = []
    for l in range(1, len(all_bands) + 1):
        band_combos.extend(set(bands) for bands in itertools.combinations(all_bands, l))

    print(band_combos)
    band_combos_nobjects = [len(object_bands.loc[(band_combo - object_bands) == set()].index) for band_combo in band_combos]

    band_combos_df = pd.DataFrame({"bands": band_combos, "nobjects": band_combos_nobjects})
    band_combos_df = band_combos_df.sort_values("nobjects", ascending=False)

    return band_combos_df

band_combos_df = calc_nobjects_per_band_combo(drop_df_lc)
```

```{code-cell} ipython3
def missingdata_drop_bands(df_lc, bands_to_keep, verbose=False):
    # drop all rows where 'band' is not in 'bands_to_keep'
    bands_to_drop = set(df_lc.band.unique()) - set(bands_to_keep)
    clean_df = df_lc.loc[~df_lc.band.isin(bands_to_drop)]

    if verbose:
        #how many objects did we start with?
        print(len(clean_df.objectid.unique()), "n objects before removing missing band data")

    # keep only objects with observations in all remaining bands
    # first, get a boolean series indexed by objectid
    has_all_bands = clean_df.groupby('objectid').band.aggregate(lambda x: set(x) == set(bands_to_keep))
    # extract the objectids that are 'True'
    objectids_to_keep = has_all_bands[has_all_bands].index
    # keep only these objects
    clean_df = clean_df.loc[clean_df.objectid.isin(objectids_to_keep)]

    if verbose:
        # How many objects are left?
        print(len(clean_df.objectid.unique()), "n objects after removing missing band data")

    return clean_df


```

```{code-cell} ipython3
#choose what to do with missing data...
#df_lc = missingdata_to_zeros(df_lc)
bands_to_keep = ['W1','W2','panstarrs g','panstarrs i', 'panstarrs r','panstarrs y','panstarrs z','zg','zr']
test = missingdata_drop_bands(df_lc, bands_to_keep, verbose = True)
```

### 2.6  Make all objects and bands have identical time arrays (uniform length and spacing)

It is very hard to find time-domain ML algorithms which can work with non uniform length datasets. Therefore we make the light curves uniform by interpolating using KNN from scikit-learn which fills in the uniform length arrays with a final frequency chosen by the user.  We choose KNN as very straightforward method. This function also shows the framework in case the user wants to choose a different scikit-learn function to do the interpolation.  Another natural choice would be to use gaussian processes (GP) to do the interpolation, but this is not a good solution for our task because the flux values go to zero at times before and after the observations.  Because we include the entire time array from beginning of the first mission to end of the last mission, most individual bands require interpolation before and after their particular observations.  In other words, our light curves span the entire range from 2010 with the start of panstarrs and WISE to the most recent ZTF data release (at least 2023), even though most individual missions do not cover that full range of time.

It is important to choose the frequency over which the data is interpolated wisely.  Experimentation with treating this variable like a hyperparam and testing sktime algorithms shows slightly higher accuracy values for a suite of algorithms for a frequency of one interpolated observation per 60 days.

```{code-cell} ipython3
#what does the dataframe look like at this point in the code?
df_lc
```

```{code-cell} ipython3
#this cell takes 13seconds to run on the sample of 458 sources
def uniform_length_spacing(df_lc, final_freq_interpol, include_plot = True):
    """
    Make all light curves have the same length and equal spacing
       
    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info

    final_freq_interpol: int 
        timescale of interpolation in units of days
        
    include_plot: bool
        will make an example plot of the interpolated light curve of a single object
        
    Returns
    --------
    df_interpol: MultiIndexDFObject with interpolated light curves
        
    """
    #make a time array with the minimum and maximum of all light curves in the sample
    x_interpol = np.arange(df_lc.time.min(), df_lc.time.max(), final_freq_interpol)
    x_interpol = x_interpol.reshape(-1, 1) # needed for sklearn
    lc_interpol = []  # list to store interpolated light curves

    #look at each object in each band
    for (band,oid) , singleband_oid in df_lc.groupby(["band", "objectid"]):
        #singleband_oid is now a dataframe with just one object and one band
        X = np.array(singleband_oid["time"]).reshape(-1, 1)
        y = np.array(singleband_oid["flux"])
        dy = np.array(singleband_oid["err"])

        #could imagine using GP to make the arrays equal length and spacing
        #however this sends the flux values to zero at the beginning and end of 
        #the arrays if there is time without observations.  This is not ideal
        #because it significantly changes the shape of the light curves.
        
        #kernel = 1.0 * RBF(length_scale=30)
        #gp = GaussianProcessRegressor(kernel=kernel, alpha=dy**2, normalize_y = False)
        #gp.fit(X, y)
        #mean_prediction,std_prediction = gp.predict(x_interpol, return_std=True)

        #try KNN
        KNN = KNeighborsRegressor(n_neighbors = 3)
        KNN.fit(X, y)
        mean_prediction = KNN.predict(x_interpol)
        
        #KNN doesnt output an uncertainty array, so make our own:
        #an array of the same length as mean_prediction
        #having values equal to the mean of the original uncertainty array
        err = np.full_like(mean_prediction, singleband_oid.err.mean())  
        
        #get these values into the dataframe
        # append the results as a dict. the list will be converted to a dataframe later.
        lc_interpol.append(
            {"objectid": oid, "label": str(singleband_oid.label.unique().squeeze()), "band": band, "time": x_interpol.reshape(-1), 
             "flux": mean_prediction, "err": err}
        )
    
        if include_plot:
            #see what this looks like on just a single light curve for now
            if (band == 'zr') and (oid == 9) :  
                #see if this looks reasonable
                plt.errorbar(X,y,dy,linestyle="None",color="tab:blue",marker=".")
                plt.plot(x_interpol, mean_prediction, label="Mean prediction")
                plt.legend()
                plt.xlabel("time")
                plt.ylabel("flux")
                _ = plt.title("KNN regression")
        
        
    # create a dataframe of the interpolated light curves
    df_interpol = pd.DataFrame(lc_interpol)
    return df_interpol
```

```{code-cell} ipython3
#change this to change the frequency of the time array
final_freq_interpol = 60  #this is the timescale of interpolation in units of days

df_interpol = uniform_length_spacing(df_lc, final_freq_interpol, include_plot = True )

# df_lc_interpol has one row per dict in lc_interpol. time and flux columns store arrays.
# "explode" the dataframe to get one row per light curve point. time and flux columns will now store floats.
df_lc = df_interpol.explode(["time", "flux","err"], ignore_index=True)
df_lc = df_lc.astype({col: "float" for col in ["time", "flux", "err"]})
```

```{code-cell} ipython3
df_lc
```

### 2.7  Restructure dataframe 
- Make columns have band names in them and then remove band from the index
- pivot the dataframe so that SKTIME understands its format
- this will put it in the format expected by sktime

```{code-cell} ipython3
#keep some columns out of the mix when doing the pivot by bandname
#set them as indices and they won't get pivoted into
df_lc = df_lc.set_index(["objectid", "label", "time"])

#attach bandname to all the fluxes and uncertainties
df_lc = df_lc.pivot(columns = "band")

#rename the columns
df_lc.columns = ["_".join(col) for col in df_lc.columns.values]

#many of these flux columns still have a space in them from the bandnames, 
#convert that space to underscore
df_lc.columns = df_lc.columns.str.replace(' ', '_') 

#and get rid of that index to cleanup
df_lc = df_lc.reset_index()  
```

```{code-cell} ipython3
#look at a single object to see what this array looks like
ob_of_interest = 4
singleob = df_lc[df_lc['objectid'] == ob_of_interest]
singleob
```

### 2.8 Normalize 
- this is normalizing across all bands
- think this is the right place to do this, rather than interpolate over time 
    so that the final light curves are normalized since that is the chunk of information 
    which goes into the ML algorithms.
- chose max and not median or mean because there are some objects where the median flux = 0.0
    - if we did this before the interpolating, the median might be a non-zero value
- normalizing is required so that the CLAGN and it's comparison SDSS sample don't have different flux levels.  ML algorithms will easily choose to classify based on overall flux levels, so we want to discourage that by normalizing the fluxes.


Idea here is that we normalize across each object.  So the algorithms will know, for example, that within one object W1 will be brighter than ZTF bands but from one object to the next, it will not know that one is brighter than the other.

```{code-cell} ipython3
def local_normalization(df_lc, norm_column = "flux_W1"):
    """
    normalize each individual light curve
       
    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info

    norm_column: str
        name of the flux column in df_lc by which the light curves should be normalized
        ie., which band should be the normalizing band?
        
    Returns
    --------
    df_lc: MultiIndexDFObject with interpolated light curves
        
    """
    #make a copy to not alter the original df_lc inside of this function
    norm_df_lc = df_lc
    
    # make a new column with max_r_flux for each objectid
    norm_df_lc['max_W1'] = norm_df_lc.groupby('objectid', sort=False)[norm_column].transform('max')

    #figure out which columns in the dataframe are flux columns
    flux_cols = [col for col in norm_df_lc.columns if 'flux' in col]

    # make new normalized flux columns for all fluxes
    norm_df_lc[flux_cols] = norm_df_lc[flux_cols].div(norm_df_lc['max_W1'], axis=0)

    #now drop max_W1 as a column so it doesn't get included as a variable in multivariate analysis
    norm_df_lc.drop(columns = ['max_W1'], inplace = True)

    return norm_df_lc
```

```{code-cell} ipython3
#normalize by W1 band
df_lc = local_normalization(df_lc, norm_column = "flux_W1")
```

### 2.9 Cleanup
- Make [datetime](https://docs.python.org/3/library/datetime.html#module-datetime) column
    - SKTime wants a datetime column
- Save dataframe
- Make into multi-index
      - SKtime wants multi-index

```{code-cell} ipython3
#need to convert df_lc time into datetime
mjd = df_lc.time

#convert to JD
jd = mjd_to_jd(mjd)

#convert to individual components
t = Time(jd, format = 'jd' )

#t.datetime is now an array of type datetime
#make it a column in the dataframe
df_lc['datetime'] = t.datetime
```

```{code-cell} ipython3
#save this dataframe to use for the ML below so we don't have to make it every time
parquet_savename = 'output/df_lc_ML.parquet'
#df_lc.to_parquet(parquet_savename)
#print("file saved!")
```

```{code-cell} ipython3
#give this dataframe a multiindex
df_lc = df_lc.set_index(["objectid", "label", "datetime"])
```

## 3. Prep for ML algorithms

```{code-cell} ipython3
# could load a previously saved file in order to plot
#parquet_loadname = 'output/df_lc_ML.parquet'
#df_lc = MultiIndexDFObject()
#df_lc.data = pd.read_parquet(parquet_loadname)
#print("file loaded!")
```

```{code-cell} ipython3
#try dropping the uncertainty columns as variables for sktime
df_lc.drop(columns = ['err_panstarrs_g',	'err_panstarrs_i',	'err_panstarrs_r',	'err_panstarrs_y',	
                      'err_panstarrs_z',	'err_W1',	'err_W2',	'err_zg',	'err_zr'], inplace = True)

#drop also the time column because time shouldn't be a feature
df_lc.drop(columns = ['time'],inplace = True)
```

```{code-cell} ipython3
#what does the dataframe look like now?
df_lc
```

### 3.1 Train test split 
- Because thre are uneven numbers of each type (many more SDSS than CLAGN), we want to make sure to stratify evenly by type
- Random split

```{code-cell} ipython3
#y is defined to be the labels
y = df_lc.droplevel('datetime').index.unique().get_level_values('label').to_series()

#want a stratified split based on label
test_size = 0.25
train_ix, test_ix = train_test_split(df_lc.index.levels[0], stratify = y, shuffle = True, random_state = 43, test_size = test_size)

train_df = df_lc.loc[train_ix]  
test_df = df_lc.loc[test_ix]   
```

```{code-cell} ipython3
print(train_df.groupby([ "objectid"]).ngroups, "n groups in train sample")
print(test_df.groupby(["objectid"]).ngroups, "n groups in test sample")
```

```{code-cell} ipython3
#plot to show how many of each type of object in the test dataset
plt.figure(figsize=(6,4))
plt.title("Objects in the Test dataset")
h = plt.hist(test_df.droplevel('datetime').index.unique().get_level_values('label').to_series(),histtype='stepfilled',orientation='horizontal')
```

```{code-cell} ipython3
#divide the dataframe into X and y for ML algorithms 

#X is the multiindex light curve without the labels
X_train  = train_df.droplevel('label')
X_test = test_df.droplevel('label')

#y are the labels, should be a series 
y_train = train_df.droplevel('datetime').index.unique().get_level_values('label').to_series()
y_test = test_df.droplevel('datetime').index.unique().get_level_values('label').to_series()
```

```{code-cell} ipython3
print(X_train.groupby([ "objectid"]).ngroups, "n groups in train sample")
print(X_test.groupby(["objectid"]).ngroups, "n groups in test sample")
```

```{code-cell} ipython3
X_train
```

## 4. Run sktime algorithms on the light curves

We choose to use [sktime](https://www.sktime.net/en/stable/index.html) algorithms beacuse it is a library of many algorithms specifically tailored to time series datasets.  It is based on the sklearn library so syntax is familiar to many users.

Types of classifiers are listed [here](https://www.sktime.net/en/stable/api_reference/classification.html).

This notebook will first show you an example of a single algorithm classifier. Then it will show how to write a for loop over a bunch of classifiers while outputting some metrics of accuracy.

+++

### 4.1 Check that the data types are ok for sktime
This test needs to pass in order for sktime to run

```{code-cell} ipython3
#ask sktime if it likes the data type of X_train

check_is_mtype(X_train, mtype="pd-multiindex", scitype="Panel", return_metadata=True)
```

```{code-cell} ipython3
#show the list of all possible classifiers that work with multivariate data
#all_tags(estimator_types = 'classifier')
#classifiers = all_estimators("classifier", filter_tags={'capability:multivariate':True})
#classifiers
```

### 4.1 A single Classifier

```{code-cell} ipython3
%%time

#setup the classifier
clf = Arsenal(time_limit_in_minutes=1, n_jobs = -1)

#fit the classifier on the training dataset
clf.fit(X_train, y_train)

#make predictions on the test dataset using the trained model 
y_pred = clf.predict(X_test)

print(f"Accuracy of Random Interval Classifier: {accuracy_score(y_test, y_pred)}\n", flush=True)

#plot a confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
    
plt.show()
```

### 4.2 Loop over a bunch of classifiers

Our method is to do a cursory check of a bunch of classifiers and then later drill down deeper on anything with good initial results.  We choose to run a loop over ~10 classifiers that seem promising and check the accuracy scores for each one.  Any classifier with a promising accuracy score could then be followed up with detailed hyperparameter tuning, or potentially with considering other classifiers in that same type.

```{raw-cell}
%%time
#This cell is currently not being run because it takes a while so is not good for testing/debugging

#which classifiers are we interestd in
#roughly one from each type of classifier

names = ["Arsenal",                     #kernel based
        "RocektClassifier",             #kernel based
        "CanonicalIntervalForest",      #interval based
        "HIVECOTEV2",                   #hybrid
#        "CNNClassifier",               #Deep Learning  - **requires tensorflow which is giving import errors
#        "WeightedEnsembleClassifier",   #Ensemble - **maybe use in the future if we find good options
        "IndividualTDE",               #Dictionary-based
        "KNeighborsTimeSeriesClassifier", #Distance Based
        "RandomIntervalClassifier",     #Feature based
        "Catch22Classifier",            #Feature based
        "ShapeletTransformClassifier"   #Shapelet based
        "DummyClassifier"]             #Dummy - ignores input

#for those with an impossible time limit, how long to let them run for before cutting off
nmins = 3

#these could certainly be more tailored
classifier_call = [Arsenal(time_limit_in_minutes=nmins, n_jobs = -1), 
                  RocketClassifier(num_kernels=2000),
                  CanonicalIntervalForest(n_jobs = -1),
                  HIVECOTEV2(time_limit_in_minutes=nmins, n_jobs = -1),
#                  CNNClassifier(),
#                  WeightedEnsembleClassifier(),
                  IndividualTDE(n_jobs=-1),
                  KNeighborsTimeSeriesClassifier(n_jobs = -1),
                  RandomIntervalClassifier(n_intervals = 20, n_jobs = -1, random_state = 43),
                  Catch22Classifier(outlier_norm = True, n_jobs = -1, random_state = 43),
                  ShapeletTransformClassifier(time_limit_in_minutes=nmins,n_jobs = -1),
                  DummyClassifier()]

#setup to store the accuracy scores
accscore_dict = {}

# iterate over classifiers
for name, clf in tqdm(zip(names, classifier_call)):
    #fit the classifier
    clf.fit(X_train, y_train)
    
    #make predictions on the test dataset
    y_pred = clf.predict(X_test)

    #calculate and track accuracy score
    accscore = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {name} classifier: {accscore}\n", flush=True)
    accscore_dict[name] = accscore
    
    #plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
    disp.plot()
    plt.show()
```

```{raw-cell}
#show the summary of the algorithms used and their accuracy score
accscore_dict
```

```{code-cell} ipython3
#save statistics from these runs

# Serialize data into file:
#json.dump( accscore_dict, open( "output/accscore.json", 'w' ) )
#json.dump( completeness_dict, open( "output/completeness.json", 'w' ) )
#json.dump( homogeneity_dict, open( "output/homogeneity.json", 'w' ) )

# Read data from file:
#accscore_dict = json.load( open( "output/accscore.json") )
```

## 5.0 Create a candidate list 
Lets assume we now have a classifier which can accurately differentiate CLAGN from SDSS QSOs.  Next, we would like to use that classifier on our favorite unlabeled sample to identify CLAGN candidates.  To do this, we need to:
- read in a dataframe of our new sample
- get that dataset in the same format as what was fed into the classifiers
- run clf.predict() on that re-formatted dataset
- retrace those objectids to an ra & dec
- write an observing proposal (ok, you have to do that one yourself)

```{code-cell} ipython3
#get dataset in same format as what was run through sktime
#This is not exactly the same as re-running the whole notebook on a different sample,
#but it is pretty close.  We don't need to do all of the same cleaning because some of that 
#was to appease sktime in training the algorithms.

def prepare_mysample_sktime(path_to_sample):
    """
    run your own favorite sample through all the necessary data prep to be able to use the sktime trained classifer on it
    
    Parameters
    ----------
    path_to_sample: str
        the location of your favorite sample

     Returns
    --------
    X_mysample: MultiIndexDFObject with interpolated light curves
        ready to be used by sktime classifier
        
    """

    #read in a dataframe of our new sample:
    # we are going to cheat here and use the same file as we used for input to the above, but you should
    # replace this with your favorite sample run through the light_curve_generator in this same repo.

    my_sample = pd.read_parquet(path_to_sample)

    #get rid of indices set in the light curve code and reset them as needed before sktime algorithms
    my_sample = my_sample.reset_index()  

    # get rid of some of the bands that don't have enough data for all the sources
    #CLAGN are generall fainter targets, and therefore mostly not found in datasets like TESS & K2
    bands_to_drop = ["IceCube", "TESS", "FERMIGTRIG", "K2"]
    my_sample = my_sample[~my_sample["band"].isin(bands_to_drop)]

    #drop rows which have Nans
    my_sample.dropna(inplace = True, axis = 0)

    #remove outliers
    sigmaclip_value = 10.0
    my_sample = sigmaclip_lightcurves(my_sample, sigmaclip_value, include_plot = False, verbose= False)

    #remove objects without W1 fluxes
    my_sample = remove_objects_without_band(my_sample, 'W1', verbose=False)

    #remove incomplete data
    threshold_too_few = 3
    my_sample = remove_incomplete_data(my_sample, threshold_too_few, verbose = False)

    #drop missing bands
    my_sample = missingdata_drop_bands(my_sample, verbose = False)

    #make arrays have uniform length and spacing
    final_freq_interpol = 60  #this is the timescale of interpolation in units of days
    df_interpol = uniform_length_spacing(my_sample, final_freq_interpol, include_plot = False )
    my_sample = df_interpol.explode(["time", "flux","err"], ignore_index=True)
    my_sample = my_sample.astype({col: "float" for col in ["time", "flux", "err"]})

    #reformat
    my_sample = my_sample.set_index(["objectid", "label", "time"])
    my_sample = my_sample.pivot(columns = "band")
    my_sample.columns = ["_".join(col) for col in my_sample.columns.values]
    my_sample.columns = my_sample.columns.str.replace(' ', '_') 
    my_sample = my_sample.reset_index()  

    #normalize
    my_sample = local_normalization(my_sample, norm_column = "flux_W1")

    #make datetime column
    mjd = my_sample.time
    jd = mjd_to_jd(mjd)
    t = Time(jd, format = 'jd' )
    my_sample['datetime'] = t.datetime

    #set index expected by sktime
    my_sample = my_sample.set_index(["objectid", "label", "datetime"])

    #drop the uncertainty columns 
    my_sample.drop(columns = ['err_panstarrs_g',	'err_panstarrs_i',	'err_panstarrs_r',	'err_panstarrs_y',	
                          'err_panstarrs_z',	'err_W1',	'err_W2',	'err_zg',	'err_zr'], inplace = True)

    #drop also the time column 
    my_sample.drop(columns = ['time'],inplace = True)

    #make X 
    X_mysample  = my_sample.droplevel('label')

    return X_mysample
```

```{code-cell} ipython3
%%time
#read in data and prepare it for use
path_to_sample = "./data/df_lc_458sample.parquet"
X_mysample = prepare_mysample_sktime(path_to_sample)

#what does this look like to make sure we are on track
X_mysample
```

```{code-cell} ipython3
%%time
#use the trained sktime algorithm to make predictions on the test dataset
y_mysample = clf.predict(X_mysample)
```

```{code-cell} ipython3
len(y_mysample)
```

```{code-cell} ipython3
#associate these predicted CLAGN with RA & Dec

#I might have removed some objects in my cleaning, so need to first associate objectid with each of y_mysample
#this doesn't work, but is what I wanted to work
X_mysample["predicted_labels"] = pd.Series(y_mysample)

#now theoretically X_mysample has both objectid and predicted_labels
#maybe drop the flux & time columns for ease of use and makes it nice and small
flux_cols = [col for col in norm_df_lc.columns if 'flux' in col]
candidate_CLAGN = X_mysample.drop(columns = flux_cols)
candidate_CLAGN.drop(columns = ['time'], inplace = True)

#if I am only interested in the CLAGN, could drop anything with label = SDSS
querystring = 'label == "SDSS"'
candidate_CLAGN = band_lc.drop(candidate_CLAGN.query(querystring ).index)

# need to read in the ecsv file that is sample_table
#what format is this table again?

#then will need to join candidate_CLAGN with sample_table along objectid
```

```{code-cell} ipython3
type(sample_table)
```

## References:
Markus Löning, Anthony Bagnall, Sajaysurya Ganesh, Viktor Kazakov, Jason Lines, Franz Király (2019): “sktime: A Unified Interface for Machine Learning with Time Series”
Markus Löning, Tony Bagnall, Sajaysurya Ganesh, George Oastler, Jason Lines, ViktorKaz, …, Aadesh Deshmukh (2020). sktime/sktime. Zenodo. http://doi.org/10.5281/zenodo.3749000

```{code-cell} ipython3

```
