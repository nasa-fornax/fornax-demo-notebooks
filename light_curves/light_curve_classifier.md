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
```

```{code-cell} ipython3
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from astropy.time import Time
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
# has known CLAGN & random SDSS small sample of 458 targets, all bands
#https://drive.google.com/file/d/13RiPODiz2kI8j1OKpP1vfh6ByIUNsKEz/view?usp=share_link

gdd.download_file_from_google_drive(file_id='13RiPODiz2kI8j1OKpP1vfh6ByIUNsKEz',
                                    dest_path='./data/df_lc_458sample.parquet',
                                    unzip=True)

df_lc = pd.read_parquet("./data/df_lc_458sample.parquet")

#get rid of indices set in the light curve code and reset them as needed before sktime algorithms
df_lc = df_lc.reset_index()  
```

```{code-cell} ipython3
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
df_lc['label'] = df_lc.label.str.replace('Lyu 21', 'CLAGN')
df_lc['label'] = df_lc.label.str.replace('Hon 22', 'CLAGN')
df_lc['label'] = df_lc.label.str.replace('Sheng 20', 'CLAGN')
df_lc['label'] = df_lc.label.str.replace('MacLeod 19', 'CLAGN')
df_lc['label'] = df_lc.label.str.replace('Green 22', 'CLAGN')
df_lc['label'] = df_lc.label.str.replace('Lopez-Navas 22', 'CLAGN')
```

### 2.3 Data Visualization
- can we see any trends by examining plots of a subset of the data?

```{code-cell} ipython3

#plot a single band for all objects
band_of_interest = 'w1'
W1_band = df_lc[df_lc['band'] == band_of_interest]
W1_band.set_index('time', inplace = True)

#drop some objects to try to clear up plot
querystring1 = 'objectid < 100'
querystring2 = 'objectid > 300'
W1_band = W1_band.drop(W1_band.query(querystring1 ).index)
W1_band = W1_band.drop(W1_band.query(querystring2 ).index)

#quick normalization for plotting
#we normalize for real after cleaning the data
# make a new column with max_r_flux for each objectid
W1_band['mean_W1'] = W1_band.groupby('objectid', sort=False)["flux"].transform('mean')
W1_band['sigma_W1'] = W1_band.groupby('objectid', sort=False)["flux"].transform('std')

W1_band['flux'] = (W1_band['flux'] - W1_band['mean_W1']).div(W1_band['sigma_W1'], axis=0)

#want to have two different sets so I can color code
clagn_df = W1_band[W1_band['label'] == 'CLAGN']
sdss_df = W1_band[W1_band['label'] == 'SDSS']
print(clagn_df.groupby(["objectid"]).ngroups, "n objects CLAGN ")
print(sdss_df.groupby(["objectid"]).ngroups, "n objects SDSS ")

#groupy objectid, plot
fig, ax = plt.subplots(figsize=(8,6))
lc_sdss = sdss_df.groupby(['objectid'])['flux'].plot(kind='line', ax=ax, color = 'gray', label = 'SDSS')
lc_clagn = clagn_df.groupby(['objectid'])['flux'].plot(kind='line', ax=ax, color = 'orange', label = 'CLAGN')

legend_elements = [Line2D([0], [0], color='orange', lw=4, label='CLAGN'),
                   Line2D([0], [0], color='gray', lw=4, label='SDSS')]
ax.legend(handles=legend_elements, loc='best')

ax.set_ylabel('Normalized Flux')
plt.title("W1 light curves")
```

### 2.4 Clean the dataset of unwanted data
"unwanted" includes:
- NaNs
- zero flux
- outliers in uncertainty
- objects with no measurements in WISE W1 band
- objects with not enough flux measurements to make a good light curve

```{code-cell} ipython3
def sigmaclip_lightcurves(df_lc, sigmaclip_value = 10.0, include_plot = False):
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

    end_len = len(clipped_df_lc.index)
    fraction = (start_len - end_len) / start_len
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
def remove_incomplete_data(df_lc, threshold_too_few = 3):
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
    print(df_lc.groupby(["band", "objectid"]).ngroups, "n groups before")

    #use pandas .filter to remove small groups
    complete_df_lc = df_lc.groupby(["band", "objectid"]).filter(lambda x: len(x) > threshold_too_few)

    #how many groups do we have after culling?
    print(complete_df_lc.groupby(["band", "objectid"]).ngroups, "n groups after")

    return complete_df_lc
    
```

```{code-cell} ipython3
#drop rows which have Nans
df_lc.dropna(inplace = True, axis = 0)

#drop rows with zero flux
querystring = 'flux < 0.000001'
df_lc = df_lc.drop(df_lc.query(querystring).index)

#remove outliers
#This is a tricky job because we want to keep astrophysical sources that are 
#variable objects, but remove instrumental noise and CR (ground based).
sigmaclip_value = 10.0
df_lc = sigmaclip_lightcurves(df_lc, sigmaclip_value, include_plot = True)

#remove objects without W1 fluxes
#We want to normalize all light curves by W1, so we neeed to remove those 
#without W1 fluxes as there will be nothing to normalize those light curves 
#with and we don't want to have un-normalized data or data that has been 
#normalized by a different band.  
df_lc = remove_objects_without_band(df_lc, 'w1', verbose=True)

#remove incomplete data
#Some bands in some objects have only a few datapoints. Three data points 
#is not large enough for KNN interpolation, so we will consider any array 
#with fewer than 4 photometry points to be incomplete data.  Another way 
#of saying this is that we choose to remove those light curves with 3 or 
#fewer data points.
threshold_too_few = 3
df_lc = remove_incomplete_data(df_lc, threshold_too_few)
    
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
def missingdata_drop_bands(df_lc, verbose = False):
    """
    Drop bands with the most missing data and objects without all remaining bands so that there is no missing data going forward.
       
    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info
    
    verbose: bool
    
    Returns
    --------
    df_lc: MultiIndexDFObject with all  light curves
        
    """

    #require that all objects have a curated list of bands

    #first drop the bands not included from all objects
    #these are bands with significantly fewer data points in them than the other bands
    #ie., fewer objects have these bands
    bands_to_drop = ['zi', 'Gaia g', 'Gaia bp', 'Gaia rp']
    drop_df_lc = df_lc[~df_lc["band"].isin(bands_to_drop)]
    
    #now a list of all remaining bands that we want to keep
    bands_to_keep = drop_df_lc.band.unique()
    
    if verbose:
        #how many objects did we start with?
        print(drop_df_lc.groupby(["objectid"]).ngroups, "n objects before removing missing band data")

    # Identify objects with all bands that we want to keep
    complete_objects = drop_df_lc.groupby('objectid')['band'].apply(lambda x: set(x) == set(bands_to_keep))

    # Filter the DataFrame based on complete objects
    filter_df_lc = drop_df_lc[drop_df_lc['objectid'].isin(complete_objects[complete_objects].index)]

    if verbose:
        # How many objects are left?
        print(filter_df_lc.groupby(["objectid"]).ngroups, "n objects after removing missing band data")

    return(filter_df_lc)
```

```{code-cell} ipython3
#choose what to do with missing data...
#df_lc = missingdata_to_zeros(df_lc)
df_lc = missingdata_drop_bands(df_lc, verbose = True)
```

### 2.6  Make all objects and bands have identical time arrays (uniform length and spacing)

It is very hard to find time-domain ML algorithms which can handle non uniform length datasets. Therefore we make them uniform by interpolating using KNN from scikit-learn which fills in the uniform length arrays with a final frequency chosen by the user.  We choose KNN as very straightforward. This functional also shows the framework in case the user wants to choose a different scikit-learn function to do the interpolation.  Another natural choice would be to use gaussian processes to do the interpolation, but this is not a good solution for our task because the flux values go to zero at times before and after the observations.  Because we include the entire time array from beginning of the first band to end of the second band, most individual bands require interpolation before and after their particular observations.  In other words, our light curves span

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
                #plt.fill_between(
                #    x_interpol.ravel(),
                #    mean_prediction - 1.96 * std_prediction,
                #    mean_prediction + 1.96 * std_prediction,
                #    color="tab:orange",
                #    alpha=0.5,
                #    label=r"95% confidence interval",
                #)
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
#experimentation with treating this variable like a hyperparam and testing
#sktime algorithms shows slightly higher accuracy values for a suite of algorithms
#for a frequency of 60 days.  
final_freq_interpol = 60  #this is the timescale of interpolation in units of days

df_interpol = uniform_length_spacing(df_lc, final_freq_interpol )

# df_lc_interpol has one row per dict in lc_interpol. time and flux columns store arrays.
# "explode" the dataframe to get one row per light curve point. time and flux columns will now store floats.
df_lc = df_interpol.explode(["time", "flux","err"], ignore_index=True)
df_lc = df_lc.astype({col: "float" for col in ["time", "flux", "err"]})
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
- normalizing is required so that the CLAGN and it's comparison SDSS sample don't have different flux levels.


Idea here is that we normalize across each object.  So the algorithms will know, for example, that within one object W1 will be brighter than ZTF bands but from one object to the next, it will not know that one is brighter than the other.

```{code-cell} ipython3
# make a new column with max_r_flux for each objectid
df_lc['max_W1'] = df_lc.groupby('objectid', sort=False)["flux_w1"].transform('max')

#figure out which columns in the dataframe are flux columns
flux_cols = [col for col in df_lc.columns if 'flux' in col]

# make new normalized flux columns for all fluxes
df_lc[flux_cols] = df_lc[flux_cols].div(df_lc['max_W1'], axis=0)

#now drop max_W1 as a column so it doesn't get included as a variable in multivariate analysis
df_lc.drop(columns = ['max_W1'], inplace = True)
```

### 2.9 Cleanup
- Make datetime column
https://docs.python.org/3/library/datetime.html#module-datetime
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
df_lc.to_parquet(parquet_savename)
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
                      'err_panstarrs_z',	'err_w1',	'err_w2',	'err_zg',	'err_zr'], inplace = True)

#drop also the time column because that shouldn't be a feature
df_lc.drop(columns = ['time'],inplace = True)
```

### 3.0 Consider data augmentation

1. https://arxiv.org/pdf/1811.08295.pdf which has the following github

    - https://github.com/gioramponi/GAN_Time_Series/tree/master
    - not easily usable
2. https://arxiv.org/pdf/2205.06758.pdf

3. ChatGPT - give multiindex df function and it will give a starting point for augmenting


Worried that augmenting noisy data just makes more noise

+++

### 3.1 Train test split 
- Because thre are uneven numbers of each type (many more SDSS than CLAGN), we want to make sure to stratify evenly by type
- Random split

```{code-cell} ipython3
#what does the dataframe look like now?
df_lc
```

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
y
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

## 4. Run sktime algorithms on the light curves

We choose to use [sktime](https://www.sktime.net/en/stable/index.html) algorithms beacuse it is a library of many algorithms specifically tailored to time series datasets.  It is based on the sklearn library so syntax is familiar to many users.

Types of classifiers are listed [here](https://www.sktime.net/en/stable/api_reference/classification.html).

This notebook will invert the actual workflow and show you a single example of the algorithm which best fits the data and has the most accurate classifier. Then it will show how to write a for loop over a bunch of classifiers before narrowing it down to the most accurate.

+++

### 4.1 Check that the data types are ok for sktime

```{code-cell} ipython3
#ask sktime if it likes the data type of X_train
from sktime.datatypes import check_is_mtype

check_is_mtype(X_train, mtype="pd-multiindex", scitype="Panel", return_metadata=True)
#check_is_mtype(X_test, mtype="pd-multiindex", scitype="Panel", return_metadata=True)

#This test needs to pass in order for sktime to run
```

```{code-cell} ipython3
#what is the list of all possible classifiers that work with multivariate data
#all_tags(estimator_types = 'classifier')
#classifiers = all_estimators("classifier", filter_tags={'capability:multivariate':True})
#classifiers
```

### 4.1 A single Classifier
See section 4.3 for how we landed with this algorithm

```{code-cell} ipython3
%%time
#looks like RandomIntervalClassifier is performing the best for the CLAGN (not for the SDSS)

#setup the classifier
clf = RandomIntervalClassifier(n_intervals = 12, n_jobs = -1, random_state = 43)

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
nmins = 10

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

#just for keeping track, I also tried 
#clf = SignatureClassifier(depth = 2, window_depth = 3, random_state = 43)
#this fails to complete, and is a known limitation of this algorithm.  
```

```{raw-cell}
#show the summary of the algorithms used and their accuracy score
accscore_dict
```

```{raw-cell}
#save statistics from these runs

# Serialize data into file:
json.dump( accscore_dict, open( "output/accscore.json", 'w' ) )
json.dump( completeness_dict, open( "output/completeness.json", 'w' ) )
json.dump( homogeneity_dict, open( "output/homogeneity.json", 'w' ) )

# Read data from file:
#accscore_dict = json.load( open( "output/accscore.json") )
```

## 5.0 Run pyts algorithms on a single band of the light curves
[pyts](https://pyts.readthedocs.io/en/stable/) is a python package for time series classification.

We run univariate classification here with just the W1 WISE band.  This is a change from the multivariate sktime classification above.

+++

### 5.1 Get the data into the correct shape for pyTS

```{code-cell} ipython3
#How many objects are we working with?
#this is repeated here in case sktime is not run above.
print(X_train.groupby([ "objectid"]).ngroups, "n groups in train sample")
print(X_test.groupby(["objectid"]).ngroups, "n groups in test sample")
```

```{code-cell} ipython3
#input to pyTS must be a  numpy.ndarray 
# a 2d array with shape (n_samples, n_timestamps), where the first axis represents the 
# samples and the second axis represents time

#start working on X
X_train_pyts = X_train.reset_index()
X_test_pyts = X_test.reset_index()

# Extract univariate flux values into NumPy arrays and reshape them
X_train_np = X_train_pyts.pivot(index='objectid',  columns='time',values='flux_w1').to_numpy() 
X_test_np = X_test_pyts.pivot(index='objectid', columns='time', values='flux_w1').to_numpy()

#now work on the y
train_df_pyts = train_df.reset_index()
test_df_pyts = test_df.reset_index()

# Extract unique labels for each objectid and convert to a NumPy array
y_train_np = train_df_pyts.groupby('objectid')['label'].first().to_numpy()
y_test_np = test_df_pyts.groupby('objectid')['label'].first().to_numpy()
```

### 5.2 A single classifier

```{code-cell} ipython3
#setup to store the accuracy et al. scores
accscore_dict = {}
MCC_dict ={}
homogeneity_dict = {}
completeness_dict = {}
f1_dict = {}
```

```{code-cell} ipython3
clf = LearningShapelets(random_state=42, tol=0.01)
clf.fit(X_train_np, y_train_np)
clf.score(X_test_np, y_test_np)
y_pred = clf.predict(X_test_np)

#plot confusion matrix
cm = confusion_matrix(y_test_np, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
#calculate and track accuracy score
name = "learningshapelets"
accscore = accuracy_score(y_test, y_pred)
print(f"Accuracy of {name} classifier: {accscore}\n", flush=True)
accscore_dict[name] = accscore
    
MCC = matthews_corrcoef(y_test, y_pred)
print(f"MCC of {name} classifier: {MCC}\n", flush=True)
MCC_dict[name] = MCC
    
completeness = completeness_score(y_test, y_pred)
print(f"Completeness of {name} classifier: {completeness}\n", flush=True)
completeness_dict[name] = completeness
    
homogeneity = homogeneity_score(y_test, y_pred)
print(f"Homogeneity of {name} classifier: {homogeneity}\n", flush=True)
homogeneity_dict[name] = homogeneity
    
f1 = f1_score(y_test, y_pred, average='macro')
print(f"F1 score of {name} classifier: {f1}\n", flush=True)
f1_dict[name] = f1
```

### 5.3 Loop over a bunch of classifiers

```{raw-cell}
%%time
#This cell is currently not being run because it takes a while so is not good for testing/debugging

names = ["KNNDTW","saxvsm","bossvs", "learningshapelets","timeseriesforest"]
         

#these could certainly be more tailored
classifier_call = [KNeighborsClassifier(metric='dtw'), 
                   SAXVSM(window_size=34, sublinear_tf=False, use_idf=False),
                   BOSSVS(window_size=28),
                   LearningShapelets(random_state=43, tol=0.01),
                   TimeSeriesForest(random_state=43)]
                   
#setup to store the accuracy scores
accscore_dict = {}

# iterate over classifiers
for name, clf in tqdm(zip(names, classifier_call)):
    #fit the classifier
    clf.fit(X_train_np, y_train_np)
    
    #make predictions on the test dataset
    y_pred = clf.predict(X_test_np)

    #calculate and track accuracy score
    accscore = accuracy_score(y_test_np, y_pred)
    print(f"Accuracy of {name} classifier: {accscore}\n", flush=True)
    accscore_dict[name] = accscore
    
    #plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
    disp.plot()
    plt.show()
```

```{raw-cell}
#save statistics from these runs

# Serialize data into file:
json.dump( accscore_dict, open( "output/pyts_accscore.json", 'w' ) )
json.dump( completeness_dict, open( "output/pyts_completeness.json", 'w' ) )
json.dump( homogeneity_dict, open( "output/pyts_homogeneity.json", 'w' ) )

# Read data from file:
#accscore_dict = json.load( open( "output/pyts_accscore.json") )
```

## 6.0 Conclusions:  
This classifier can be used to predict CLAGN.  The feature based algorithms do the best jobs of having little to no predicted CLAGN that are truly normal SDSS quasars.  We infer then that if the trained model predicts CLAGN, it is a very good target for follow-up spectroscopy to confirm CLAGN.  However this algorthim will not catch all CLAGN, and will incorrectly labels some CLAGN as being normal SDSS quasars.  THis algorithm can therefore not be used to find a complete sample of CLAGN, but can be used to increase the known sample.

+++

### 6.1 Potential Areas of improvement
- Data is messy
    - ZTF calibration??
- Label inaccuracy is a concern
    - mostly SDSS, 
    - but CLAGN papers all have different selection criteria
- Not enough data on CLAGN
    - limited number of lightcurves
    - consider data augmentation

+++

## References:
Markus Löning, Anthony Bagnall, Sajaysurya Ganesh, Viktor Kazakov, Jason Lines, Franz Király (2019): “sktime: A Unified Interface for Machine Learning with Time Series”
Markus Löning, Tony Bagnall, Sajaysurya Ganesh, George Oastler, Jason Lines, ViktorKaz, …, Aadesh Deshmukh (2020). sktime/sktime. Zenodo. http://doi.org/10.5281/zenodo.3749000

```{code-cell} ipython3

```
