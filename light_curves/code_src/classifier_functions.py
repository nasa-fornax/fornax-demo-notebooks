import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sigmaclip
import itertools
from sklearn.neighbors import KNeighborsRegressor
from astropy.time import Time

# local code imports
from fluxconversions import mjd_to_jd


def sigmaclip_lightcurves(df_lc, sigmaclip_value=10.0, include_plot=False, verbose=False):
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
    # keep track of how many rows this removes
    start_len = len(df_lc.index)

    # setup to collect the outlier thresholds per band to later reject
    nsigmaonmean = {}

    if include_plot:
        # create the figure and axes
        fig, axs = plt.subplots(5, 3, figsize=(12, 12))

        # unpack all the axes subplots
        axe = axs.ravel()

    # for each band
    for count, (bandname, singleband) in enumerate(df_lc.groupby("band")):

        # use scipy sigmaclip to iteratively determine sigma on the dataset
        clippedarr, lower, upper = sigmaclip(
            singleband.err, low=sigmaclip_value, high=sigmaclip_value)

        # store this value for later
        nsigmaonmean[bandname] = upper

        if include_plot:
            # plot distributions and print stddev
            singleband.err.plot(kind='hist', bins=30, subplots=True,
                                ax=axe[count], label=bandname + ' ' + str(upper), legend=True)

    # remove data that are outside the sigmaclip_value
    # make one large querystring joined by "or" for all bands in df_lc
    querystring = " | ".join(
        f'(band == {bandname!r} & err > {cut})' for bandname, cut in nsigmaonmean.items())
    clipped_df_lc = df_lc.drop(df_lc.query(querystring).index)

    # how much data did we remove with this sigma clipping?
    # This should inform our choice of sigmaclip_value.

    if verbose:
        end_len = len(clipped_df_lc.index)
        fraction = ((start_len - end_len) / start_len) * 100.
        print(f"This {sigmaclip_value} sigma clipping removed {fraction}% of the rows in df_lc")

    return clipped_df_lc


def remove_objects_without_band(df_lc, bandname_to_keep="W1", verbose=False):
    # drop objects that do not have data in the band bandname_to_keep
    """
    Drop objects that do not have data in the band bandname_to_keep.  This is needed for normalization.

    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info

    bandname_to_keep: str "w1"

    verbose: should the function provide feedback

    Returns
    --------
    drop_df_lc: MultiIndexDFObject with all  light curves

    """
    # remove objects without bandname_to_keep by filtering the dataframe grouped by objectid
    drop_df_lc = df_lc.groupby('objectid').filter(lambda x: bandname_to_keep in x['band'].values)

    if verbose:
        # if you want to track how many objects were removed by this function...
        dropcount = df_lc.groupby("objectid").ngroups - drop_df_lc.groupby("objectid").ngroups
        print(dropcount, "objects without", bandname_to_keep, " were removed")

    return drop_df_lc


def remove_incomplete_data(df_lc, threshold_too_few=3, verbose=True):
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

    # how many groups do we have before we start
    if verbose:
        print(df_lc.groupby(["band", "objectid"]).ngroups,
              "n groups before removing incomplete data")

    # use pandas .filter to remove small groups
    complete_df_lc = df_lc.groupby(["band", "objectid"]).filter(
        lambda x: len(x) > threshold_too_few)

    # how many groups do we have after culling?
    if verbose:
        print(complete_df_lc.groupby(["band", "objectid"]).ngroups,
              "n groups after removing incomplete data")

    return complete_df_lc


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
    # randomly choose some times during the WISE survey
    # these will all get fleshed out in the section on making uniform length time arrays
    # so the specifics are not important now
    timelist = [55230.0, 57054.0, 57247.0, 57977.0, 58707.0]

    # make a dictionary to hold the light curve
    zerosingle = {"objectid": oid, "label": label, "band": band, "time": timelist,
                  "flux": np.zeros(len(timelist)), "err": np.zeros(len(timelist))}

    return zerosingle


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
    # what is the full set of unique band names?
    full_bandname = df_lc.band.unique()

    # setup a list to store empty light curves
    zerosingle_list = []

    # for each object in each band
    for oid, singleoid in df_lc.groupby("objectid"):

        # this is the list of bandnames for that object
        oid_bandname = singleoid.band.unique()

        # figure out which bands are missing
        missing = list(set(full_bandname).difference(oid_bandname))

        # if it is not the complete list, ie some bandnames are missing:
        if len(missing) > 0:

            # make new dataframe for this object with zero flux and err values
            for band in missing:
                label = str(singleoid.label.unique().squeeze())
                zerosingle = make_zero_light_curve(oid, band, label)
                # keep track of these empty light curces in a list
                zerosingle_list.append(zerosingle)

    # turn the empty light curves into a dataframe
    df_empty = pd.DataFrame(zerosingle_list)
    # df_empty has one row per dict. time,flux, and err columns store arrays.
    # "explode" the dataframe to get one row per light curve point. time, flux, and err columns will now store floats.
    df_empty = df_empty.explode(["time", "flux", "err"], ignore_index=True)
    df_empty = df_empty.astype({col: "float" for col in ["time", "flux", "err"]})

    # now put the empty light curves back together with the main light curve dataframe
    zeros_df_lc = pd.concat([df_lc, df_empty])

    return (zeros_df_lc)


def calc_nobjects_per_band_combo(df_lc):
    # this function is not currently in use, but would be good to keep it around for potential later testing.  It helps to determine which combination of existing bands has the most objects to use in classification
    # calling sequence: band_combos_df = calc_nobjects_per_band_combo(df_lc)
    """
    Convert mising data into zero fluxes and uncertainties

    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info

    Returns
    --------
    band_combos_df: pandas dataframe

    """
    all_bands = df_lc.band.unique()
    object_bands = df_lc.groupby("objectid").band.aggregate(lambda x: set(x))

    band_combos = []
    for l in range(1, len(all_bands) + 1):
        band_combos.extend(set(bands) for bands in itertools.combinations(all_bands, l))

    print(band_combos)
    band_combos_nobjects = [
        len(object_bands.loc[(band_combo - object_bands) == set()].index) for band_combo in band_combos]

    band_combos_df = pd.DataFrame({"bands": band_combos, "nobjects": band_combos_nobjects})
    band_combos_df = band_combos_df.sort_values("nobjects", ascending=False)

    return band_combos_df


def missingdata_drop_bands(df_lc, bands_to_keep, verbose=False):
    """
    Drop bands for which too many objects have no observations.

    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info
    bands_to_keep = list of strings with band names
        example: ['W1','W2','panstarrs g','panstarrs i', 'panstarrs r','panstarrs y','panstarrs z','zg','zr']

    Returns
    --------
    clean_df: MultiIndexDFObject with light curves

    """
    # drop all rows where 'band' is not in 'bands_to_keep'
    bands_to_remove = set(df_lc.band.unique()) - set(bands_to_keep)
    clean_df = df_lc.loc[~df_lc.band.isin(bands_to_remove)]

    if verbose:
        # how many objects did we start with?
        print(len(df_lc.objectid.unique()), "n objects before removing missing band data")

    # keep only objects with observations in all remaining bands
    # first, get a boolean series indexed by objectid
    has_all_bands = clean_df.groupby('objectid').band.aggregate(
        lambda x: set(x) == set(bands_to_keep))
    # extract the objectids that are 'True'
    objectids_to_keep = has_all_bands[has_all_bands].index
    # keep only these objects
    clean_df = clean_df.loc[clean_df.objectid.isin(objectids_to_keep)]

    if verbose:
        # How many objects are left?
        print(len(clean_df.objectid.unique()), "n objects after removing missing band data")

    return clean_df


def uniform_length_spacing(df_lc, final_freq_interpol, include_plot=True):
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
    # make a time array with the minimum and maximum of all light curves in the sample
    x_interpol = np.arange(df_lc.time.min(), df_lc.time.max(), final_freq_interpol)
    x_interpol = x_interpol.reshape(-1, 1)  # needed for sklearn
    lc_interpol = []  # list to store interpolated light curves

    # look at each object in each band
    for (band, oid), singleband_oid in df_lc.groupby(["band", "objectid"]):
        # singleband_oid is now a dataframe with just one object and one band
        X = np.array(singleband_oid["time"]).reshape(-1, 1)
        y = np.array(singleband_oid["flux"])
        dy = np.array(singleband_oid["err"])

        # could imagine using GP to make the arrays equal length and spacing
        # however this sends the flux values to zero at the beginning and end of
        # the arrays if there is time without observations.  This is not ideal
        # because it significantly changes the shape of the light curves.

        # kernel = 1.0 * RBF(length_scale=30)
        # gp = GaussianProcessRegressor(kernel=kernel, alpha=dy**2, normalize_y = False)
        # gp.fit(X, y)
        # mean_prediction,std_prediction = gp.predict(x_interpol, return_std=True)

        # try KNN
        KNN = KNeighborsRegressor(n_neighbors=3)
        KNN.fit(X, y)
        mean_prediction = KNN.predict(x_interpol)

        # KNN doesnt output an uncertainty array, so make our own:
        # an array of the same length as mean_prediction
        # having values equal to the mean of the original uncertainty array
        err = np.full_like(mean_prediction, singleband_oid.err.mean())

        # get these values into the dataframe
        # append the results as a dict. the list will be converted to a dataframe later.
        lc_interpol.append(
            {"objectid": oid, "label": str(singleband_oid.label.unique().squeeze()), "band": band, "time": x_interpol.reshape(-1),
             "flux": mean_prediction, "err": err}
        )

        if include_plot:
            # see what this looks like on just a single light curve for now
            if (band == 'zr') and (oid == 9):
                # see if this looks reasonable
                plt.errorbar(X, y, dy, linestyle="None", color="tab:blue", marker=".")
                plt.plot(x_interpol, mean_prediction, label="Mean prediction")
                plt.legend()
                plt.xlabel("time")
                plt.ylabel("flux")
                _ = plt.title("KNN regression")

    # create a dataframe of the interpolated light curves
    df_interpol = pd.DataFrame(lc_interpol)
    return df_interpol


def reformat_df(df_lc):
    """
    Reformat dataframe into shape expected by sktime

    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info

    Returns
    --------
    res_df: MultiIndexDFObject with sktime appropriate shape

    """

    # keep some columns out of the mix when doing the pivot by bandname
    # set them as indices and they won't get pivoted into
    reformat_df = df_lc.set_index(["objectid", "label", "time"])

    # attach bandname to all the fluxes and uncertainties
    reformat_df = reformat_df.pivot(columns="band")

    # rename the columns
    reformat_df.columns = ["_".join(col) for col in reformat_df.columns.values]

    # many of these flux columns still have a space in them from the bandnames,
    # convert that space to underscore
    reformat_df.columns = reformat_df.columns.str.replace(' ', '_')

    # and get rid of that index to cleanup
    reformat_df = reformat_df.reset_index()

    return reformat_df


def local_normalization_max(df_lc, norm_column="flux_W1"):
    """
    normalize each individual light curve by the max value in one band

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
    # make a copy to not alter the original df_lc inside of this function
    norm_df_lc = df_lc.copy()

    # make a new column with max_r_flux for each objectid
    max_W1 = norm_df_lc.groupby('objectid', sort=False)[norm_column].transform('max')

    # figure out which columns in the dataframe are flux columns
    flux_cols = [col for col in norm_df_lc.columns if 'flux' in col]

    # make new normalized flux columns for all fluxes
    norm_df_lc[flux_cols] = norm_df_lc[flux_cols].div(max_W1, axis=0)
    return norm_df_lc


def mjd_to_datetime(df_lc):
    """
    convert time column in dataframe into datetime

    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info

    Returns
    --------
    t.datetime: array of type datetime

    """
    # need to convert df_lc time into datetime
    mjd = df_lc.time

    # convert to JD
    jd = mjd_to_jd(mjd)

    # convert to individual components
    t = Time(jd, format='jd')

    # t.datetime is now an array of type datetime
    # make it a column in the dataframe
    return t.datetime
