import numpy as np
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt
# import numba
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from tqdm.auto import tqdm


def translate_bitwise_sum_to_labels(bitwise_sum):
    """
    Translate a bitwise sum back to the labels which were set to 1.

    Parameters:
    - bitwise_sum: Integer, the bitwise sum representing the combination of labels.
    - labels: List of strings, the labels corresponding to each bit position.

    Returns:
    - List of strings, the labels that are set to 1.
    """
    # Initialize agnlabels
    agnlabels = ['SDSS_QSO', 'WISE_Variable', 'Optical_Variable', 'Galex_Variable',
                 'Turn-on', 'Turn-off',
                 'SPIDER', 'SPIDER_AGN', 'SPIDER_BL', 'SPIDER_QSOBL', 'SPIDER_AGNBL',
                 'TDE', 'Fermi_blazar']
    active_labels = []
    for i, label in enumerate(agnlabels):
        # Check if the ith bit is set to 1
        if bitwise_sum & (1 << i):
            active_labels.append(label)
    return active_labels


def update_bitsums(df, label_num=64):
    '''To update the bitwise summed labels by subtracting the 64s added in'''

    # Extract index as a list of tuples if MultiIndex, or adjust accordingly
    index_list = list(df.index)

    # Prepare a new list for the updated index
    updated_index = []

    # Track whether any changes are made to avoid unnecessary DataFrame recreation
    changes_made = False

    for idx in index_list:
        current_label = int(idx[1])  # Assuming 'label' is the second level in the multi-index

        # Check if 64 is part of the bitwise sum
        if current_label & label_num != 0:
            new_label = current_label ^ label_num  # Calculate the new label by removing 64 using XOR
            new_idx = list(idx)
            new_idx[1] = new_label  # Update the label in the index tuple
            updated_index.append(tuple(new_idx))
            changes_made = True
        else:
            updated_index.append(idx)

    # If changes were made, update the DataFrame index
    if changes_made:
        df_updated = df.set_index(pd.MultiIndex.from_tuples(updated_index, names=df.index.names))
    else:
        df_updated = df  # No changes, return original DataFrame

    return df_updated


def autopct_format(values):

    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%)'.format(pct)

    return my_format


def unify_lc(df_lc, bands_inlc=['zr', 'zi', 'zg'], xres=160, numplots=1, low_limit_size=5):
    '''
    Function to preprocess and unify time dimension of light curve data with linear interpolation.

    Parameters:
    - df_lc: DataFrame with light curve data.
    - bands_inlc: List of bands to include in the analysis (default: ['zr', 'zi', 'zg']).
    - xres: Resolution for interpolation (default: 160).
    - numplots: Number of plots to display (default: 1).
    - low_limit_size: Minimum number of data points required in a band (default: 5).
    '''

    # Creating linearly spaced arrays for interpolation for different instruments
    x_ztf = np.linspace(0, 1600, xres)  # For ZTF
    x_wise = np.linspace(0, 4000, xres)  # For WISE

    # Extract unique object IDs from the DataFrame
    objids = df_lc.index.get_level_values('objectid')[:].unique()

    # Initialize variables for storing results
    printcounter = 0
    objects, dobjects, flabels, keeps = [], [], [], []

    # Iterate over each object ID
    for keepindex, obj in tqdm(enumerate(objids)):
        singleobj = df_lc.loc[obj, :, :, :]  # Extract data for the single object
        label = singleobj.index.unique('label')  # Get the label of the object
        bands = singleobj.loc[label[0], :, :].index.get_level_values('band')[
            :].unique()  # Extract bands

        keepobj = 0  # Flag to determine if the object should be kept

        # Check if the object has all required bands
        if len(np.intersect1d(bands, bands_inlc)) == len(bands_inlc):
            if printcounter < numplots:
                fig, ax = plt.subplots(figsize=(15, 5))  # Set up plot if within numplots limit

            # Initialize arrays for new interpolated Y and dY values
            obj_newy = [[] for _ in range(len(bands_inlc))]
            obj_newdy = [[] for _ in range(len(bands_inlc))]

            keepobj = 1  # Set keepobj to 1 (true) initially

            # Process each band in the included bands
            for l, band in enumerate(bands_inlc):
                band_lc = singleobj.loc[label[0], band, :]  # Extract light curve data for the band
                # Clean data to remove times greater than a threshold (65000)
                band_lc_clean = band_lc[band_lc.index.get_level_values('time') < 65000]
                x, y, dy = np.array(band_lc_clean.index.get_level_values(
                    'time') - band_lc_clean.index.get_level_values('time')[0]), np.array(band_lc_clean.flux), np.array(band_lc_clean.err)

                # Sort data based on time
                x2, y2, dy2 = x[np.argsort(x)], y[np.argsort(x)], dy[np.argsort(x)]

                # Check if there are enough points for interpolation
                if len(x2) > 5:
                    # Handle time overlaps in light curves
                    n = np.sum(x2 == 0)
                    for b in range(1, n):
                        x2[::b + 1] = x2[::b + 1] + 1 * 0.001

                    # Interpolate the data
                    f = interpolate.interp1d(x2, y2, kind='previous', fill_value="extrapolate")
                    df = interpolate.interp1d(x2, dy2, kind='previous', fill_value="extrapolate")

                    # Plot data if within the numplots limit
                    if printcounter < numplots:
                        plt.errorbar(x2, y2, dy2, capsize=1.0, marker='.',
                                     linestyle='', label=label[0] + band)
                        if band in ['W1', 'W2']:
                            plt.plot(x_wise, f(x_wise), '--',
                                     label='nearest interpolation ' + str(band))
                        else:
                            plt.plot(x_ztf, f(x_ztf), '--',
                                     label='nearest interpolation ' + str(band))

                    # Assign interpolated values based on the band
                    if band == 'W1' or band == 'W2':
                        obj_newy[l] = f(x_wise)  # /f(x_wise).max()
                        obj_newdy[l] = df(x_wise)
                    else:
                        obj_newy[l] = f(x_ztf)  # /f(x_ztf).max()
                        obj_newdy[l] = df(x_ztf)  # /f(x_ztf).max()

                # don't keep objects which have less than x datapoints in any keeping bands
                if len(obj_newy[l]) < low_limit_size:
                    keepobj = 0

            if printcounter < numplots:
                plt.title('Object '+str(obj))  # +' from '+label[0]+' et al.')
                plt.xlabel('Time(MJD)')
                plt.ylabel('Flux(mJy)')
                plt.legend()
                plt.show()
                printcounter += 1

        if keepobj:
            objects.append(obj_newy)
            dobjects.append(obj_newdy)
            flabels.append(label[0])
            keeps.append(keepindex)
    return np.array(objects), np.array(dobjects), flabels, keeps


def unify_lc_gp(df_lc, bands_inlc=['zr', 'zi', 'zg'], xres=160, numplots=1, low_limit_size=5):
    '''
    Function to preprocess and unify the time dimension of light curve data using Gaussian Processes.

    Parameters:
    - df_lc: DataFrame with light curve data.
    - bands_inlc: List of bands to include in the analysis (default: ['zr', 'zi', 'zg']).
    - xres: Resolution for interpolation (default: 160).
    - numplots: Number of plots to display (default: 1).
    - low_limit_size: Minimum number of data points required in a band (default: 5).
    '''
    x_ztf = np.linspace(0, 1600, xres).reshape(-1, 1)  # X array for interpolation
    x_wise = np.linspace(0, 4000, xres).reshape(-1, 1)  # X array for interpolation
    objids = df_lc.index.get_level_values('objectid')[:].unique()

    printcounter = 0
    objects, dobjects, flabels, keeps = [], [], [], []
    for keepindex, obj in tqdm(enumerate(objids)):

        singleobj = df_lc.loc[obj, :, :, :]
        label = singleobj.index.unique('label')
        bands = singleobj.loc[label[0], :, :].index.get_level_values('band')[:].unique()
        keepobj = 0
        if len(np.intersect1d(bands, bands_inlc)) == len(bands_inlc):
            if printcounter < numplots:
                fig = plt.subplots(figsize=(15, 5))

            obj_newy = [[] for _ in range(len(bands_inlc))]
            obj_newdy = [[] for _ in range(len(bands_inlc))]

            keepobj = 1
            for l, band in enumerate(bands_inlc):
                band_lc = singleobj.loc[label[0], band, :]
                band_lc_clean = band_lc[band_lc.index.get_level_values('time') < 65000]
                x, y, dy = np.array(band_lc_clean.index.get_level_values(
                    'time')-band_lc_clean.index.get_level_values('time')[0]), np.array(band_lc_clean.flux), np.array(band_lc_clean.err)

                x2, y2, dy2 = x[np.argsort(x)], y[np.argsort(x)], dy[np.argsort(x)]
                if len(x2) > low_limit_size:

                    n = np.sum(x2 == 0)
                    # this is a hack of shifting time of different lightcurves by a bit so I can interpolate!
                    for b in range(1, n):
                        x2[::b+1] = x2[::b+1]+1*0.001
                    X = x2.reshape(-1, 1)
                    if band == 'W1' or band == 'W2':
                        kernel = 1.0 * RBF(length_scale=200)
                        gpw = GaussianProcessRegressor(kernel=kernel, alpha=(dy2)**2)
                        gpw.fit(X, y2)
                        obj_newy[l], obj_newdy[l] = gpw.predict(x_wise, return_std=True)
                    else:
                        kernel = 1.0 * RBF(length_scale=len(x_ztf)/len(x2))
                        gp = GaussianProcessRegressor(kernel=kernel, alpha=dy2**2)
                        gp.fit(X, y2)
                        obj_newy[l], obj_newdy[l] = gp.predict(x_ztf, return_std=True)

                    if printcounter < numplots:
                        plt.errorbar(x2, y2, dy2, capsize=1.0, marker='.',
                                     linestyle='', label=label[0]+band)
                        if band == 'W1' or band == 'W2':
                            y_pred, sigma = gpw.predict(x_wise, return_std=True)
                            plt.plot(x_wise, y_pred, '--', label='Gaussian Process Reg.'+str(band))
                            plt.fill_between(x_wise.flatten(), y_pred - 1.96 * sigma,
                                             y_pred + 1.96 * sigma, alpha=0.2, color='r')
                        else:
                            kernel = 1.0 * RBF(length_scale=len(x_ztf)/len(x2))
                            y_pred, sigma = gp.predict(x_ztf, return_std=True)
                            plt.plot(x_ztf, y_pred, '--', label='Gaussian Process Reg.'+str(band))
                            plt.fill_between(x_ztf.flatten(), y_pred - 1.96 * sigma,
                                             y_pred + 1.96 * sigma, alpha=0.2, color='r')
                else:
                    keepobj = 0
            if (printcounter < numplots):
                plt.title('Object '+str(obj))  # +' from '+label[0]+' et al.')
                plt.xlabel('Time(MJD)')
                plt.ylabel('Flux(mJy)')
                plt.legend()
                plt.show()
                printcounter += 1
        if keepobj:
            objects.append(obj_newy)
            dobjects.append(obj_newdy)
            flabels.append(label[0])
            keeps.append(keepindex)
    return np.array(objects), np.array(dobjects), flabels, keeps


def combine_bands(objects, bands):
    '''
    combine all lightcurves in individual bands of an object
    into one long array, by appending the indecies.
    '''
    dat = []
    for o, ob in enumerate(objects):
        obj = []
        for b in range(len(bands)):
            obj = np.append(obj, ob[b], axis=0)
        dat.append(obj)
    return np.array(dat)


def mean_fractional_variation(lc, dlc):
    '''A common way of defining variability'''
    meanf = np.mean(lc)  # mean flux of all points
    varf = np.std(lc)**2
    deltaf = np.mean(dlc)**2
    if meanf <= 0:
        meanf = 0.0001
    fvar = (np.sqrt(varf-deltaf))/meanf
    return fvar


def stat_bands(objects, dobjects, bands, sigmacl=5):
    '''
    returns arrays with maximum,mean,std flux in the 5sigma clipped lightcurves of each band .
    '''
    fvar, maxarray, meanarray = np.zeros((len(bands), len(objects))), np.zeros(
        (len(bands), len(objects))), np.zeros((len(bands), len(objects)))
    for o, ob in enumerate(objects):
        for b in range(len(bands)):
            clipped_arr, l, u = stats.sigmaclip(ob[b], low=sigmacl, high=sigmacl)
            clipped_varr, l, u = stats.sigmaclip(dobjects[o, b, :], low=sigmacl, high=sigmacl)
            maxarray[b, o] = clipped_arr.max()
            meanarray[b, o] = clipped_arr.mean()
            fvar[b, o] = mean_fractional_variation(clipped_arr, clipped_varr)
    return fvar, maxarray, meanarray


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


def normalize_clipmax_objects(data, maxarr, band=1):
    '''
    normalize combined data array by by max value after clipping the outliers in one band (second band here).
    '''
    d2 = np.zeros_like(data)
    for i, d in enumerate(data):
        if band <= np.shape(maxarr)[0]:
            d2[i] = (d/maxarr[band, i])
        else:
            d2[i] = (d/maxarr[0, i])
    return d2

# Shuffle before feeding to umap


def shuffle_datalabel(data, labels):
    """shuffles the data, labels and also returns the indecies """
    p = np.random.permutation(len(data))
    data2 = data[p, :]
    fzr = np.array(labels)[p.astype(int)]
    return data2, fzr, p

# @numba.njit()


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
