import numpy as np
from scipy.ndimage import label

def extension_vs_wavebin(data):
    """
    Generate a 1D list of candidate emission spatial extents for each 
    wavelength bin of a given spectral cube. Replace the contents of
    this function withyour favorite image segmentation algorithm. The desired
    output is any 1D python list where a peak in some way corresponds to an
    increased chance of a jet being present at the corresponding wavebin.

    Parameters
    ----------
    data : np.ndarray
        A spectral cube as a numpy.ndarray,
        with first axis wavelength,
        second and third axes spatial.

    Returns
    ----------
    blobs : list[float]
        List of length equal to the number of wavebins, with values
        corresponding to the fraction of pixels from that slice above
        a brightness threshold.

    """

    # Let's instantiate a list "blobs".
    # This will contain a list of the sizes (as a fraction of non-NaN pixels in each slice)
    # of contiguous regions ("blobs") of pixels brighter than some threshold in each slice.
    blobs = []

    for i in range(0, data.shape[0]): # For every image slice in the spectral cube:

        # Grab the image for this slice.
        dataslice = data[i]

        # Set a threshold flux value above which a pixel is considered to not be part of the background.
        # Here, we base this threshold on the median brightness of all pixels within +/- 10 slices of the current slice.
        # This is pretty arbitrary: demonstration purposes only!
        start = max(0, i-10)
        end = min(data.shape[0], i+10)

        # Check for the case where all data is NaN
        if np.all(np.isnan(data[start:end])):
            blobs.append(0.)
            continue  # Skip the rest of this loop
            
        threshold = 3.0*np.nanmedian(data[start:end])

        # Define a mask for this slice,
        # where a pixel has a value of True if it exceeds the threshold,
        # and a value of False if it does not.
        bright_pixels = dataslice > threshold

        # Use scipy.ndimage.label to define contiguous regions (blobs) of these bright pixels.
        labeled_array, num_features = label(bright_pixels)

        # In labeled_array, pixels with a value of 0 did not meet the bright_pixels condition; 
        # these are background pixels.
        # Any other labels (1, 2, 3, and so on) represent different contiguous regions of pixels
        # meeting the bright_pixels condition. These are our blobs.

        # Get the number of pixels in each blob, in order of label (0, 1, 2, 3...).
        sizes = np.bincount(labeled_array.ravel())

        # If all the pixels are labeled 0, then there is no candidate jet feature in this slice.
        # So we'll assign 0 to the size of the jet feature candidate in this slice.
        if len(sizes) <= 1:
            blobs.append(0.)

        # If there's more than one label...
        if len(sizes) > 1:

            #...the background pixels (label 0) are first in the list we made with np.bincount.
            # The rest are blobs.
            # Let's get the size of the biggest blob: this is probably our best jet candidate.
            jet_size = np.max(sizes[1:])

            # Get the number of total pixels in the slice.
            num_pixels = len(dataslice.flatten())

            # Get the number of pixels in the slice that are not not NaN.
            num_goodpixels = len(dataslice[~np.isnan(dataslice)].flatten())

            # What fraction of the non-NaN pixels are occupied by the jet candidate?
            jet_fraction = jet_size / num_goodpixels

            # To improve the likelihood that this is a jet,
            # let's require that the jet candidate be at least 10 pixels in size,
            # but occupy less than half of the slice.
            if (jet_size >= 10) and (jet_fraction < 0.5):
                blobs.append(jet_fraction)  # Add the jet candidate size fraction for this slice to the list.
            else:  # Otherwise, assume that we don't have a good jet candidate.
                blobs.append(0.)

    return blobs