# Detect spikes in the array of candidate jet sizes (or probability, etc.) vs. wavelength.
# Replace the contents of this function with your favorite outlier detection algorithm.

import numpy as np
from scipy.signal import find_peaks

def detect_spikes(blobs, pos, sig=3.0, detect_halfwidth=10, match_halfwidth=2):
    """

    Input parameters
    ----------
    blobs : 1D Python list
        The list returned by extension_vs_wavebin.
        List of length equal to the number of wavebins, with values
        corresponding to the fraction of pixels from that slice above
        a brightness threshold.
    pos : integer
        The expected wavebin index of a particular spectral line.
    sig : float
        Scaling factor used to set the detection threshold.
    detect_halfwidth : integer
        The number of pixels +/- offset from pos to run peak detector on.
    match_halfwidth : integer
        The max number of pixels away from pos that a peak can be centered on
        and still have this function yield a detection at pos.
    

    Returns
    ----------
    True (if spike is detected near pos) or False (if not)

    """

    # Excerpt a range of +/- 10 pixels from the expected wavelength of the Fe II line.
    offset = detect_halfwidth
    start = max(0, pos - offset)
    end = min(len(blobs), pos + offset + 1)
    excerpt = np.asarray(blobs[start:end])

    # Check for the case where all non-zero excerpted data is NaN
    if np.all(np.isnan(excerpt[excerpt!=0])):
        return False

    # Set a detection threshold equal to the standard deviation of this excerpt,
    # multiplied by a scaling factor.
    threshold = np.nanstd(excerpt[excerpt!=0])*sig
    
    # Find peaks with scipy.signal.find_peaks
    peaks, properties = find_peaks(excerpt, prominence=threshold)

    # If any of the detected peaks are within a couple pixels of the expected position of the Fe II line, 
    # then return True.
    # Otherwise, return False.
    return any((pos-match_halfwidth <= peak+(pos-offset) <= pos+match_halfwidth) for peak in peaks)