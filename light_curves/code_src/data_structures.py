# setup to store the light curves in a data structure
import pandas as pd


class MultiIndexDFObject:
    """
    Container for multiband light-curve data stored as a Pandas MultiIndex DataFrame.

    This class is intended to be used by the `*_get_lightcurves()` functions
    that fetch data from the archives(Gaia, ZTF, WISE, Pan-STARRS,
    TESS/Kepler, IceCube, Rubin, HEASARC, HCV) and return light-curve data.
    Each contributing function produces a pandas.DataFrame indexed by:

        objectid : int
            Unique per-source identifier from the user's sample table.
        label : str
            Literature or source-provenance label associated with the object.
        band : str
            Photometric band or mission/instrument label.
        time : float
            Time of observation in MJD.

    And containing the following columns:

        flux : float
            Flux measurement in the native calibrated units for that archive
            (e.g., mJy for Gaia, WISE, Pan-STARRS, ZTF, Rubin, HCV;
            electrons/s for TESS/Kepler; log(GeV) proxy for IceCube).

        err : float
            Uncertainty on the flux measurement, in the same units.

    The object provides convenience methods for appending, combining,
    and deleting light-curve segments.

    Examples
    --------

    # Initialize Pandas MultiIndex data frame for storing the light curve
    df_lc = MultiIndexDFObject()

    #make a single multiindex dataframe
    dfsingle = pd.DataFrame(dict(flux=[0.1], err=[0.1], time=[time_mjd], objectid=[ccount + 1], /
        band=[mission], label=lab)).set_index(["objectid", "label", "band", "time"])

    # Append to existing MultiIndex light curve object
    df_lc.append(dfsingle)

    #Show the contents
    df_lc.data


   df_lc.data
   """

    def __init__(self, data=None):
        """Create a MultiIndex DataFrame that is empty if data is None, else contains the data.

        Parameters
        ----------
        data : pd.DataFrame, optional
            A DataFrame already indexed by
            [objectid, label, band, time] and containing columns ['flux', 'err'].
            If None, an empty MultiIndex structure is created.
        """
        index = ["objectid", "label", "band", "time"]
        columns = ["flux", "err"]
        self.data = pd.DataFrame(columns=index + columns).set_index(index)
        if data is not None:
            self.append(data)

    def _repr_html_(self):
        return self.data._repr_html_()

    def __repr__(self):
        return self.data.__repr__()

    def append(self, x):
        """
        Append new light-curve data to the object.

        Parameters
        ----------
        x : pandas.DataFrame or MultiIndexDFObject
            Data to append. Must contain columns ['flux', 'err']
            and be indexed by [objectid, label, band, time].

        Notes
        -----
        - If `x` is empty, nothing is changed.
        - If the current container is empty, it is replaced by `x`.
        - Otherwise, the new data are concatenated with the existing MultiIndex.

        """
        if isinstance(x, self.__class__):
            # x is a MultiIndexDFObject. extract the DataFrame
            new_data = x.data
        else:
            # assume x is a pd.DataFrame
            new_data = x

        # if either new_data or self.data is empty we should not try to concat
        if new_data.empty:
            # leave self.data as is
            return
        if self.data.empty:
            # replace self.data with new_data
            self.data = new_data
            return

        # if we get here, both new_data and self.data contain data, so concat
        self.data = pd.concat([self.data, new_data])

    def remove(self, x):
        """
        Drop a light curve from the dataframe

        Parameters
        ----------
        x : list of values
            MultiIndex coordinates identifying the rows to drop.
            Must match the existing index levels:
            (objectid, label, band, time).
        """
        self.data = self.data.drop(x)
