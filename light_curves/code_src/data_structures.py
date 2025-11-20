# setup to store the light curves in a data structure
import pandas as pd
from astropy.table import vstack
from astropy.timeseries import TimeSeries


class MultiIndexDFObject:
    """
    Container for multiband light-curve data stored as a Pandas MultiIndex DataFrame.

    This class standardizes how all archives (Gaia, ZTF, WISE, Pan-STARRS,
    TESS/Kepler, IceCube, Rubin, HEASARC, HCV) return light-curve data.
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

    def combine_Samples(self, other_df):
        """
        Combine two MultiIndexDFObject instances into a new one.

        Parameters
        ----------
        other_df : MultiIndexDFObject
            Another instance whose light-curve data will be concatenated.

        Returns
        -------
        MultiIndexDFObject
            A new container with combined data. The objectid values in
            `other_df` are shifted upward so that there are no collisions
            between the two samples.

        Notes
        -----
        This is useful when building a composite sample from multiple
        literature lists or survey queries.
        """
        if not isinstance(other_df, self.__class__):
            raise ValueError("Input must be an instance of MultiIndexDFObject")

        # Get the length of the initial DataFrame's object IDs
        initial_object_ids_length = self.data.index.get_level_values('objectid').max() + 1

        # Update the object IDs in the second DataFrame
        other_df.data.index = other_df.data.index.set_levels(
            other_df.data.index.levels[0] + initial_object_ids_length,
            level='objectid'
        )
        # Concatenate the dataframes
        concatenated_data = pd.concat([self.data, other_df.data])

        # Create a new MultiIndexDFObject with the concatenated data
        concatenated_df_obj = self.__class__(data=concatenated_data)

        return concatenated_df_obj

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

# From Brigitta as a possible data structure
# Not currently in use


class MultibandTimeSeries(TimeSeries):
    def __init__(self, *, data=None, time=None, **kwargs):
        # using kwargs to swallow all other arguments a TimeSeries/QTable can have,
        # but we dont explicitly use. Ideally they are spelt out if we have docstrings here.
        # Also using keyword only arguments everywhere to force being explicit.
        super().__init__(data=data, time=time, **kwargs)

    def add_band(self, *, time=None, data=None, band_name="None"):
        '''Add a time, flux/mag data set and resort the arrays. ``time`` can be a TimeSeries instance'''
        if 'time' not in self.colnames:
            if isinstance(time, TimeSeries):
                super().__init__(time)
            else:
                super().__init__(data={band_name: data}, time=time)
        else:
            if time is None:
                # this assumes ``band_name`` fluxes are taken at the very same times as the already exitsing bands
                # TODO: include checks for sizes and other assumptions
                self[band_name] = data
                return
            elif not isinstance(time, TimeSeries):
                # TODO: handle band_name=None case
                time = TimeSeries(time=time, data={band_name: data})
            super().__init__(vstack([self, time]))
