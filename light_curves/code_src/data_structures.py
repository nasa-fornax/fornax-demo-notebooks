#setup to save the light curves in a data structure
import pandas as pd
from astropy.table import vstack
from astropy.timeseries import TimeSeries


class MultiIndexDFObject:
    """
    Pandas MultiIndex data frame to store & manipulate multiband light curves 

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


    """
    def __init__(self, data=None):
        """Create a MultiIndex DataFrame that is empty if data is None, else contains the data.
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            Dataframe to store in the `data` attribute.
        """
        index = ["objectid", "label", "band", "time"]
        columns = ["flux", "err"]
        self.data = pd.DataFrame(columns=index + columns).set_index(index)
        if data is not None:
            self.append(data)
    
    def append(self,x):
        """Add a new band of light curve data to the dataframe
        
        Parameters
        ----------
        x : Pandas dataframe
            contains columns [flux, fluxerr] and multi-index [objectid, label, band, time]
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
            
    def remove(self,x):
        """ Drop a light curve from the dataframe
        
        Parameters
        ----------
        x : list of values
             Index values identifying rows to be dropped.
        """
        self.data = self.data.drop(x)
        
## From Brigitta as a possible data structure
## Not currently in use
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
            

            