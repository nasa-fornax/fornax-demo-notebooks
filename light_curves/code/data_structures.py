#setup to save the light curves in a data structure
from astropy.timeseries import TimeSeries
import pandas as pd

            
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
    def __init__(self):
        pass
    
    def append(self,x):
        """Add a new band of light curve data to the dataframe
        
        Parameters
        ----------
        x : `dict`
            contains flux, fluxerr, time, objectid, bandname, reflabel 
        """
        try:
            self.data
        except AttributeError:
            self.data = x.copy()
        else:
            self.data = pd.concat([self.data , x])
            
    def pickle(self,x):
        """ Save the multiindex data frame to a pickle file
        
        Parameters
        ----------
        x : `dict`
            contains flux, fluxerr, time, objectid, bandname, reflabel 
        """
        
        self.data.to_pickle(x)  
        
    def load_pickle(self,x):
        """ Load the multiindex data frame from a pickle file
        
        Parameters
        ----------
        x : `dict`
            contains flux, fluxerr, time, objectid, bandname, reflabel 
        """
        with open(x , "rb") as f:
            self.data = pickle.load(f)
            
    def remove(self,x):
        """ Drop a light curve from the dataframe
        
        Parameters
        ----------
        x : `dict`
            contains flux, fluxerr, time, objectid, bandname, reflabel 
        """
        self.data.drop(x,inplace=True)
        self.data.reset_index()
        
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
            

            