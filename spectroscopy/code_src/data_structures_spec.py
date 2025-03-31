# setup to store the spectra in a data structure
import pandas as pd


class MultiIndexDFObject:
    """
    Pandas MultiIndex data frame to store & manipulate spectra.

    Examples
    --------
    # Initialize Pandas MultiIndex data frame for storing the spectra
    df_spec = MultiIndexDFObject()

    # Make a single multiindex dataframe
    df_single = pd.DataFrame(dict(wave=[0.1], flux=[0.1], err=[0.1], instrument=[instrument_name],
                                  objectid=[ccount + 1], filter=[filter_name],
                                  mission=[mission_name], label=[lab]))
    df_single = df_single.set_index(["objectid", "label", "filter", "mission"])

    # Append to existing MultiIndex object
    df_spec.append(dfsingle)

    # Show the contents
    df_spec.data
    """

    def __init__(self, data=None):
        """
        Create a MultiIndex DataFrame that is empty if data is None, else contains the data.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Dataframe to store in the `data` attribute.
        """

        index = ["objectid", "label", "filter", "mission"]
        columns = ["wave", "flux", "err", "instrument"]
        self.data = pd.DataFrame(columns=index + columns).set_index(index)
        if data is not None:
            self.append(data)

    def append(self, x):
        """
        Add a new spectra data to the dataframe.

        Parameters
        ----------
        x : Pandas dataframe
            Contains columns ["wave", "flux", "err", "instrument"]
            and multi-index ["objectid", "label", "filter", "mission"].
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
        Drop a row from the dataframe.

        Parameters
        ----------
        x : list of values
            Index values identifying rows to be dropped.
        """

        self.data = self.data.drop(x)
