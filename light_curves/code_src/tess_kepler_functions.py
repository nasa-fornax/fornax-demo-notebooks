import lightkurve as lk
import pandas as pd
from tqdm.auto import tqdm

from data_structures import MultiIndexDFObject


def clean_filternames(lightcurve):
    """Simplify mission name from a combined list

    Mission names are returned including quarter numbers, remove those to
    simplify. For this use case, we really only need to know which mission, 
    not which quarter.

    Parameters
    ----------
    lightcurve : lightkurve result
        object detailing the light curve object found with lightkurve

    Returns
    -------
    filtername : str
        name of the mission without quarter information
    """
    filtername = lightcurve.mission
    # clean this up a bit so all Kepler quarters etc., get the same filtername
    # we don't need to track the individual names for the quarters, just need to know which mission it is
    if 'Kepler' in filtername:
        filtername = 'Kepler'
    if 'TESS' in filtername:
        filtername = 'TESS'
    if 'K2' in filtername:
        filtername = 'K2'
    return (filtername)


def tess_kepler_get_lightcurves(sample_table, *, radius=1.0):
    """Searches TESS, Kepler, and K2 for light curves from a list of input coordinates.  This is the MAIN function

    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    radius : float
        search radius, how far from the source should the archives return results

    Returns
    -------
    df_lc : MultiIndexDFObject
        the main data structure to store all light curves
    """

    df_lc = MultiIndexDFObject()

    # for all objects
    for row in tqdm(sample_table):
        # for testing, this has 79 light curves between the three missions.
        # for ccount in range(1):
        #    coord = '19:02:43.1 +50:14:28.7'

        # use lightkurve to search TESS, Kepler and K2. if nothing is found, continue to the next object.
        # https://docs.lightkurve.org/tutorials/1-getting-started/searching-for-data-products.html
        search_result = lk.search_lightcurve(row["coord"], radius=radius)
        if not search_result:
            continue

        # download all of the returned light curves from TESS, Kepler, and K2
        # results occasionally include an unsupported product and this raises a LightkurveError
        try:
            lc_collection = search_result.download_all()
        except lk.LightkurveError:
            continue

        # can't get the whole collection directly into pandas multiindex
        # pull out inidividual light curves, convert to uniform units, and put them in pandas
        for lightcurve in lc_collection:  # for testing 0 is Kepler, #69 is TESS

            # convert to Pandas
            lcdf = lightcurve.to_pandas().reset_index()

            # record band name
            filtername = clean_filternames(lightcurve)

            # filter out TESS negative fluxes (non-detections) and NaNs
            # https://tess.mit.edu/public/tesstransients/pages/readme.html
            if filtername == "TESS":
                lcdf = lcdf.loc[lcdf.flux > 0]

            # these light curves are too highly sampled for our AGN use case, so reduce their size
            # by choosing only to keep every nth sample
            nsample = 30
            # selects every nth row starting with row 0
            lcdf_small = lcdf[lcdf.index % nsample == 0]

            # convert time to mjd
            time_lc = lcdf_small.time  # in units of time - 2457000 BTJD days
            # now in MJD days within a few minutes (except for the barycenter correction)
            time_lc = time_lc + 2457000 - 2400000.5

            # TESS, Kepler, and K2 report flux in units of electrons/s
            # there is no good way to convert this to anything more useful because the bandpasses are very wide and nonstandard
            # really we don't care about absolute scale, but want to scale the light curve to be on the same plot as other light curves
            # save as electron/s here and scale when plotting
            flux_lc = lcdf_small.flux  # in electron/s
            fluxerr_lc = lcdf_small.flux_err  # in electron/s

            # put this single object light curves into a pandas multiindex dataframe
            # fluxes are in units of electrons/s and will be scaled to fit the other fluxes when plotting
            dfsingle = pd.DataFrame(dict(flux=flux_lc, err=fluxerr_lc, time=time_lc, objectid=row["objectid"], band=filtername, label=row["label"])).set_index(
                ["objectid", "label", "band", "time"])

            # then concatenate each individual df together
            df_lc.append(dfsingle)

    return (df_lc)
