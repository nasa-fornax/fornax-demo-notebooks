import lsdb
import numpy as np
import pandas as pd
from dask.distributed import Client

from data_structures import MultiIndexDFObject

# panstarrs light curves from hats catalog in S3 using lsdb


def panstarrs_get_lightcurves(sample_table, *, radius=1):
    """Searches Pan-STARRS HATS files on S3 for light curves from a list of input coordinates.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table containing the source sample. The following columns must be present:
            coord : astropy.coordinates.SkyCoord
                Sky position of each source.
            objectid : int
                Unique identifier for each source in the sample.
            label : str
                Literature label for tracking source provenance.
    radius : float
        Angular search radius in arcseconds used for the crossmatch between the input
        sample and the Pan-STARRS object catalog. Default is 1.0 arcsec.

    Returns
    -------
    df_lc : MultiIndexDFObject
        Indexed by [objectid, label, band, time]. The resulting internal pandas DataFrame
        contains the following columns:

            flux : float
                Flux values in millijansky (mJy), converted from PS1 psfFlux (Jy × 1e3).
            err : float
                Flux uncertainties in millijansky (mJy), converted from PS1 psfFluxErr.
            time : float
                Time of observation in MJD (Pan-STARRS obsTime in MJD).
            objectid : int
                Input sample object identifier.
            band : str
                Pan-STARRS filter name ('Pan-STARRS g', 'Pan-STARRS r', 'Pan-STARRS i',
                'Pan-STARRS z', or 'Pan-STARRS y').
            label : str
                Literature label associated with each source.
    """

    # read in the panstarrs object table to lsdb
    # this table will be used for cross matching with our sample's ra and decs
    # but does not have light curve information
    panstarrs_object = lsdb.open_catalog(
        's3://stpubdata/panstarrs/ps1/public/hats/otmo',
        columns=["objID",  # PS1 ID
                 "raMean", "decMean",  # coordinates to use for cross-matching
                 "nStackDetections",  # some other data to use
                 ]
    )
    # read in the panstarrs light curves to lsdb
    # panstarrs recommendation is not to index into this table with ra and dec
    # but to use object ids from the above object table
    panstarrs_detect = lsdb.open_catalog(
        's3://stpubdata/panstarrs/ps1/public/hats/detection',
        columns=["objID",  # PS1 object ID
                 "detectID",  # PS1 detection ID
                 # light-curve stuff
                 "obsTime", "filterID", "psfFlux", "psfFluxErr",
                 ]
    )
    # convert astropy table to pandas dataframe
    # special care for the SkyCoords in the table
    sample_df = pd.DataFrame({'objectid': sample_table['objectid'],
                              'ra_deg': sample_table['coord'].ra.deg,
                              'dec_deg': sample_table['coord'].dec.deg,
                              'label': sample_table['label']})

    # convert dataframe to hipscat
    sample_lsdb = lsdb.from_dataframe(
        sample_df,
        ra_column="ra_deg",
        dec_column="dec_deg",
        margin_threshold=10,
        # Optimize partition size
        drop_empty_siblings=True
    )

    # plan to cross match panstarrs object with my sample
    # only keep the best match
    matched_objects = sample_lsdb.crossmatch(
        panstarrs_object,
        radius_arcsec=radius,
        n_neighbors=1,
        suffixes=("", "")
    )

    # plan to join that cross match with detections to get light-curves
    matched_lc = matched_objects.join(
        panstarrs_detect,
        left_on="objID",
        right_on="objID",
        output_catalog_name="yang_ps_lc",
        suffixes=["", ""]
    )

    # Create default local cluster
    # here is where the actual work gets done
    with Client():
        # compute the cross match with object table
        # and the join with the detections table
        matched_df = matched_lc.compute()

    # handle the case where there are no matches and return empty df
    if len(matched_df["filterID"]) == 0:
        return MultiIndexDFObject(data=pd.DataFrame())

    # cleanup the filter names to the expected letters
    filter_id_to_name = {
        1: 'Pan-STARRS g',
        2: 'Pan-STARRS r',
        3: 'Pan-STARRS i',
        4: 'Pan-STARRS z',
        5: 'Pan-STARRS y'
    }

    get_name_from_filter_id = np.vectorize(filter_id_to_name.get)
    filtername = get_name_from_filter_id(matched_df["filterID"])

    # make the dataframe of light curves
    # the data conversions are to change from pyarrow datatypes to numpy datatypes

    df_lc = pd.DataFrame({
        'flux': pd.to_numeric(matched_df['psfFlux'] * 1e3, errors='coerce').astype(np.float64),
        'err': pd.to_numeric(matched_df['psfFluxErr'] * 1e3, errors='coerce').astype(np.float64),
        'time': pd.to_numeric(matched_df['obsTime'], errors='coerce').astype(np.float64),
        'objectid': matched_df['objectid'].astype(np.int64),
        'band': filtername,
        'label': matched_df['label'].astype(str)
    }).set_index(["objectid", "label", "band", "time"])

    return MultiIndexDFObject(data=df_lc)
