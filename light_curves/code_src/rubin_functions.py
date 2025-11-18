from pathlib import Path

import pyvo
from astropy.table import vstack

from data_structures import MultiIndexDFObject


def rubin_authenticate():
    """
    Authenticate to the Rubin Science Platform (RSP) TAP service using a
    locally stored authentication token located at ~/.rsp-tap.token.

    Returns
    -------
    rsp_tap : pyvo.dal.TAPService
        An authenticated TAPService instance pointing to DP0 data at
        https://data.lsst.cloud/api/tap 

    Raises
    ------
    FileNotFoundError
        If the token file does not exist or is empty.
    ValueError
        If the token file exists but contains no valid token string.
    RuntimeError
        If the TAP session cannot be created or the returned baseurl mismatches the expected URL.

    Notes
    -----
    The ~/.rsp-tap.token file must contain a single line with a valid RSP
    access token. This token must have TAP search permissions enabled.

    """
    # Read token from home directory
    token_file = Path.home() / '.rsp-tap.token'
    if not token_file.exists():
        raise FileNotFoundError(f"Token file not found: {token_file}")
    with open(token_file, 'r') as f:
        token = f.readline().strip()
    if not token:
        raise ValueError(f"No token found in token file: {token_file}")

    # Build credential and authenticated session
    cred = pyvo.auth.CredentialStore()
    cred.set_password("x-oauth-basic", token)
    session = cred.get("ivo://ivoa.net/sso#BasicAA")
    if session is None:
        raise RuntimeError("Failed to obtain authenticated session. Check your token permissions.")

    # Instantiate TAPService
    rsp_tap_url = 'https://data.lsst.cloud/api/tap'
    rsp_tap = pyvo.dal.TAPService(rsp_tap_url, session=session)
    if rsp_tap.baseurl != rsp_tap_url:
        raise RuntimeError(
            f"rubin_authenticate: unexpected TAPService.baseurl "
            f"'{rsp_tap.baseurl}', expected '{rsp_tap_url}'"
        )
    return rsp_tap


def rubin_get_objectids(sample_table, rsp_tap, search_radius=0.001):
    """
    Perform cone searches for each row in sample_table to retrieve matching DP0 object IDs.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Input table containing at minimum the following columns:
            coord : astropy.coordinates.SkyCoord
                Sky position for each object.
            objectid : int
                Integer identifier assigned to each input source.
            label : str
                Source label, provenance tag, or literature reference.
    rsp_tap : pyvo.dal.TAPService
        Authenticated TAP service instance returned by `rubin_authenticate()`.
    search_radius : float, deg
        Radius of the cone search in degrees (default: 0.001).

    Returns
    -------
    combined : astropy.table.Table
         Table containing Rubin matches with columns:
            coord_ra : float
                RA of the matched DP0 Object (deg).
            coord_dec : float
                Dec of the matched DP0 Object (deg).
            objectId : int
                Rubin DP0 Object identifier.
            in_objid : int
                Original objectid from the sample table.
            in_label : str
                Original label from the sample table.
        Returns None if no matches are found.

    Notes
    -----
    Only DP0 objects with detect_isPrimary = 1 are returned. If multiple
    DP0 objects fall within the cone, all are returned.

    """
    # could consider parallelizing this but for now we wait for TAP table upload and real data
    all_tables = []
    for row in sample_table:
        ra = row['coord'].ra.deg
        dec = row['coord'].dec.deg
        in_objid = row['objectid']
        in_label = row['label']

        query = (f"""
        SELECT coord_ra, coord_dec, objectId
        FROM dp02_dc2_catalogs.Object
        WHERE CONTAINS(
            POINT('ICRS', coord_ra, coord_dec),
            CIRCLE('ICRS', {ra}, {dec}, {search_radius})
        ) = 1
        AND detect_isPrimary = 1
        """).strip()
        tbl = rsp_tap.run_sync(query).to_table()
        # skip if no matches
        if len(tbl) == 0:
            continue
        tbl['in_objid'] = in_objid
        tbl['in_label'] = in_label
        all_tables.append(tbl)

    # in the case of an empty table, return an empty table
    if not all_tables:
        return

    # otherwise, stack all the tables together
    combined = vstack(all_tables)
    return combined


def rubin_access_lc_catalog(object_table, rsp_tap):
    """
    Retrieve forced-source light curves for the given DP0 Object IDs and attach
    the original sample metadata. This function executes a TAP query against the
    DP0.2 `ForcedSource` and `CcdVisit` tables, merges the results with the input
    sample_table identifiers, and returns the result as a MultiIndexDFObject.

    Parameters
    ----------
    object_table : astropy.table.Table
        Table containing:
            objectId : int
                Rubin DP0 Object identifier.
            in_objid : int
                Original objectid from the input sample table.
            in_label : str
                Original label from the input sample table.
    rsp_tap : pyvo.dal.TAPService
        Authenticated TAP service from rubin_authenticate().

    Returns
    -------
    df_lc : MultiIndexDFObject
        Indexed by [objectid, label, band, time].  
        The resulting internal pandas DataFrame contains the following columns:

            band : str
                Rubin filter name, prefixed with 'rubin_' (e.g., 'rubin_g', 'rubin_r').
            ccdVisitId : int
                Visit identifier of the CCD exposure used for the forced measurement.
            coord_ra : float
                Source right ascension (deg) at the time of measurement.
            coord_dec : float
                Source declination (deg) at the time of measurement.
            psfFlux : float
                PSF flux measurement in nanojanskys (nJy).
            psfFluxErr : float
                Uncertainty on psfFlux (nJy).
            psfMag : float
                AB magnitude computed from psfFlux by `scisql_nanojanskyToAbMag()`.
            visitId : int
                Identifier of the parent visit from the CcdVisit table.
            zeroPoint : float
                Zero point used to calibrate the flux measurement.
            time : float
                Midpoint of the exposure in MJD (taken from `expMidptMJD`).
            objectid : int
                Input sample object identifier (from `in_objid`).
            label : str
                Literature label associated with each input source (from `in_label`).

    Notes
    -----
    - The returned dataframe preserves *all* flux-related Rubin columns; no
    flux conversion is applied.
    - The 'objectId' column is dropped because it is replaced by 'objectid'
    from the input sample.
    - All time values are in MJD, consistent with other archives.
   """
    # Pull out a unique, sorted list of integer IDs:
    objids = sorted(set(object_table['objectId']))

    # build the string for the query below
    # will probably need to break this into chunks when working with large samples (> 50)
    id_tuple_str = f"({','.join(map(str, objids))})"

    query_lc = (f"""
    SELECT src.band, src.ccdVisitId, src.coord_ra, src.coord_dec,
           src.objectId, src.psfFlux, src.psfFluxErr,
           scisql_nanojanskyToAbMag(src.psfFlux) AS psfMag,
           vis.ccdVisitId AS visitId,
           vis.expMidptMJD, vis.zeroPoint
    FROM dp02_dc2_catalogs.ForcedSource AS src
    JOIN dp02_dc2_catalogs.CcdVisit AS vis
      ON vis.ccdVisitId=src.ccdVisitId
    WHERE src.objectId IN {id_tuple_str} AND src.detect_isPrimary=1
    """).strip()

    # run the query on the tap server
    srcs = rsp_tap.run_sync(query_lc).to_table()

    # Convert the result to pandas
    df_src = srcs.to_pandas()

    # Prefix every band value with "rubin_"
    df_src['band'] = 'rubin_' + df_src['band'].astype(str)

    # Convert object_table (Astropy) to pandas, keep only the three cols
    df_obj = (
        object_table[['objectId', 'in_objid', 'in_label']]
        .to_pandas()
    )
    # Merge on the Rubin objectId to bring in your sample metadata
    df_merged = df_src.merge(
        df_obj,
        on='objectId',
        how='left'
    )
    # Drop the raw objectId column; otherwise there are two in returned df
    df_merged = df_merged.drop(columns=['objectId'])

    # Rename columns and set a MultiIndex
    df_merged = (
        df_merged
        .rename(columns={'in_objid': 'objectid', 'in_label': 'label', 'expMidptMJD': 'time'})
        .set_index(['objectid', 'label', 'band', 'time'], drop=True)
    )        # explicitly drop=True (default) so the renamed 'objectid' column

    # 6. Wrap in your MultiIndexDFObject
    df_lc = MultiIndexDFObject(df_merged)
    return df_lc


def rubin_get_lightcurves(sample_table, search_radius=0.001):
    """
    Main entry point: authenticate, get object IDs, and fetch light curves.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table containing the input sample with required columns:
            coord : astropy.coordinates.SkyCoord
                Sky position of each source.
            objectid : int
                Unique integer ID for each input source.
            label : str
                Label or reference tag carried through to the output.
    search_radius : float, optional
        Cone search radius in degrees.

    Returns
    -------
    df_lc : MultiIndexDFObject
        Indexed by [objectid, label, band, time].
        The resulting internal pandas DataFrame contains the following columns:

            psfFlux : float
                Measured PSF flux for the forced-source detection (in nanojanskys).
            psfFluxErr : float
                Uncertainty of the PSF flux (in nanojanskys).
            psfMag : float
                AB magnitude corresponding to the measured PSF flux.
            coord_ra : float
                Right Ascension of the forced-source measurement (degrees).
            coord_dec : float
                Declination of the forced-source measurement (degrees).
            ccdVisitId : int
                Unique visit/exposure identifier for the detection.
            visitId : int
                Visit identifier returned by the CcdVisit table.
            zeroPoint : float
                Photometric zeropoint used for the magnitude calibration.
            objectid : int
                Identifier from the original sample_table.
            band : str
                Rubin filter name (for example, 'rubin_r', 'rubin_i').
            label : str
                Literature label associated with each source.
            time : float
                Midpoint of the exposure in Modified Julian Date (MJD).

        These columns are obtained from the Rubin DP0 ForcedSource and CcdVisit
        catalogs and merged with the original sample identifiers.
    
    Raises
    ------
    FileNotFoundError
        If the Rubin TAP token file is missing or empty.
    RuntimeError
        If TAP authentication or queries fail.
    """
    # authenticate and set up TAP service
    rsp_tap = rubin_authenticate()

    # get the Rubin objectids which correspond to our coordinates
    obj_table = rubin_get_objectids(sample_table, rsp_tap, search_radius)

    if not obj_table:
        return MultiIndexDFObject()

    # use those objectids to get the time domain info
    lc = rubin_access_lc_catalog(obj_table, rsp_tap)

    return lc
