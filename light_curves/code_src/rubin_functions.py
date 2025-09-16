from pathlib import Path

import pyvo
from astropy.table import vstack

from data_structures import MultiIndexDFObject


def rubin_authenticate():
    """
    Authenticate to the Rubin Science Platform TAP service using a locally stored token.

    Returns
    -------
    rsp_tap : pyvo.dal.TAPService
        An authenticated TAPService instance pointing to DP0 data, assigned to variable `rsp_tap`.

    Raises
    ------
    FileNotFoundError
        If the token file does not exist or is empty.
    RuntimeError
        If the TAP session cannot be created or the returned baseurl mismatches the expected URL.
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
        Table with columns ['coord', 'objectid', 'label'], where 'coord' is a SkyCoord.
    rsp_tap : pyvo.dal.TAPService
        Authenticated TAP service instance returned by `rubin_authenticate()`.
    search_radius : float, deg
        Radius of the cone search in degrees (default: 0.001).

    Returns
    -------
    combined : astropy.table.Table
        Combined table with columns ['coord_ra', 'coord_dec', 'objectId',
        'in_objid', 'in_label'] tagging each DP0 Object match with the input sample info.
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
    Retrieve forced-source light curves for given DP0 object IDs and attach sample metadata.

    Parameters
    ----------
    object_table : astropy.table.Table
        Table with ['objectId', 'in_objid', 'in_label'] from get_objectids().
    rsp_tap : pyvo.dal.TAPService
        Authenticated TAP service from rubin_authenticate().

    Returns
    -------
    MultiIndexDFObject
        Indexed by ['objectid', 'label', 'band', 'time'] with flux data.
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
        Table with ['coord', 'objectid', 'label'] input sample.
    search_radius : float, optional
        Cone search radius in degrees.

    Returns
    -------
    MultiIndexDFObject
        Light curves indexed by ['objectid', 'label', 'band', 'time'].

    Raises
    ------
    FileNotFoundError
        If the RSP token file is missing or contains no token.
    RuntimeError
        If authentication fails or any of the TAP queries encounters an error.
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
