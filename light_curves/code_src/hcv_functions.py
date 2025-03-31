import pandas as pd
import requests
from astropy.table import Table
from data_structures import MultiIndexDFObject
from fluxconversions import convertACSmagtoflux

# Functions related to the HCV.
# Code partially taken from https://archive.stsci.edu/hst/hsc/help/HCV/HCV_API_demo.html


def get_hscapiurl():
    """ Return the url to use for the HSC API

    Returns
    -------
    the HSC API Url

    """

    hscapiurl = "https://catalogs.mast.stsci.edu/api/v0.1/hsc"
    return (hscapiurl)


def hcvcone(ra, dec, radius, table="hcvsummary", release="v3", format="csv", magtype="magaper2",
            columns=None, baseurl=get_hscapiurl(), verbose=False,
            **kw):
    """Do a cone search of the HSC catalog (including the HCV)

    Parameters
    ----------
    ra: float (degrees)
        J2000 Right Ascension
    dec: float (degrees)
        J2000 Declination
    radius: float (degrees)
        Search radius (<= 0.5 degrees)
    table: string
        hcvsummary, hcv, summary, detailed, propermotions, or sourcepositions
    release: string
        v3 or v2
    format: string
        csv, votable, json
    magtype: string
        magaper2 or magauto (only applies to summary table)
    columns: list of strings
        list of column names to include (None means use defaults)
    baseurl: string
        base URL for the request
    verbose: int
        print info about request
    **kw: other parameters (e.g., 'numimages.gte':2)

    Returns
    -------
    search results: Table

    """

    data = kw.copy()
    data['ra'] = ra
    data['dec'] = dec
    data['radius'] = radius
    return hcvsearch(table=table, release=release, format=format, magtype=magtype,
                     columns=columns, baseurl=baseurl, verbose=verbose, **data)


def hcvsearch(table="hcvsummary", release="v3", magtype="magaper2", format="csv",
              columns=None, baseurl=get_hscapiurl(), verbose=False,
              **kw):
    """Do a general search of the HSC catalog (possibly without ra/dec/radius)

    Parameters
    ----------
    table: string
        hcvsummary, hcv, summary, detailed, propermotions, or sourcepositions
    release: string
        v3 or v2
    magtype: string
        magaper2 or magauto (only applies to summary table)
    format: string
        csv, votable, json
    columns: list of strings
        list of column names to include (None means use defaults)
    baseurl: string
        base URL for the request
    verbose: int
        print info about request
    **kw: other parameters (e.g., 'numimages.gte':2).  Note this is required!

    Returns
    -------
    search results
    """

    data = kw.copy()
    if not data:
        raise ValueError("You must specify some parameters for search")
    if format not in ("csv", "votable", "json"):
        raise ValueError("Bad value for format")
    url = "{}.{}".format(cat2url(table, release, magtype, baseurl=baseurl), format)
    if columns:
        # check that column values are legal
        # create a dictionary to speed this up
        dcols = {}
        for col in hcvmetadata(table, release, magtype)['name']:
            dcols[col.lower()] = 1
        badcols = []
        for col in columns:
            if col.lower().strip() not in dcols:
                badcols.append(col)
        if badcols:
            raise ValueError('Some columns not found in table: {}'.format(', '.join(badcols)))
        # two different ways to specify a list of column values in the API
        # data['columns'] = columns
        data['columns'] = '[{}]'.format(','.join(columns))

    # either get or post works
    # r = requests.post(url, data=data)
    r = requests.get(url, params=data)

    if verbose:
        print(r.url)
    r.raise_for_status()
    if format == "json":
        return r.json()
    else:
        return r.text


def hcvmetadata(table="hcvsummary", release="v3", magtype="magaper2", baseurl=get_hscapiurl()):
    """Return metadata for the specified catalog and table

    Parameters
    ----------
    table: string
        hcvsummary, hcv, summary, detailed, propermotions, or sourcepositions
    release: string
        v3 or v2
    magtype: string
        magaper2 or magauto (only applies to summary table)
    baseurl: string
        base URL for the request

    Returns
    -------
    astropy table with columns name, type, description
    """
    url = "{}/metadata".format(cat2url(table, release, magtype, baseurl=baseurl))
    r = requests.get(url)
    r.raise_for_status()
    v = r.json()
    # convert to astropy table
    tab = Table(rows=[(x['name'], x['type'], x['description']) for x in v],
                names=('name', 'type', 'description'))
    return tab


def cat2url(table="hcvsummary", release="v3", magtype="magaper2", baseurl=get_hscapiurl()):
    """Return URL for the specified catalog and table

    Parameters
    ----------
    table : string
        hcvsummary, hcv, summary, detailed, propermotions, or sourcepositions
    release: string
        v3 or v2
    magtype: string
        magaper2 or magauto (only applies to summary table)
    baseurl: string
        base URL for the request

    Returns
    -------
    base URL for this request: string
    """
    checklegal_hcv(table, release, magtype)
    if table == "summary":
        url = "{baseurl}/{release}/{table}/{magtype}".format(**locals())
    else:
        url = "{baseurl}/{release}/{table}".format(**locals())
    return url


def checklegal_hcv(table, release, magtype):
    """Checks if this combination of table and release is acceptable
    Raises a ValueError exception if there is problem

    Parameters
    ----------
    table: string list
        "summary", "detailed", "propermotions", "sourcepositions",
                    "hcvsummary", "hcv"
    release: string
        v2 or v3
    magtype:string
        "magaper2", "magauto"

    """

    releaselist = ("v2", "v3")
    if release not in releaselist:
        raise ValueError("Bad value for release (must be one of {})".format(
            ', '.join(releaselist)))
    if release == "v2":
        tablelist = ("summary", "detailed")
    else:
        tablelist = ("summary", "detailed", "propermotions", "sourcepositions",
                     "hcvsummary", "hcv")
    if table not in tablelist:
        raise ValueError("Bad value for table (for {} must be one of {})".format(
            release, ", ".join(tablelist)))
    if table == "summary":
        magtypelist = ("magaper2", "magauto")
        if magtype not in magtypelist:
            raise ValueError("Bad value for magtype (must be one of {})".format(
                ", ".join(magtypelist)))


def hcv_get_lightcurves(sample_table, *, radius=1/3600):
    """Searches Hubble Catalog of variables for light curves from a list of input coordinates

    Parameters
    ----------
    sample_table:  `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    radius : float
        search radius, how far from the source should the archives return results

    Returns
    -------
    df_lc : MultiIndexDFObject
        the main data structure to store all light curves
    """

    df_lc = MultiIndexDFObject()

    for row in sample_table:

        ra = row['coord'].ra.deg
        dec = row['coord'].dec.deg
        lab = row['label']
        objectid = row['objectid']

        # IC 1613 from the demo for testing
        # ra = 16.19913
        # dec = 2.11778

        # look in the summary table for anything within a radius of our targets
        tab = hcvcone(ra, dec, radius, table="hcvsummary")
        if tab == '':
            continue

        tab = Table.read(tab, format='ascii')

        matchid = tab['MatchID'][0]  # take the first one, assuming it is the nearest match

        # just pulling one filter for an example (more filters are available)
        try:
            src_814 = Table.read(hcvsearch(table='hcv', MatchID=matchid,
                                 Filter='ACS_F814W'), format='ascii')
        except FileNotFoundError:
            # that filter doesn't exist for this target
            continue

        time_814 = src_814['MJD']
        mag_814 = src_814['CorrMag']  # need to convert this to flux
        magerr_814 = src_814['MagErr']

        filterstring = 'F814W'

        # uggg, ACS has time dependent flux zero points.....
        # going to cheat for now and only use one time, but could imagine this as a loop
        # https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
        flux, fluxerr = convertACSmagtoflux(time_814[0], filterstring, mag_814, magerr_814)
        flux = flux * 1E3  # convert to mJy
        fluxerr = fluxerr*1E3  # convert to mJy

        # put this single object light curves into a pandas multiindex dataframe
        dfsingle_814 = pd.DataFrame(dict(flux=flux, err=fluxerr, time=time_814, objectid=objectid,
                                    band='F814W', label=lab)).set_index(["objectid", "label", "band", "time"])

        # then concatenate each individual df together
        df_lc.append(dfsingle_814)

    return (df_lc)
