import astropy.units as u
from alerce.core import Alerce
from astropy.coordinates import SkyCoord
from astropy.table import Table, join, join_skycoord, unique
from astroquery.ipac.ned import Ned
from astroquery.sdss import SDSS
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from requests.exceptions import ConnectionError


# lamassa et al., 2015  1 source
def get_lamassa_sample(coords, labels, *, verbose=1):
    """    
    Append coordinates and labels for the LaMassa et al. (2015) CLAGN sample.

    This function performs a NED reference-code query to retrieve the
    coordinates of the LaMassa+2015 changing-look AGN and appends them to
    the shared `coords` and `labels` lists.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
        
    Returns
    -------
    None
        The input lists are modified in place.

    """
    lamassa_CLQ = Ned.query_refcode('2015ApJ...800..144L')
    lamassa_CLQ = lamassa_CLQ[0]  # dont know what those other targets are.
    coords.append(SkyCoord(lamassa_CLQ['RA'], lamassa_CLQ['DEC'], frame='icrs', unit='deg'))
    labels.append('LaMassa 15')
    if verbose:
        print('Changing Look AGN- Lamassa et al: ', len(labels))


# MacLeod et al., 2016
def get_macleod16_sample(coords, labels, *, verbose=1):
    """    
    Append coordinates and labels for the MacLeod et al. (2016) CLAGN sample.

    Coordinates are parsed from the HTML table hosted by the journal website
    and converted into SkyCoord objects.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    Returns
    -------
    None
        The input lists are modified in place.
    """
    macleod_CSQ = Table.read('https://academic.oup.com/mnras/article/457/1/389/989199',
                             htmldict={'table_id': 5}, format='ascii.html')

    # get coords from "name" column for this
    for i in range(len(macleod_CSQ)):
        coord_str = macleod_CSQ['Name\n            .'][i]
        test_str = coord_str[0:2] + " " + coord_str[2:4] + " " + coord_str[4:9] + \
            " " + coord_str[9:12] + " " + coord_str[12:14] + " " + coord_str[14:]
        coords.append(SkyCoord(test_str, unit=(u.hourangle, u.deg)))
        labels.append('MacLeod 16')
    if verbose:
        print('Changing Look AGN- MacLeod et al 2016: ', len(macleod_CSQ))


# Ruan et al., 2016  3 sources
def get_ruan_sample(coords, labels, *, verbose=1):
    """
    Append coordinates and labels for the Ruan et al. (2016) CLAGN sample.
    
    This function performs a NED reference-code query to retrieve the
    coordinates
    
    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    Returns
    -------
    None
    """
    ruan_CSQ = Ned.query_refcode('2016ApJ...826..188R')

    ruan_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg')
                   for ra, dec in zip(ruan_CSQ['RA'], ruan_CSQ['DEC'])]
    ruan_labels = ['Ruan 16' for ra in ruan_CSQ['RA']]
    coords.extend(ruan_coords)
    labels.extend(ruan_labels)
    # one of these is a repeat of lamassa et al.
    if verbose:
        print('Changing Look AGN- Ruan et al 2016: ', len(ruan_CSQ['RA']))


# MacLeod et al., 2019 17 sources
def get_macleod19_sample(coords, labels, *, verbose=1):
    """
    Append coordinates and labels for the MacLeod et al. (2019) CLAGN sample.

    Coordinates are parsed from Vizier service
    
    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
        
    Returns
    -------
    None

    """
    # try vizier
    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('2019ApJ...874....8M')
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    table2 = catalogs[0]  # more than one table

    # filter for the actual CLQs
    macleod19_CSQ = table2[(table2['CLQ_'] > 0) & (table2['Nsigma'] > 3)]

    macleod19_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg')
                        for ra, dec in zip(macleod19_CSQ['_RA'], macleod19_CSQ['_DE'])]
    macleod19_labels = ['MacLeod 19' for ra in macleod19_CSQ['_RA']]
    coords.extend(macleod19_coords)
    labels.extend(macleod19_labels)
    if verbose:
        print('Changing Look AGN- MacLeod et al 2017: ', len(macleod19_CSQ['_RA']))


# sheng et al., 2020
def get_sheng_sample(coords, labels, *, verbose=1):
    """
    Append coordinates and labels for the Sheng et al. (2020) CLAGN sample.

    This function performs a NED reference-code query to retrieve the
    coordinates
    
    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source

    Returns
    -------
    None
    """
    CLQ = Ned.query_refcode('2020ApJ...889...46S')
    sheng_CLQ = CLQ[[0, 1, 3]]  # need the first 3 objects in their table,
    sheng_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg')
                    for ra, dec in zip(sheng_CLQ['RA'], sheng_CLQ['DEC'])]
    sheng_labels = ['Sheng 20' for ra in sheng_CLQ['RA']]

    coords.extend(sheng_coords)
    labels.extend(sheng_labels)
    if verbose:
        print('Changing Look AGN- Sheng et al 2020: ', len(sheng_CLQ['RA']))


# green et al., 2022  19 sources
def get_green_sample(coords, labels, *, verbose=1):
    """
    Append coordinates and labels for the Green et al. (2022) CLAGN sample.

    Coordinates are parsed from Vizier service

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source

    Returns
    -------
    None
    """

    # try vizier
    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/ApJ/933/180')
    catalogs = Vizier.get_catalogs(catalog_list.keys())
    table2 = catalogs[0]

    # go to pandas to manipulate the table
    green_CSQ = table2.to_pandas()

    # filter only those that are confirmed CLQ in the notes column
    green_CSQ = green_CSQ[green_CSQ['Notes'].str.contains("CLQ", na=False)]

    # pick out the coordinates from the 'SDSS' column
    coord_str = green_CSQ['SDSS']
    coord_str.astype('string')
    test_str = coord_str.str[1:3] + " " + coord_str.str[3:5] + " " + coord_str.str[5:10] + \
        " " + coord_str.str[10:13] + " " + coord_str.str[13:15] + " " + coord_str.str[15:]
    green_labels = ['Green 22' for ra in green_CSQ['SDSS']]

    # convert from pandas series to list as input to SkyCoord
    coords.extend(SkyCoord(test_str.values.tolist(), unit=(u.hourangle, u.deg)))
    labels.extend(green_labels)
    if verbose:
        print('Changing Look AGN- Green et al 2022: ', len(green_labels))

# Lyu et al., 2021  lists a known sample of 68 sources to date!!!


def get_lyu_sample(coords, labels, *, verbose=1):
    """
    Append coordinates and labels for the Lyu et al. (2021/2022) CLAGN sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source

    Returns
    -------
    None
    """
    CLQ = Ned.query_refcode('2022ApJ...927..227L')
    lyu_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg')
                  for ra, dec in zip(CLQ['RA'], CLQ['DEC'])]
    lyu_labels = ['Lyu 22' for ra in CLQ['RA']]

    coords.extend(lyu_coords)
    labels.extend(lyu_labels)
    if verbose:
        print('Changing Look AGN- Lyu et al 2021: ', len(CLQ['RA']))


# Lopez-navas et al., 2022
def get_lopeznavas_sample(coords, labels, *, verbose=1):
    """
    Append coordinates and labels for the López-Navas et al. (2022) CLAGN sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source

    Returns
    -------
    None
    """
    result_table = Simbad.query_bibobj('2022MNRAS.513L..57L')
    result_table = result_table[[0, 1, 2, 3]]  # pick the correct sources by hand

    ln_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg')
                 for ra, dec in zip(result_table['RA'], result_table['DEC'])]
    ln_labels = ['Lopez-Navas 22' for ra in result_table['RA']]

    coords.extend(ln_coords)
    labels.extend(ln_labels)
    if verbose:
        print('Changing Look AGN- Lopez-navas et al 2022: ', len(result_table['RA']))


# Hon et al., 2022
def get_hon_sample(coords, labels, *, verbose=1):
    """
    Append coordinates and labels for the Hon et al. (2022) CLAGN sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list

        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source

    Returns
    -------
    None
    """
    CLQ = Ned.query_refcode('2022MNRAS.511...54H')
    hon_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg')
                  for ra, dec in zip(CLQ['RA'], CLQ['DEC'])]
    hon_labels = ['Hon 22' for ra in CLQ['RA']]

    coords.extend(hon_coords)
    labels.extend(hon_labels)
    if verbose:
        print('Changing Look AGN- Hon et al 2022: ', len(CLQ['RA']))

# yang et al., 2018


def get_yang_sample(coords, labels, *, verbose=1):
    """
    Append coordinates and labels for the Yang et al. (2018) CLAGN sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source

    Returns
    -------
    None
    """
    CLQ = Ned.query_refcode('2018ApJ...862..109Y')
    yang_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg')
                   for ra, dec in zip(CLQ['RA'], CLQ['DEC'])]
    yang_labels = ['Yang 18' for ra in CLQ['RA']]

    coords.extend(yang_coords)
    labels.extend(yang_labels)
    if verbose:
        print('Changing Look AGN- Yang et al: ', len(CLQ['RA']))


# Here are additional CLAGN samples
# but not spectroscopically confirmed
# Sanchez Saez et al., 2021
def get_sanchezsaez_sample(coords, labels, *, verbose=1):
    """
    Append coordinates and labels for the Sánchez-Sáez et al. (2021) CLAGN sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source

    Returns
    -------
    None
    """

    CSAGN = Ned.query_refcode('2021AJ....162..206S')  # from Sanchez-Saez 2021

    ss_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg')
                 for ra, dec in zip(CSAGN['RA'], CSAGN['DEC'])]
    ss_labels = ['Sanchez-Saez 21' for ra in CSAGN['RA']]
    coords.extend(ss_coords)
    labels.extend(ss_labels)
    if verbose:
        print('Changing Look AGN- Sanchez et al: ', len(CSAGN['RA']))

# Graham et al., 2019


def get_graham_sample(coords, labels, *, verbose=1):
    """
    Append coordinates and labels for the Graham et al. (2019) CLAGN sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source

    Returns
    -------
    None
    """
    # use astropy table to get larger sample that neither NED nor astropy can access.
    CSQ = Table.read('https://academic.oup.com/mnras/article/491/4/4925/5634279',
                     htmldict={'table_id': 5}, format='ascii.html')

    # get coords from "name" column for this
    for i in range(len(CSQ)):
        coord_str = CSQ['Name\n            .'][i]
        test_str = coord_str[6:8] + " " + coord_str[8:10] + " " + coord_str[10:14] + \
            " " + coord_str[14:17] + " " + coord_str[17:19] + " " + coord_str[19:]
        coords.append(SkyCoord(test_str, unit=(u.hourangle, u.deg)))
        labels.append('Graham 19')
    if verbose:
        print('Changing Look AGN- Graham et al: ', len(CSQ))


def get_ztf_objectid_sample(coords, labels, *, objectids=["ZTF18aabtxvd", "ZTF18aahqkbt"], verbose=1):
    """ 
    Append coordinates for sources known only by their ZTF object IDs.

    Coordinates are retrieved via the ALeRCE API and added to the provided
    `coords` and `labels` lists.

    Parameters
    ----------
    coords : list of astropy skycoords
        the coordinates of the targets for which a user wants light curves
    labels: list of strings
        journal articles associated with the target coordinates
    objectids: list of strings
        List of ZTF objectid. eg., [ "ZTF18accqogs", "ZTF19aakyhxi", "ZTF19abyylzv", "ZTF19acyfpno"]
    verbose: int
        print out debugging info (1) or not(0)

    Returns
    -------
    None
    """
    alerce = Alerce()
    objects = alerce.query_objects(oid=objectids, format="pandas")
    tde_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg')
                  for ra, dec in zip(objects['meanra'], objects['meandec'])]
    tde_labels = ['ZTF-Objname' for _ in objects['meanra']]
    coords.extend(tde_coords)
    labels.extend(tde_labels)
    if verbose:
        print('number of ztf coords added by Objectname:', len(objects['meanra']))


# SDSS QSO sample of any desired number
# These are "normal" QSOs to use in the classifier
def get_sdss_sample(coords, labels, *, num=10, zmin=0, zmax=10, randomize_z=False, verbose=1):
    """
    Append coordinates and labels for SDSS quasars selected within a redshift range.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    num : int < 500K
        How many quasars for which to return the coords and labels
    zmin : float
        Lower limit (non-inclusive) redshift of sources to return.
    zmax : float
        Upper limit (non-inclusive) redshift of sources to return.
    randomize_z : bool
        Whether to sample randomly within the range zmin to zmax. By default (`False`), this will return
        the `num` SDSS sources with the smallest redshift (within the range). To get a sample with
        randomized redshifts, set this to `True`, but beware that the returned sample will be random
        and thus not reproducible.
    verbose : int, optional
        Print out the length of the sample derived from this literature source

    Returns
    -------
    None
    """
    # Define the query
    query = "SELECT TOP " + str(num) + " specObjID, ra, dec, z FROM SpecObj \
    WHERE ( z > " + str(zmin) + "AND z < " + str(zmax) + " AND class='QSO' AND zWARNING=0 )"
    if randomize_z:
        query = query + " ORDER BY NEWID()"

    # use astroquery to return an astropy table of results
    if num > 0:
        res = SDSS.query_sql(query, data_release=16)

        SDSS_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg')
                       for ra, dec in zip(res['ra'], res['dec'])]
        SDSS_labels = ['SDSS' for ra in res['ra']]

        coords.extend(SDSS_coords)
        labels.extend(SDSS_labels)

    if verbose:
        print('SDSS Quasar: ' + str(num))


def get_paper_sample(coords, labels, *, paper_link="2019A&A...627A..33D", label="Cicco19", verbose=1):
    """
    Append coordinates and labels from a reference-code query to NED.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source

    Returns
    -------
    None
    """
    try:
        paper = Ned.query_refcode(paper_link)
    except ConnectionError:
        print(f"WARNING: encountered a ConnectionError error for paper {paper_link}. skipping.")
        return

    paper_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg')
                    for ra, dec in zip(paper['RA'], paper['DEC'])]
    paper_labels = [label for ra in paper['RA']]
    coords.extend(paper_coords)
    labels.extend(paper_labels)
    if verbose:
        print("number of sources added from " + str(label) + " :" + str(len(paper_coords)))


def get_papers_list_sample(coords, labels, *, paper_kwargs=[dict(),]):
    """
    Wrapper for get_paper_sample. Calls get_paper_sample for each item in paper_kwargs.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    paper_kwargs : list[dict]
        List of dicts containing keyword arguments passed on to get_paper_sample.

    Returns
    -------
    None
    """
    # loop over the papers in paper_kwargs and call get_paper_sample for each
    for kwargs in paper_kwargs:
        get_paper_sample(coords, labels, **kwargs)


def get_csv_sample(coords, labels, *, csv_path, label, ra_colname="ra", dec_colname="dec", frame="icrs", unit="deg"):
    """
    Loads coordinates from file at `csv_path` and adds them to `coords` and `labels`

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    labels : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    csv_path : str
        Path to a csv file containing (at least) columns
    label : str
        The label to apply to these coordinates.
    ra_colname : str
        Name of the column containing the RA coord.
    dec_colname : str
        Name of the column containing the Dec coord.
    frame : str
        Coordinate frame to pass to SkyCoord.
    unit : str
        Coordinate unit to pass to SkyCoord.

    Returns
    -------
    None
    """
    csv_sample = Table.read(csv_path)[ra_colname, dec_colname]
    coords.extend([SkyCoord(ra, dec, frame=frame, unit=unit)
                  for (ra, dec) in csv_sample.iterrows()])
    labels.extend([label] * len(csv_sample))


def clean_sample(coords_list, labels_list, *, consolidate_nearby_objects=True, verbose=1):
    """
    Construct a cleaned sample table with unique on-sky sources.

    Combines coordinate and label lists into an Astropy Table and optionally
    merges entries within 0.005 deg (~18 arcsec). Each retained source is
    assigned a unique integer `objectid`, starting at 1.

    Parameters
    ----------
    coords_list : list
        list of Astropy SkyCoords derived from literature sources
    labels_list : list
        List of the first author name and publication year for tracking the sources
    consolidate_nearby_objects : bool
        When True (default), merge objects that are closer than 0.005 deg
        (~18 arcsec) using an on-sky join. All coordinates within this
        angular separation are treated as the same source and assigned a
        single `objectid`.

        When False, each entry in `coords_list` becomes a separate row and
        receives a unique objectid regardless of separation.
    verbose : int, optional
        Print out the length of the sample after applying this function

    Returns
    -------
    sample_table : astropy.table.Table
        Table containing the source sample. The following columns must be present:
            coord : astropy.coordinates.SkyCoord
                Sky position of each source.
            objectid : int
                Unique identifier for each source in the sample.
            label : str
                Literature label for tracking source provenance.        
    
    The returned table has one row per retained object after optional
        consolidation.
    """

    sample_table = Table([coords_list, labels_list], names=['coord', 'label'])

    if not consolidate_nearby_objects:
        # create a range 'objectid'. must start with 1 to match what the astropy `join` produces below.
        nsample = len(sample_table)
        sample_table['objectid'] = list(range(1, nsample + 1))
        print(f'Object sample size: {nsample}')
        return sample_table['objectid', 'coord', 'label']

    # now join the table with itself within a defined radius.
    # We keep one set of original column names to avoid later need for renaming
    tjoin = join(sample_table, sample_table, keys='coord',
                 join_funcs={'coord': join_skycoord(0.005 * u.deg)},
                 uniq_col_name='{col_name}{table_name}', table_names=['', '_2'])

    # this join will return 4 entries for each redundant coordinate:
    # 1 for the match with itself and 1 for the match with the similar
    # enough coord target then the same thing again for the match, but all of
    # these 4 will have the same id in the new 'coords_id' column

    # keep only those entries in the resulting table which are unique
    uniqued_table = unique(tjoin, keys='coord_id')['coord_id', 'coord', 'label']
    uniqued_table.rename_column('coord_id', 'objectid')

    if verbose:
        print(f'Object sample size, after duplicates removal: {len(uniqued_table)}')

    return uniqued_table
def validate_sample_table(tbl):
    """
    Validate the structure of a user-supplied ``sample_table``.

    This function verifies that the supplied object is an
    ``astropy.table.Table`` and that it contains the required
    columns (``coord``, ``objectid``, ``label``). It checks that
    ``coord`` entries are ``SkyCoord`` objects, that ``objectid``
    values are integers, unique, and sequential starting at 1, and
    that ``label`` contains strings. Informative exceptions are
    raised if validation fails.
    
    Parameters
    ----------
    tbl : astropy.table.Table
        Table to validate. Must represent a cleaned source sample
        with one row per sky object.

    Raises
    ------
    ValueError
        If required columns are missing or ``objectid`` values are
        non-unique or non-sequential.
    TypeError
        If column types are not as expected (e.g., ``coord`` not
        containing ``SkyCoord`` objects).

    """
    from astropy.coordinates import SkyCoord
    
    # Check that tbl is an Astropy Table
    if not isinstance(tbl, Table):
        raise TypeError("Input must be an astropy.table.Table instance.")


    required = {"coord", "objectid", "label"}
    missing = required - set(tbl.colnames)
    if missing:
        raise ValueError(f"sample_table is missing required columns: {missing}")

    # coord type check
    if not all(isinstance(c, SkyCoord) for c in tbl["coord"]):
        raise TypeError("Column 'coord' must contain astropy.coordinates.SkyCoord objects.")

    # objectid checks
    obj = tbl["objectid"]
    if not (obj.dtype.kind in ("i", "u")):
        raise TypeError("Column 'objectid' must contain integers.")

    if len(set(obj)) != len(obj):
        raise ValueError("Column 'objectid' must contain unique values.")

    if sorted(obj) != list(range(1, len(obj) + 1)):
        raise ValueError("objectid must start at 1 and increment without gaps.")

    # label check
    if not all(isinstance(x, str) for x in tbl["label"]):
        raise TypeError("Column 'label' must contain strings.")

    print("sample_table format is valid.")