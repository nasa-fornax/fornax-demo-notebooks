import multiprocessing as mp
import re

import astropy.units as u
import pandas as pd
import pyvo
import s3fs
from tqdm import tqdm

from data_structures import MultiIndexDFObject
from sample_selection import make_coordsTable

# the catalog is stored in an AWS S3 bucket
DATARELEASE = "dr18"
BUCKET = "irsa-mast-tike-spitzer-data"
CATALOG_ROOT = f"{BUCKET}/data/ZTF/lc/lc_{DATARELEASE}/"

# get a list of files in the dataset using the checksums file
CATALOG_FILES = (
    pd.read_table(f"s3://{CATALOG_ROOT}checksum.md5", sep="\s+", names=["md5", "path"], usecols=["path"])
    .squeeze()  # there's only 1 column. squeeze it into a Series
    .str.removeprefix("./")
    .to_list()
)


def ZTF_get_lightcurve(coords_list, labels_list, nworkers=6, ztf_radius=0.000278 * u.deg):
    """Function to add the ZTF lightcurves in all three bands to a multiframe data structure

    Parameters
    ----------
    coords_list : list of astropy skycoords
        the coordinates of the targets for which a user wants light curves
    labels_list: list of strings
        journal articles associated with the target coordinates
    nworkers : int or None
        number of workers in the multiprocessing pool used in the load_lightcurves function. 
        This must be None if this function is being called from within a child process already. 
        (This function does not support nested multiprocessing.)
    ztf_radius : astropy Quantity
        search radius, how far from the source should the archives return results

    Returns
    -------
    df_lc : MultiIndexDFObject
        the main data structure to store all light curves
    """
    # the catalog is in parquet format with one file per ZTF filter, field, ccd, and quadrant
    # use a TAP query to locate which files each object is in
    locations_df = locate_objects(coords_list, labels_list, ztf_radius)

    # if none of the objects were found, there's nothing to load and the load_lightcurves fnc will raise a ValueError
    # just return an empty dataframe instead of proceeding
    if len(locations_df.index) == 0:
        return MultiIndexDFObject()

    # the catalog is stored in an AWS S3 bucket. loop over the files and load the light curves
    ztf_df = load_lightcurves(locations_df, nworkers=nworkers)

    # clean and transform the data into the form needed for a MultiIndexDFObject
    ztf_df = transform_lightcurves(ztf_df)

    # return the light curves as a MultiIndexDFObject
    indexes, columns = ["objectid", "label", "band", "time"], ["flux", "err"]
    return MultiIndexDFObject(data=ztf_df.set_index(indexes)[columns])


def file_name(filtercode, field, ccdid, qid, basedir=None):
    """Lookup the filename for this filtercode, field, ccdid, qid.

    File name syntax starts with: {basedir}/field{field}/ztf_{field}_{filtercode}_c{ccdid}_q{qid}

    Parameters
    ----------
    filtercode : str
        ZTF band name
    field : int
        ZTF field
    ccdid : int
        ZTF CCD
    qid : int
        ZTF quadrant
    basedir : int, optional
        Either 0 or 1. The base directory this file is located in.

    Returns
    -------
    file_name : str
        Parquet file name containing this filtercode, field, ccdid, and qid.

    Raises
    ------
    AssertionError
        if exactly one matching file name is not found in the CATALOG_FILES list
    """
    # if this comes from a TAP query we won't know the basedir,
    # so do a regex search through the CATALOG_FILES list instead
    if basedir is None:
        fre = re.compile(f"[01]/field{field:06}/ztf_{field:06}_{filtercode}_c{ccdid:02}_q{qid}")
        files = [CATALOG_ROOT + f for f in filter(fre.match, CATALOG_FILES)]
        # expecting exactly 1 filename. make it fail if there's more or less.
        assert len(files) == 1, f"found {len(files)} files. expected 1."
        return files[0]

    f = f"{basedir}/field{field:06}/ztf_{field:06}_{filtercode}_c{ccdid:02}_q{qid}_{DATARELEASE}.parquet"
    return CATALOG_ROOT + f


def locate_objects(coords_list, labels_list, radius):
    """The catalog's parquet files are organized by filter, field, CCD, and quadrant. Use TAP to look them up.

    https://irsa.ipac.caltech.edu/docs/program_interface/TAP.html

    Parameters
    ----------
    coords_list : list of tuples
        one tuple per target: (objectid, SkyCoord)
    labels_list: list of strings
        journal articles associated with target coordinates, indexed by objectid
    radius : astropy Quantity
        search radius, how far from the source should the archives return results

    Returns
    -------
    locations_df : pd.DataFrame
        Dataframe with ZTF field, CCD, quadrant and other information that identifies each `coords_list` 
        object and which parquet files it is in. One row per ZTF objectid.
    """
    # setup for tap query
    tap_service = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
    coords_tbl = make_coordsTable(coords_list, labels_list)
    coordscols = [f"coords.{c}" for c in ["objectid", "label"]]
    ztfcols = [f"ztf.{c}" for c in ["oid", "filtercode", "field", "ccdid", "qid", "ra", "dec"]]
    query = f"""SELECT {', '.join(coordscols + ztfcols)} 
        FROM ztf_objects_{DATARELEASE} ztf, TAP_UPLOAD.coords coords 
        WHERE CONTAINS(
            POINT('ICRS', coords.ra, coords.dec), CIRCLE('ICRS', ztf.ra, ztf.dec, {radius.value})
        )=1"""

    # this tap query is much faster when submitting less than ~10,000 coords at a time
    # so iterate over chunks of coords_tbl and then concat results
    chunksize = 10_000
    # calculate the number of iterations needed. it would be easier to just use this while loop directly
    # for the tap calls, but we want to use tqdm and that's easier in a for loop
    niterations = 0
    while niterations * chunksize < len(coords_tbl):
        niterations += 1
    # do the tap calls
    locations = []
    for i in tqdm(range(niterations)):
        result = tap_service.run_async(query, uploads={"coords": coords_tbl[i * chunksize : (i + 1) * chunksize]})
        locations.append(result.to_table().to_pandas())

    # locations may contain more than one ZTF object id per band (e.g., yang sample coords_list[10])
    # Sánchez-Sáez et al., 2021 (2021AJ....162..206S)
    # return all the data -- transform_lightcurves will choose which to keep
    return pd.concat(locations, ignore_index=True)


def load_lightcurves(locations_df, nworkers=6):
    """Loop over the catalog's parquet files (stored in an AWS S3 bucket) and load light curves.

    Parameters
    ----------
    locations_df : pd.DataFrame
        Dataframe with ZTF field, CCD, quadrant and other information that identifies each `coord` object
        and which parquet files it is in.
    nworkers : int or None
        Number of workers in the multiprocessing pool. Use None to turn off multiprocessing. This must be None 
        if this function is being called from within a child process (no nested multiprocessing).

    Returns
    -------
    ztf_df : pd.DataFrame
        Dataframe of light curves. Expect one row per oid in locations_df. Each row
        stores a full light curve. Elements in the columns "mag", "hmjd", etc. are arrays.
    """
    # one group per parquet file
    location_grps = locations_df.groupby(["filtercode", "field", "ccdid", "qid"])

    # if no multiprocessing requested, loop over files serially and load data. return immediately
    if nworkers is None:
        lightcurves = []
        for location in tqdm(location_grps):
            lightcurves.append(load_lightcurves_one_file(location))
        return pd.concat(lightcurves, ignore_index=True)
    
    # if we get here, multiprocessing has been requested

    # number of files to be sent in as one "chunk" of work
    chunksize = 100
    # make sure the chunksize isn't so big that some workers will sit idle
    if len(location_grps) < nworkers * chunksize:
        chunksize = len(location_grps) // nworkers + 1

    # "spawn" new processes because it uses less memory and is thread safe (req'd for pd.read_parquet)
    # https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn
    mp.set_start_method("spawn", force=True)
    
    # start a pool of background processes to load data in parallel
    with mp.Pool(nworkers) as pool:
        lightcurves = []
        # use imap because it's lazier than map and can reduce memory usage for long iterables
        # use unordered because we don't care about the order in which results are returned
        # using a large chunksize can make it much faster than the default of 1
        for ztf_df in tqdm(
            pool.imap_unordered(load_lightcurves_one_file, location_grps, chunksize=chunksize),
            total=len(location_grps)  # must tell tqdm how many files we're iterating over
        ):
            lightcurves.append(ztf_df)

    return pd.concat(lightcurves, ignore_index=True)


def load_lightcurves_one_file(location):
    """Load light curves from one file.

    Parameters
    ----------
    location : tuple
        tuple containing two elements that describe one parquet file. the elements are:
            location_keys : tuple(str, int, int int)
                Keys that uniquely identify a ZTF location (filtercode, field, CCD, and quadrant).
                Used to lookup the file name.
            location_df : pd.DataFrame
                Dataframe of objects in this location. Used to filter data from this file.

    Returns
    -------
    ztf_df : pd.DataFrame
        Dataframe of light curves. Expect one row per oid in location_df. Each row
        stores a full light curve. Elements in the columns "mag", "hmjd", etc. are arrays.
    """
    location_keys, location_df = location

    # load light curves from this file, filtering for the ZTF object IDs in location_df
    ztf_df = pd.read_parquet(
        file_name(*location_keys),
        engine="pyarrow",
        filesystem=s3fs.S3FileSystem(),
        columns=["objectid", "hmjd", "mag", "magerr", "catflags"],
        filters=[("objectid", "in", location_df["oid"].to_list())],
    )

    # in the parquet files, "objectid" is the ZTF object ID
    # in the MultiIndexDFObject, "objectid" is the ID of a coords_list object
    # rename the ZTF object ID that just got loaded to avoid confusion
    ztf_df = ztf_df.rename(columns={"objectid": "oid"})

    # add the coords_list object ID and label columns by mapping thru the ZTF object ID
    oidmap = location_df.set_index("oid")["objectid"].to_dict()
    lblmap = location_df.set_index("oid")["label"].to_dict()
    ztf_df["objectid"] = ztf_df["oid"].map(oidmap)
    ztf_df["label"] = ztf_df["oid"].map(lblmap)
    
    # add the band (i.e., filtercode)
    ztf_df["band"] = location_keys[0]

    return ztf_df


def transform_lightcurves(ztf_df):
    """Clean and transform the data into the form expected for a `MultiIndexDFObject`.

    Parameters
    ----------
    ztf_df : pd.DataFrame
        Dataframe of light curves as returned by `load_lightcurves`.

    Returns
    -------
    ztf_df : pd.DataFrame
        The input dataframe, cleaned and transformed.
    """
    # ztf_df might have more than one light curve per (band + coords_list id) if the ra/dec is close
    # to a CCD-quadrant boundary (Sanchez-Saez et al., 2021). keep the one with the most datapoints
    ztf_df_list = []
    for _, singleband_object in ztf_df.groupby(["objectid", "band"]):
        if len(singleband_object.index) == 1:
            ztf_df_list.append(singleband_object)
        else:
            npoints = singleband_object["mag"].str.len()
            npointsmax_object = singleband_object.loc[npoints == npoints.max()]
            # this may still have more than one light curve if they happen to have the same number
            # of datapoints (e.g., Yang sample coords_list[6], band 'zr').
            # arbitrarily pick the one with the min oid.
            # depending on your science, you may want (e.g.,) the largest timespan instead
            if len(npointsmax_object.index) == 1:
                ztf_df_list.append(npointsmax_object)
            else:
                minoid = npointsmax_object.oid.min()
                ztf_df_list.append(npointsmax_object.loc[npointsmax_object.oid == minoid])

    ztf_df = pd.concat(ztf_df_list, ignore_index=True)

    # store "hmjd" as "time".
    # note that other light curves in this notebook will have "time" as MJD instead of HMJD.
    # if your science depends on precise times, this will need to be corrected.
    ztf_df = ztf_df.rename(columns={"hmjd": "time"})

    # "explode" the data structure into one row per light curve point
    ztf_df = ztf_df.explode(["time", "mag", "magerr", "catflags"], ignore_index=True)

    # remove data flagged as bad
    ztf_df = ztf_df.loc[ztf_df["catflags"] < 32768, :]

    # calc flux [https://arxiv.org/pdf/1902.01872.pdf zeropoint corrections already applied]
    magupper = ztf_df["mag"] + ztf_df["magerr"]
    maglower = ztf_df["mag"] - ztf_df["magerr"]
    flux = 10 ** ((ztf_df["mag"] - 23.9) / -2.5)  # uJy
    flux_upper = abs(flux - 10 ** ((magupper - 23.9) / -2.5))
    flux_lower = abs(flux - 10 ** ((maglower - 23.9) / -2.5))
    fluxerr = (flux_upper + flux_lower) / 2.0
    ztf_df.loc[:, "flux"] = flux * 1e-3  # now in mJy
    ztf_df.loc[:, "err"] = fluxerr * 1e-3

    return ztf_df
