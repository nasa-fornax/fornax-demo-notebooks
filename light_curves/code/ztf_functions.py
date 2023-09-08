import multiprocessing as mp
import re

import astropy.units as u
import pandas as pd
import pyvo
import s3fs

from data_structures import MultiIndexDFObject
from sample_selection import make_coordsTable

DATARELEASE = "dr18"
BUCKET = "irsa-mast-tike-spitzer-data"
CATALOG_ROOT = f"{BUCKET}/data/ZTF/lc/lc_{DATARELEASE}/"

# get a list of files in the dataset using the checksums file
CATALOG_FILES = (
    pd.read_table(
        f"s3://{CATALOG_ROOT}checksum.md5", sep="\s+", names=["md5", "path"], usecols=["path"]
    )
    .squeeze()  # there's only 1 column. squeeze it into a Series
    .str.removeprefix("./")
    .to_list()
)


def ZTF_get_lightcurve(coords_list, labels_list, ztf_radius=0.000278 * u.deg):
    """Function to add the ZTF lightcurves in all three bands to a multiframe data structure

    Parameters
    ----------
    coords_list : list of astropy skycoords
        the coordinates of the targets for which a user wants light curves
    labels_list: list of strings
        journal articles associated with the target coordinates
    ztf_radius : float
        search radius, how far from the source should the archives return results

    Returns
    -------
    df_lc : MultiIndexDFObject
        the main data structure to store all light curves
    """
    # light curves are stored in parquet files that are organized by filter, field, ccd, and quadrant
    # locate which files each object is in
    locations_df = locate_objects(coords_list, labels_list, ztf_radius)

    # load light curves by looping over files and filtering for ZTF objectid
    lc_df = load_lightcurves(locations_df)

    # clean and transform the data into the form expected for a MultiIndexDFObject
    lc_df = transform_lightcurves(lc_df)

    return MultiIndexDFObject(
        lc_df[["flux", "err", "objectid", "label", "band", "time"]].set_index(
            ["objectid", "label", "band", "time"]
        )
    )


def file_name(filtercode, field, ccdid, qid, basedir=None):
    """Lookup the filename for this filtercode, field, ccdid, qid.

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
    # if this comes from a TAP query, we won't know the basedir
    if basedir is None:
        # can't quite construct the file name directly because need to know whether the top-level
        # directory is 0 or 1. do a regex search through the CATALOG_FILES list.
        fre = re.compile(f"[01]/field{field:06}/ztf_{field:06}_{filtercode}_c{ccdid:02}_q{qid}")
        files = [CATALOG_ROOT + f for f in filter(fre.match, CATALOG_FILES)]
        # expecting exactly 1 filename. make it fail if there's more or less.
        assert len(files) == 1, f"found {len(files)} files. expected 1."
        return files[0]

    f = f"{basedir}/field{field:06}/ztf_{field:06}_{filtercode}_c{ccdid:02}_q{qid}_{DATARELEASE}.parquet"
    return CATALOG_ROOT + f


def locate_objects(coords_list, labels_list, radius):
    """Parquet files are organized by filter, field, ccd, and quadrant. Use TAP to look them up.

    https://irsa.ipac.caltech.edu/docs/program_interface/TAP.html

    Parameters
    ----------
    coords_list : list of tuples
        one tuple per target: (objectid, SkyCoord)
    labels_list: list of strings
        journal articles associated with target coordinates, indexed by object ID
    radius : float
        search radius, how far from the source should the archives return results

    Returns
    -------
    locations_df : pd.DataFrame
        Dataframe with ZTF field, CCD, quadrant and other information useful for
        locating the `coord` object in the parquet files. One row per ZTF objectid.
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

    # tap query is much faster when submitting less than ~10,000 coords at a time
    # so iterate over chunks of coords_tbl and then concat results
    i, chunksize = 0, 10_000
    locations = []
    while i * chunksize < len(coords_tbl):
        result = tap_service.run_async(
            query, uploads={"coords": coords_tbl[i * chunksize : (i + 1) * chunksize]}
        )
        locations.append(result.to_table().to_pandas())
        i += 1

    # results may contain more than one ZTF object id per band (e.g., yang sample coords_list[10])
    # Sánchez-Sáez et al., 2021 (2021AJ....162..206S)
    # return all the data -- transform_lightcurves will choose which to keep
    return pd.concat(locations, ignore_index=True)


def load_lightcurves(locations_df, npool=6):
    """"""
    # one group per parquet file
    location_grps = locations_df.groupby(["filtercode", "field", "ccdid", "qid"])
    # number of files to send to a background process as one "chunk" of work
    chunksize = 100

    # load the light curve data
    # if not doing at least 2 chunks it's not worth the multiprocessing overhead. just do them serially
    # else, use multiprocessing and do chunks in parallel
    if len(location_grps) < 2 * chunksize:
        lightcurves = [load_lightcurves_one_file(group) for group in location_grps]

    else:
        # "spawn" new processes because it uses less memory and is thread safe
        # https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn
        mp.set_start_method("spawn", force=True)
        with mp.Pool(npool) as pool:
            lightcurves = []
            # use imap because it's lazier than map and can reduce memory usage for long iterables
            # (using a large chunksize can make it much faster)
            # use unordered because we don't care about the order in which results are returned
            for lc_df in pool.imap_unordered(
                load_lightcurves_one_file, location_grps, chunksize=chunksize
            ):
                lightcurves.append(lc_df)

    return pd.concat(lightcurves, ignore_index=True)


def load_lightcurves_one_file(locations):
    """Load light curves from the file identified by `location_keys`, for objects in location_df.

    Parameters
    ----------
    location_keys : tuple(str, int, int int)
        Keys that uniquely identify a ZTF location (filtercode, field, CCD, and quadrant).
        Used to lookup the file name.
    location_df : pd.DataFrame
        Dataframe of objects in this location. Used to filter data from this file.

    Returns
    -------
    lc_df : pd.DataFrame
        Dataframe of light curves. Expect one row per oid in location_df. Each row
        stores a full light curve. Elements in the columns "mag", "hmjd", etc. are arrays.
    """
    location_keys, location_df = locations

    lc_df = pd.read_parquet(
        file_name(*location_keys),
        engine="pyarrow",
        filesystem=s3fs.S3FileSystem(),
        columns=["objectid", "hmjd", "mag", "magerr", "catflags"],
        filters=[("objectid", "in", location_df["oid"].to_list())],
    )

    # in the MultiIndexDFObject, "objectid" is the name of the coord id, not the ztf object id
    # rename the ztf object id to avoid confusion
    lc_df = lc_df.rename(columns={"objectid": "oid"})

    # add objectid (coords_list id) and label columns by mapping from ZTF object id
    oidmap = location_df.set_index("oid")["objectid"].to_dict()
    lblmap = location_df.set_index("oid")["label"].to_dict()
    lc_df["objectid"] = lc_df["oid"].map(oidmap)
    lc_df["label"] = lc_df["oid"].map(lblmap)
    # add the band (filtercode)
    lc_df["band"] = location_keys[0]

    return lc_df


def transform_lightcurves(lc_df):
    """Clean and transform the data into the form expected for a `MultiIndexDFObject`.

    Parameters
    ----------
    lc_df : pd.DataFrame
        Dataframe of light curves as returned by `load_lightcurves`.

    Returns
    -------
    lc_df : pd.DataFrame
        The input dataframe, cleaned and transformed.
    """
    # lc_df might have more than one light curve per (band + coords_list id) if the ra/dec is close
    # to a CCD-quadrant boundary (Sanchez-Saez et al., 2021). keep the one with the most datapoints
    lc_df_list = []
    for _, singleband_object in lc_df.groupby(["objectid", "band"]):
        if len(singleband_object.index) == 1:
            lc_df_list.append(singleband_object)
        else:
            npoints = singleband_object["mag"].str.len()
            npointsmax_object = singleband_object.loc[npoints == npoints.max()]
            # this may still have more than one light curve if they happen to have the same number
            # of datapoints (e.g., Yang sample coords_list[6], band 'zr').
            # arbitrarily pick the one with the min oid.
            # depending on your science, you may want (e.g.,) the largest timespan instead
            if len(npointsmax_object.index) == 1:
                lc_df_list.append(npointsmax_object)
            else:
                minoid = npointsmax_object.oid.min()
                lc_df_list.append(npointsmax_object.loc[npointsmax_object.oid == minoid])

    lc_df = pd.concat(lc_df_list, ignore_index=True)

    # store "hmjd" as "time".
    # note that other light curves in this notebook will have "time" as MJD instead of HMJD.
    # if your science depends on precise times, this will need to be corrected.
    lc_df = lc_df.rename(columns={"hmjd": "time"})

    # explode the data structure into one row per light curve point
    lc_df = lc_df.explode(["time", "mag", "magerr", "catflags"], ignore_index=True)

    # remove data flagged as bad
    lc_df = lc_df.loc[lc_df["catflags"] < 32768, :]

    # calc flux [https://arxiv.org/pdf/1902.01872.pdf zeropoint corrections already applied]
    magupper = lc_df["mag"] + lc_df["magerr"]
    maglower = lc_df["mag"] - lc_df["magerr"]
    flux = 10 ** ((lc_df["mag"] - 23.9) / -2.5)  # uJy
    flux_upper = abs(flux - 10 ** ((magupper - 23.9) / -2.5))
    flux_lower = abs(flux - 10 ** ((maglower - 23.9) / -2.5))
    fluxerr = (flux_upper + flux_lower) / 2.0
    lc_df.loc[:, "flux"] = flux * 1e-3  # now in mJy
    lc_df.loc[:, "err"] = fluxerr * 1e-3

    return lc_df
