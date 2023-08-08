import re

import astropy.units as u
import hpgeom
import pandas as pd
import pyarrow
import pyarrow.compute
import pyarrow.dataset
import pyvo
import s3fs
from astropy.coordinates import SkyCoord

from data_structures import MultiIndexDFObject

DATARELEASE = "dr18"
BUCKET = "irsa-mast-tike-spitzer-data"
CATALOG_ROOT = f"{BUCKET}/data/ZTF/lc/lc_{DATARELEASE}/"
FS = s3fs.S3FileSystem()

# get a list of files in the dataset using the checksums file
CATALOG_FILES = (
    pd.read_table(
        f"s3://{CATALOG_ROOT}checksum.md5", sep="\s+", names=["md5", "path"], usecols=["path"]
    )
    .squeeze()
    .str.removeprefix("./")
    .to_list()
)

# load the "helper" dataset used to locate objects in the CATALOG_FILES
LOCATIONS_DS = False  # this dataset doesn't exist yet. load locations from a TAP query instead
# LOCATIONS_DS = pyarrow.dataset.parquet_dataset(
#         CATALOG_ROOT + "object-locations.parquet/_metadata", filesystem=FS, partitioning="hive"
#     )


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
    # the parquet files are organized by filter, field, ccd, and quadrant
    # get a df of the info needed to locate each object in the parquet files
    if LOCATIONS_DS:
        locations = locate_objects(coords_list, labels_list, ztf_radius)
        location_groups = locations.groupby(["filtercode", "field", "ccdid", "qid", "basedir"])
    else:
        service = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
        locations = [
            locate_object(coord, labels_list, ztf_radius, service) for coord in coords_list
        ]
        locations = pd.concat(locations, ignore_index=True)
        location_groups = locations.groupby(["filtercode", "field", "ccdid", "qid"])

    # load light curves by looping over files and filtering for ZTF objectid
    lc_df = [load_lightcurves(keys, locdf) for keys, locdf in location_groups]
    lc_df = pd.concat(lc_df, ignore_index=True)

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

    # finish transforming the data into the form expected by a MultiIndexDFObject
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

    return MultiIndexDFObject(
        lc_df[["flux", "err", "objectid", "label", "band", "time"]].set_index(
            ["objectid", "label", "band", "time"]
        )
    )


def locate_object(coord, labels_list, ztf_radius, tap_service) -> pd.DataFrame:
    """Lookup the ZTF field, ccd, and quadrant this coord object is located in.

    Parameters
    ----------
    coord : tuple
        the object's ID and astropy skycoords
    labels_list: list of strings
        journal articles associated with target coordinates, indexed by object ID
    ztf_radius : float
        search radius, how far from the source should the archives return results
    tap_service : pyvo.dal.TAPService
        IRSA TAP service

    Returns
    -------
    field_df : pd.DataFrame
        Dataframe with ZTF field, CCD, quadrant and other information useful for
        locating the `coord` object in the parquet files. One row per ZTF objectid.
    """
    coord_id, ra, dec = coord[0], coord[1].ra.deg, coord[1].dec.deg
    # files are organized by filter, field, ccd, and quadrant
    # look them up using tap. https://irsa.ipac.caltech.edu/docs/program_interface/TAP.html
    result = tap_service.run_sync(
        f"SELECT {', '.join(['oid', 'filtercode', 'field', 'ccdid', 'qid', 'ra', 'dec'])} "
        f"FROM ztf_objects_{DATARELEASE} "
        "WHERE CONTAINS("
        f"POINT('ICRS',ra, dec), CIRCLE('ICRS',{ra},{dec},{ztf_radius.value})"
        ")=1"
    )
    field_df = result.to_table().to_pandas()

    # add fields for the MultiIndexDFObject
    field_df["objectid"] = coord_id
    field_df["label"] = labels_list[coord_id]

    # field_df may have more than one ZTF object id per band (e.g., yang sample coords_list[10])
    # following Sánchez-Sáez et al., 2021 (2021AJ....162..206S)
    # we'll load all the data and then keep the longest light curve (in the main function)



def locate_objects(coords_list, labels_list, ztf_radius):
    """Use LOCATIONS_DS to lookup the ZTF field, ccd, and quadrant this coord object is located in.

    Parameters
    ----------
    coord : tuple
        the object's ID and astropy skycoords
    labels_list: list of strings
        journal articles associated with target coordinates, indexed by object ID
    ztf_radius : float
        search radius, how far from the source should the archives return results

    Returns
    -------
    locations : pd.DataFrame
        Dataframe with ZTF field, CCD, quadrant and other information useful for
        locating the `coord` object in the parquet files. One row per ZTF objectid.
    """
    my_coords_list = []
    for objectid, coord in coords_list:
        cone_pixels = hpgeom.query_circle(
            a=coord.ra.deg,
            b=coord.dec.deg,
            radius=ztf_radius.value,
            nside=hpgeom.order_to_nside(5),  # catalog is partitioned by HEALPix order 5
            nest=True,  # catalog uses nested ordering scheme for pixel index
            inclusive=True,  # return all pixels that overlap with the circle, and maybe a few more
        )
        my_coords_list.append((objectid, coord, labels_list[objectid], cone_pixels))
    locations = pd.DataFrame(my_coords_list, columns=["objectid", "coord", "label", "pixel"])
    locations = locations.explode(["pixel"], ignore_index=True)

    locations_list = []
    for pixel, locs_df in locations.groupby("pixel"):
        region_tbl = LOCATIONS_DS.to_table(filter=(pyarrow.compute.field("healpix_k5") == pixel))
        region_skycoords = SkyCoord(ra=region_tbl["ra"] * u.deg, dec=region_tbl["dec"] * u.deg)
        objects_skycoords = SkyCoord(locs_df.coord.to_list())

        region_ilocs, object_ilocs, _, _ = objects_skycoords.search_around_sky(
            region_skycoords, ztf_radius
        )
        match_df = region_tbl.take(region_ilocs).to_pandas()
        # in the MultiIndexDFObject, "objectid" is the name of the coord id, not the ztf object id
        match_df = match_df.rename(columns={"objectid": "oid"})

        match_df["object_ilocs"] = object_ilocs
        locations_list.append(
            match_df.set_index("object_ilocs").join(locs_df.reset_index(drop=True))
        )
    locations = pd.concat(locations_list, ignore_index=True)

    # locations may have more than one ZTF object id per band (e.g., yang sample coords_list[10])
    # following Sánchez-Sáez et al., 2021 (2021AJ....162..206S)
    # we'll load all the data and then keep the longest light curve in the main function
    columns = ["oid", "filtercode", "field", "ccdid", "qid", "ra", "dec", "objectid", "label"]
    return locations[columns]


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


def load_lightcurves(location_keys, location_df):
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
    lc_df = pd.read_parquet(
        file_name(*location_keys),
        engine="pyarrow",
        filesystem=FS,
        columns=["objectid", "hmjd", "mag", "magerr", "catflags"],
        filters=[("objectid", "in", location_df["oid"].to_list())],
    )

    # in the MultiIndexDFObject, "objectid" is the name of the coord id, not the ztf object id
    lc_df = lc_df.rename(columns={"objectid": "oid"})

    # add fields for the MultiIndexDFObject
    # add the band (filtercode)
    lc_df["band"] = location_keys[0]
    # add objectid (coords_list id) and label columns by mapping from ZTF object id
    oidmap = location_df.set_index("oid")["objectid"].to_dict()
    lblmap = location_df.set_index("oid")["label"].to_dict()
    lc_df["objectid"] = lc_df["oid"].map(oidmap)
    lc_df["label"] = lc_df["oid"].map(lblmap)

    return lc_df


