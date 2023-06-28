import re

import astropy.units as u
import pandas as pd
import pyvo
from data_structures import MultiIndexDFObject


DATARELEASE = "dr17"
URLBASE = f"https://irsa.ipac.caltech.edu/data/ZTF/lc/lc_{DATARELEASE}"
# get a list of files in the dataset using the checksums file
DSFILES = (
    pd.read_table(f"{URLBASE}/checksum.md5", sep="\s+", names=["md5", "path"], usecols=["path"])
    .squeeze()
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
    df_lc = MultiIndexDFObject()

    # get a df of the info needed to locate each object in the parquet files
    service = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
    locations = pd.concat(
        [locate_object(coord, labels_list, ztf_radius, service) for coord in coords_list],
        ignore_index=True,
    )

    # parquet files are organized by filter, field, ccd, and quadrant
    # group the locations df by file, loop thru files and load, filtering for objectid
    for key, df in locations.groupby(["filtercode", "field", "ccdid", "qid"]):
        ztflc_df = pd.read_parquet(
            file_name(*key),
            engine="pyarrow",
            columns=["objectid", "filterid", "nepochs", "hmjd", "mag", "magerr", "catflags"],
            filters=[("objectid", "in", df["objectid"].to_list())],
        )

        # add oid and label columns by mapping from object id
        oidmap = df.set_index("objectid")["oid"].to_dict()
        lblmap = df.set_index("objectid")["label"].to_dict()
        ztflc_df["oid"] = ztflc_df["objectid"].map(oidmap)
        ztflc_df["label"] = ztflc_df["objectid"].map(lblmap)

        # add the band (filtercode) as a column
        ztflc_df["band"] = key[0]

        # would the "repeated lightcurve" problem be solved by returning only the single closest
        # object from locate_object()?
        # if not, choose the one with the largest nepochs?

        # need to convert hmjd -> mjd and label it "time". this isn't right but leaving it for now
        ztflc_df = ztflc_df.rename(columns={"hmjd": "time"})

        ztflc_df = ztflc_df.explode(["time", "mag", "magerr", "catflags"], ignore_index=True)

        # remove data flagged as bad (catflag = 32768)
        # guessing we want to remove single datapoints, not full light curves just because one point is bad?
        ztflc_df = ztflc_df.query("catflags != 32768")

        # calc flux [https://arxiv.org/pdf/1902.01872.pdf zeropoint corrections already applied]
        magupper = ztflc_df["mag"] + ztflc_df["magerr"]
        maglower = ztflc_df["mag"] - ztflc_df["magerr"]
        flux = 10 ** ((ztflc_df["mag"] - 23.9) / -2.5)  # uJy
        flux_upper = abs(flux - 10 ** ((magupper - 23.9) / -2.5))
        flux_lower = abs(flux - 10 ** ((maglower - 23.9) / -2.5))
        fluxerr = (flux_upper + flux_lower) / 2.0
        ztflc_df.loc[:, "flux"] = flux * 1e-3  # now in mJy
        ztflc_df.loc[:, "err"] = fluxerr * 1e-3

        df_lc.append(
            ztflc_df[["flux", "err", "time", "oid", "band", "label"]].set_index(
                ["oid", "label", "band", "time"]
            )
        )

    return df_lc


def locate_object(coord, labels_list, ztf_radius, tap_service) -> pd.DataFrame:
    """Lookup the ZTF field, ccd, and quadrant this coord is located in.

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
        Dataframe with the object's ZTF field, CCD, quadrant and other information useful for
        locating the object in the parquet files and associating it with this coord.
    """
    coord_id, ra, dec = coord[0], coord[1].ra.deg, coord[1].dec.deg
    # files are organized by filter, field, ccd, and quadrant
    # look them up using tap. https://irsa.ipac.caltech.edu/docs/program_interface/TAP.html
    result = tap_service.run_sync(
        f"SELECT {', '.join(['oid', 'filtercode', 'field', 'ccdid', 'qid', 'ra', 'dec'])} "
        f"FROM ztf_objects_{DATARELEASE} "
        "WHERE CONTAINS("
        # must be one of 'J2000', 'ICRS', and 'GALACTIC'. guessing icrs, but need to check
        f"POINT('ICRS',ra, dec), CIRCLE('ICRS',{ra},{dec},{ztf_radius.value})"
        ")=1"
    )
    field_df = result.to_table().to_pandas()

    # in the MultiIndexDFObject, "oid" is the name of the coord ID, not the ztf object id
    field_df = field_df.rename(columns={"oid": "objectid"})
    field_df["oid"] = coord_id
    field_df["label"] = labels_list[coord_id]

    # field_df may have more than one oid per band (e.g., yang sample coords_list[10])
    # guessing we'll want to use the one with the closest ra/dec?
    # skipping this for now

    return field_df


def file_name(filtercode, field, ccdid, qid):
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

    Returns
    -------
    file_name : str
        Parquet file name containing this filtercode, field, ccdid, and qid.

    Raises
    ------
    AssertionError
        if exactly one matching file name is not found in the DSFILES list
    """
    # can't quite construct the file name directly because need to know whether the top-level
    # directory is 0 or 1. do a regex search through the DSFILES list.
    fre = re.compile(f"[01]/field{field:06}/ztf_{field:06}_{filtercode}_c{ccdid:02}_q{qid}")
    files = [f"{URLBASE}/{f}" for f in filter(fre.match, DSFILES)]
    # expecting exactly 1 filename. make it fail if there's more or less.
    assert len(files) == 1, f"found {len(files)} files. expected 1."
    return files[0]
