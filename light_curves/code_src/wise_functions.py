import astropy.units as u
import hpgeom
import pandas as pd
import pyarrow.compute
import pyarrow.dataset
import pyarrow.fs
from astropy.coordinates import SkyCoord
from tqdm.auto import tqdm

from data_structures import MultiIndexDFObject
from fluxconversions import convert_wise_flux_to_millijansky

BANDMAP = {"W1": 1, "W2": 2}  # map the common names to the values actually stored in the catalog
K = 5  # HEALPix order at which the dataset is partitioned


def wise_get_lightcurves(sample_table, *, radius=1.0, bandlist=["W1", "W2"]):
    """Loads WISE data by searching the unWISE light curve catalog (Meisner et al., 2023AJ....165...36M).
    This is the MAIN function

    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    radius : float
        radius (arcsec) for the cone search to determine whether a particular detection is associated with an object
    bandlist: list of strings
        which WISE bands to search for, example: ['W1', 'W2']

    Returns
    -------
    df_lc : MultiIndexDFObject
        the main data structure to store all light curves
    """
    radius = radius * u.arcsec

    # the catalog is in parquet format, partitioned by HEALPix pixel index at order k=5
    # locate the pixels/partitions each object is in
    locations_df = locate_objects(sample_table, radius)

    # the catalog is stored in an AWS S3 bucket
    # loop over the partitions and load the light curves
    wise_df = load_lightcurves(locations_df, radius, bandlist)

    # clean and transform the data into the form needed for a MultiIndexDFObject
    wise_df = transform_lightcurves(wise_df)

    # return the light curves as a MultiIndexDFObject
    indexes, columns = ["objectid", "label", "band", "time"], ["flux", "err"]
    return MultiIndexDFObject(data=wise_df.set_index(indexes)[columns].sort_index())


def locate_objects(sample_table, radius):
    """Locate the partitions (HEALPix order 5 pixels) each sample_table object is in.

    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    radius : astropy Quantity
        radius for the cone search to determine which pixels overlap with each object

    Returns
    -------
    locations : pd.DataFrame
        dataframe of location info including ["objectid", "coord.ra", "coord.dec", "label", "pixel"]
    """
    # loop over objects and determine which HEALPix pixels overlap with a circle of r=`radius` around it
    # this determines which catalog partitions contain each object
    healpix_pixel = [hpgeom.query_circle(a=row['coord'].ra.deg, b=row['coord'].dec.deg,
                                         radius=radius.to(u.deg).value,
                                         nside=hpgeom.order_to_nside(K), nest=True, inclusive=True)
                     for row in sample_table]

    locations = sample_table.to_pandas()
    locations['pixel'] = healpix_pixel

    # locations contains one row per object, and the pixel column stores arrays of ints
    # "explode" the dataframe into one row per object per pixel. may create multiple rows per object
    return locations.explode(["pixel"], ignore_index=True)


def load_lightcurves(locations, radius, bandlist):
    """Load data from the unWISE light curve catalog (Meisner et al., 2023AJ....165...36M).

    Parameters
    ----------
    locations : pd.DataFrame
        dataframe of location info (coords, pixel index, etc.) for the objects to be loaded
    radius : astropy Quantity
        radius for the cone search to determine whether a particular detection is associated with an object
    bandlist: list of strings
        which WISE bands to search for, example: ['W1', 'W2']

    Returns
    -------
    wise_df : pd.DataFrame
        dataframe of light curve data for all objects in `locations`
    """
    # load the catalog's metadata as a pyarrow dataset. this will be used to query the catalog.
    # the catalog is stored in an AWS S3 bucket
    fs = pyarrow.fs.S3FileSystem(region="us-west-2", anonymous=True)
    bucket = "nasa-irsa-wise"
    catalog_root = f"{bucket}/unwise/neo7/catalogs/time_domain/healpix_k{K}/unwise-neo7-time_domain-healpix_k{K}.parquet"
    dataset = pyarrow.dataset.parquet_dataset(
        f"{catalog_root}/_metadata", filesystem=fs, partitioning="hive")

    # specify which columns will be loaded
    # for a complete list of column names, use: `dataset.schema.names`
    # to load the complete schema, including units and descriptions, use:
    # schema = pyarrow.dataset.parquet_dataset(f"{catalog_root}/_common_metadata", filesystem=fs).schema
    columns = ["flux", "dflux", "ra", "dec", "band", "MJDMEAN"]

    # iterate over partitions, load data, and find each object
    wise_df_list = []
    for pixel, locs_df in tqdm(locations.groupby("pixel")):
        # create a filter to pick out sources that are (1) in this partition; and (2) within the
        # coadd's primary region (to avoid duplicates when an object is near the coadd boundary)
        filter = (pyarrow.compute.field(f"healpix_k{K}") == pixel) & (
            pyarrow.compute.field("primary") == 1)
        # add a filter for the bandlist. if all bands are requested, skip this to avoid the overhead
        if len(set(BANDMAP.keys()) - set(bandlist)) > 0:
            filter = filter & (pyarrow.compute.field("band").isin(
                [BANDMAP[band] for band in bandlist]))

        # query the dataset and load the light curve data
        pixel_tbl = dataset.to_table(filter=filter, columns=columns)

        # do a cone search using astropy to select sources belonging to each object
        pixel_skycoords = SkyCoord(ra=pixel_tbl["ra"] * u.deg, dec=pixel_tbl["dec"] * u.deg)
        objects_skycoords = SkyCoord(locs_df["coord.ra"], locs_df["coord.dec"], unit=u.deg)
        object_ilocs, pixel_ilocs, _, _ = pixel_skycoords.search_around_sky(
            objects_skycoords, radius)

        # create a dataframe with all matched sources
        match_df = pixel_tbl.take(pixel_ilocs).to_pandas()
        # attach identifying info like the objectid by joining with locs_df
        match_df["object_ilocs"] = object_ilocs
        match_df = match_df.set_index("object_ilocs").join(locs_df.reset_index(drop=True))

        wise_df_list.append(match_df)

    # concat light curves into a single dataframe and return
    return pd.concat(wise_df_list, ignore_index=True)


def transform_lightcurves(wise_df):
    """Clean and transform the data into the form needed for a `MultiIndexDFObject`.

    Parameters
    ----------
    wise_df : pd.DataFrame
        dataframe of light curves as returned by `load_lightcurves`.

    Returns
    -------
    wise_df : pd.DataFrame
        the input dataframe, cleaned and transformed.
    """
    # rename columns to match a MultiIndexDFObject
    wise_df = wise_df.rename(columns={"MJDMEAN": "time", "dflux": "err"})
    # convert the band to its common name ("W1" or "W2"). need to invert the BANDMAP dict.
    wise_df["band"] = wise_df["band"].map({value: key for key, value in BANDMAP.items()})
    # filter for only positive fluxes
    wise_df = wise_df[wise_df["flux"] > 0]
    # convert units, per band
    grouped_wise_df = wise_df.groupby("band")
    wise_df["flux"] = grouped_wise_df["flux"].transform(convert_wise_flux_to_millijansky)
    wise_df["err"] = grouped_wise_df["err"].transform(convert_wise_flux_to_millijansky)
    return wise_df
