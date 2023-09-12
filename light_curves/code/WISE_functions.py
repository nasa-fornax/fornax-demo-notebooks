import astropy.units as u
import hpgeom
import pandas as pd
import pyarrow.compute
import pyarrow.dataset
import pyarrow.fs
from astropy.coordinates import SkyCoord

from data_structures import MultiIndexDFObject
from fluxconversions import convert_wise_flux_to_millijansky


BANDMAP = {"W1": 1, "W2": 2}  # map the common names to the values actually stored in the catalog
K = 5  # HEALPix order at which the dataset is partitioned


def WISE_get_lightcurves(coords_list, labels_list, radius=1.0 * u.arcsec, bandlist=["W1", "W2"]):
    """Loads WISE data by searching the unWISE light curve catalog (Meisner et al., 2023AJ....165...36M).

    Parameters
    ----------
    coords_list : list of astropy skycoords
        the coordinates of the targets for which a user wants light curves
    labels_list: list of strings
        journal articles associated with the target coordinates
    radius : astropy Quantity
        radius for the cone search to determine whether a particular detection is associated with an object
    bandlist: list of strings
        which WISE bands to search for, example: ['W1', 'W2']

    Returns
    -------
    df_lc : MultiIndexDFObject
        the main data structure to store all light curves
    """
    # the catalog is in parquet format, partitioned by HEALPix pixel index at order k=5
    # locate the pixels/partitions each object is in
    locations_df = locate_objects(coords_list, labels_list, radius)

    # the catalog is stored in an AWS S3 bucket
    # loop over the partitions and load the light curves
    wise_df = load_data(locations_df, radius, bandlist)

    # clean and transform the data into the form needed for a MultiIndexDFObject
    
    wise_df = transform_lightcurves(wise_df)

    # return the light curves as a MultiIndexDFObject
    indexes, columns = ["objectid", "label", "band", "time"], ["flux", "err"]
    return MultiIndexDFObject(data=wise_df.set_index(indexes)[columns])


def locate_objects(coords_list, labels_list, radius):
    """Locate the partitions (HEALPix order 5 pixels) each coords_list object is in.

    Parameters
    ----------
    coords_list : list of astropy skycoords
        the coordinates of the targets for which a user wants light curves
    labels_list: list of strings
        journal articles associated with the target coordinates
    radius : astropy Quantity
        radius for the cone search to determine which pixels overlap with each object

    Returns
    -------
    locations : pd.DataFrame
        dataframe of location info including ["objectid", "coord", "label", "pixel"]
    """
    # loop over objects and determine which HEALPix pixels overlap with a circle of r=`radius` around it
    # this determines which catalog partitions contain each object
    my_coords_list = []
    for objectid, coord in coords_list:
        cone_pixels = hpgeom.query_circle(
            a=coord.ra.deg,
            b=coord.dec.deg,
            radius=radius.to(u.deg).value,
            nside=hpgeom.order_to_nside(K),
            nest=True,  # catalog uses nested ordering scheme for pixel index
            inclusive=True,  # return all pixels that overlap with the circle, and maybe a few more
        )
        my_coords_list.append((objectid, coord, labels_list[objectid], cone_pixels))

    # my_coords_list is a list of tuples. turn it into a dataframe
    locations = pd.DataFrame(my_coords_list, columns=["objectid", "coord", "label", "pixel"])

    # locations contains one row per object, and the pixel column stores arrays of ints
    # "explode" the dataframe into one row per object per pixel
    # this may create multiple rows per object, the pixel column will now store single ints
    return locations.explode(["pixel"], ignore_index=True)


def load_data(locations, radius, bandlist):
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
    # the catalog is stored in an AWS S3 bucket in region us-east-1
    # load the catalog's metadata as a pyarrow dataset. this will be used to query the catalog
    fs = pyarrow.fs.S3FileSystem(region="us-east-1")
    bucket = "irsa-mast-tike-spitzer-data"
    catalog_root = f"{bucket}/data/NEOWISE/healpix_k{K}/meisner-etal/neo7/meisner-etal-neo7.parquet"
    dataset = pyarrow.dataset.parquet_dataset(f"{catalog_root}/_metadata", filesystem=fs, partitioning="hive")

    # specify which columns will be loaded
    # for a complete list of columns, use: `dataset.schema.names`
    # to load the complete schema, use:
    # schema = pyarrow.dataset.parquet_dataset(f"{catalog_root}/_common_metadata", filesystem=fs).schema
    columns = ["flux", "dflux", "ra", "dec", "band", "MJDMEAN"]

    # iterate over partitions, load data, and find each object
    wise_df_list = []
    for pixel, locs_df in locations.groupby("pixel"):
        # create a filter to pick out sources that are (1) in this partition; and (2) within the
        # coadd's primary region (to avoid duplicates when an object is near the coadd boundary)
        filters = (pyarrow.compute.field(f"healpix_k{K}") == pixel) & (pyarrow.compute.field("primary") == 1)
        # add a filter for the bandlist. if all bands are requested, skip this to avoid the overhead
        if len(set(BANDMAP.keys()) - set(bandlist)) > 0:
            filters = filters & (pyarrow.compute.field("band").isin([BANDMAP[band] for band in bandlist]))

        # query the dataset and load the light curve data
        pixel_tbl = dataset.to_table(filter=filters, columns=columns)

        # do a cone search using astropy to select sources belonging to each object
        pixel_skycoords = SkyCoord(ra=pixel_tbl["ra"] * u.deg, dec=pixel_tbl["dec"] * u.deg)
        objects_skycoords = SkyCoord(locs_df["coord"].to_list())
        object_ilocs, pixel_ilocs, _, _ = pixel_skycoords.search_around_sky(objects_skycoords, radius)

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
        dataframe of light curves as returned by `load_data`.

    Returns
    -------
    wise_df : pd.DataFrame
        the input dataframe, cleaned and transformed.
    """
    # rename columns to match a MultiIndexDFObject
    wise_df = wise_df.rename(columns={"MJDMEAN": "time", "dflux": "err"})
    # filter for only positive fluxes
    wise_df = wise_df[wise_df["flux"] > 0]
    # convert units
    wise_df["flux"] = convert_wise_flux_to_millijansky(nanomaggy_flux=wise_df["flux"])
    wise_df["err"] = convert_wise_flux_to_millijansky(nanomaggy_flux=wise_df["err"])
    # convert the band to its common name ("W1" or "W2")
    wise_df["band"] = wise_df["band"].map(BANDMAP)
    return wise_df
