import astropy.units as u
import hpgeom
import pandas as pd
import pyarrow.compute
import pyarrow.dataset
import pyarrow.fs
from astropy.coordinates import SkyCoord

from data_structures import MultiIndexDFObject
from fluxconversions import convert_WISEtoJanskies


BANDMAP = {1: "w1", 2: "w2"}
K = 5  # HEALPix order at which the dataset is partitioned


def WISE_get_lightcurves(coords_list, labels_list, radius = 1.0 * u.arcsec, bandlist = ['w1', 'w2']):
    """Searches WISE catalog from meisner et al., 2023 for light curves from a list of input coordinates
    
    Parameters
    ----------
    coords_list : list of astropy skycoords
        the coordinates of the targets for which a user wants light curves
    labels_list: list of strings
        journal articles associated with the target coordinates
    radius : astropy Quantity
        search radius, how far from the source should the archives return results
    bandlist: list of strings
        which WISE bands to search for, example: ['w1', 'w2']
        
    Returns
    -------
    df_lc : MultiIndexDFObject
        the main data structure to store all light curves
    """

    # locate the partitions/pixels each coords_list object is in
    locations = locate_objects(coords_list, labels_list, radius)

    # load the data
    wise_df = load_data(locations, radius, bandlist)

    # transform the data for a MultiIndexDFObject
    wise_df = wise_df[wise_df['flux'] > 0]
    wise_df = convert_WISEtoJanskies(wise_df)
    wise_df["band"] = wise_df["band"].map(BANDMAP)
    wise_df = wise_df.rename(columns={"MJDMEAN": "time", "dflux": "err"})

    return MultiIndexDFObject(data=wise_df.set_index(["objectid","label", "band", "time"])[["flux", "err"]])


def locate_objects(coords_list, labels_list, radius):
    """Locate the partitions (HEALPix order 5 pixels) each coords_list object is in.
    
    Parameters
    ----------
    coords_list : list of astropy skycoords
        the coordinates of the targets for which a user wants light curves
    labels_list: list of strings
        journal articles associated with the target coordinates
    radius : astropy Quantity
        search radius, how far from the source should the archives return results
        
    Returns
    -------
    locations : pd.DataFrame
        dataframe of location info including ["objectid", "coord", "label", "pixel"]
    """
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
    locations = pd.DataFrame(my_coords_list, columns=["objectid", "coord", "label", "pixel"])
    # query_circle returned a list of pixels. explode the dataframe into one row per pixel per object
    return locations.explode(["pixel"], ignore_index=True)


def load_data(locations, radius, bandlist):
    """Load light curve data from the Parquet version of the Meisner et al. (2023) catalog.
    
    Parameters
    ----------
    locations : pd.DataFrame
        dataframe of location info as returned by `locate_objects`
    radius : astropy Quantity
        search radius, how far from the source should the archives return results
    bandlist: list of strings
        which WISE bands to search for, example: ['w1', 'w2']
        
    Returns
    -------
    wise_df : pd.DataFrame
        dataframe of light curve data for all objects in `locations`
    """
    fs = pyarrow.fs.S3FileSystem(region="us-east-1")
    bucket = "irsa-mast-tike-spitzer-data"
    catalog_root = f"{bucket}/data/NEOWISE/healpix_k{K}/meisner-etal/neo7/meisner-etal-neo7.parquet"
    dataset = pyarrow.dataset.parquet_dataset(f"{catalog_root}/_metadata", filesystem=fs, partitioning="hive")

    # iterate over partitions/pixels, load data, and find the coords_list objects
    wise_df_list = []
    columns = ["flux", "dflux", "ra", "dec", "band", 'MJDMEAN']
    for pixel, locs_df in locations.groupby("pixel"):
        # filter for partition
        filters = (pyarrow.compute.field(f"healpix_k{K}") == pixel)
        # filter for bandlist. if all bands are requested, skip this
        if len(set(BANDMAP.values()) - set(bandlist)) > 0:
            filters = filters & (pyarrow.compute.field("band").isin(bandlist))
        # load
        pixel_tbl = dataset.to_table(filter=filters, columns=columns)

        # find sources that are within `radius` of any object
        pixel_skycoords = SkyCoord(ra=pixel_tbl["ra"] * u.deg, dec=pixel_tbl["dec"] * u.deg)
        objects_skycoords = SkyCoord(locs_df["coord"].to_list())
        object_ilocs, pixel_ilocs, _, _ = pixel_skycoords.search_around_sky(objects_skycoords, radius)
        
        # create a dataframe with all matched sources
        match_df = pixel_tbl.take(pixel_ilocs).to_pandas()
        # attach the objectid, etc. by joining with locs_df
        match_df["object_ilocs"] = object_ilocs
        match_df = match_df.set_index("object_ilocs").join(locs_df.reset_index(drop=True))
        
        wise_df_list.append(match_df)
    return pd.concat(wise_df_list, ignore_index=True)
