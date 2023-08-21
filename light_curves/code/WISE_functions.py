import astropy.units as u
import hpgeom
import pandas as pd
import pyarrow.compute
import pyarrow.dataset
import pyarrow.fs
from astropy.coordinates import SkyCoord
from tqdm import tqdm

from data_structures import MultiIndexDFObject
from fluxconversions import convert_wise_flux_to_mag, convert_WISEtoJanskies
from lsst_formatters import arrow_to_astropy


BANDMAP = {1: "w1", 2: "w2"}

#WISE
def WISE_get_lightcurves(coords_list, labels_list, radius = 1.0 * u.arcsec, bandlist = ['w1', 'w2']):
    """Searches WISE catalog from meisner et al., 2023 for light curves from a list of input coordinates
    
    Parameters
    ----------
    coords_list : list of astropy skycoords
        the coordinates of the targets for which a user wants light curves
    labels_list: list of strings
        journal articles associated with the target coordinates
    radius : float
        search radius, how far from the source should the archives return results
    bandlist: list of strings
        which WISE bands to search for, example: ['w1', 'w2']
        
    Returns
    -------
    df_lc : MultiIndexDFObject
        the main data structure to store all light curves
    """

    fs = pyarrow.fs.S3FileSystem(region="us-east-1")
    bucket = "irsa-mast-tike-spitzer-data"
    catalog_root = f"{bucket}/data/NEOWISE/healpix_k5/meisner-etal/neo7/meisner-etal-neo7.parquet"
    dataset = pyarrow.dataset.parquet_dataset(f"{catalog_root}/_metadata", filesystem=fs, partitioning="hive")
    k = 5  # order at which dataset is partitioned

    # per coord: find pixels
    # groupby pixels
    # per pixel group: load file, search for all coords in pixel
    
    # locate the WISE partitions/pixels each coords_list object is in
    my_coords_list = []
    for objectid, coord in coords_list:
        cone_pixels = hpgeom.query_circle(
            a=coord.ra.deg,
            b=coord.dec.deg,
            radius=radius.to(u.deg).value,
            nside=hpgeom.order_to_nside(k),
            nest=True,  # catalog uses nested ordering scheme for pixel index
            inclusive=True,  # return all pixels that overlap with the circle, and maybe a few more
        )
        my_coords_list.append((objectid, coord, labels_list[objectid], cone_pixels))
    locations = pd.DataFrame(my_coords_list, columns=["objectid", "coord", "label", "pixel"])
    # query_circle returned a list of pixels. explode the dataframe into one row per pixel per object
    locations = locations.explode(["pixel"], ignore_index=True)

    # iterate over partitions/pixels, load data, and find the coords_list objects
    wise_df_list = []
    columns = ["flux", "dflux", "ra", "dec", "band", 'MJDMIN', 'MJDMAX', 'MJDMEAN']
    for pixel, locs_df in locations.groupby("pixel"):
        objects_skycoords = SkyCoord(locs_df["coord"].to_list())
        
        # load all sources (rows) in the partition
        pixel_tbl = dataset.to_table(filter=(pyarrow.compute.field("healpix_k5") == pixel), columns=columns)
        pixel_skycoords = SkyCoord(ra=pixel_tbl["ra"] * u.deg, dec=pixel_tbl["dec"] * u.deg)

        # find sources that are within radius of an object
        object_ilocs, pixel_ilocs, _, _ = pixel_skycoords.search_around_sky(objects_skycoords, radius)
        
        # create a dataframe with all sources
        match_df = pixel_tbl.take(pixel_ilocs).to_pandas()
        # attach the objectid, etc.
        match_df["object_ilocs"] = object_ilocs
        match_df = match_df.set_index("object_ilocs").join(locs_df.reset_index(drop=True))
        
        wise_df_list.append(match_df)
    wise_df = pd.concat(wise_df_list, ignore_index=True)

    # transform the data
    mag, magerr = convert_wise_flux_to_mag(wise_df['flux'], wise_df['dflux'])
    wiseflux, wisefluxerr = convert_WISEtoJanskies(mag, magerr, wise_df["band"])
    time_mjd = wise_df['MJDMEAN']
    band = wise_df["band"].map(BANDMAP)
    lab = wise_df["label"]

    return MultiIndexDFObject(data=pd.DataFrame(dict(flux=wiseflux, err=wisefluxerr, time=time_mjd, 
                                             objectid=objectid, band=band,label=lab)
                                       ).set_index(["objectid","label", "band", "time"]))
