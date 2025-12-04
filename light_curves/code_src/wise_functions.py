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

BANDMAP = {"WISE_W1": 1, "WISE_W2": 2}  # map the common names to the values actually stored in the catalog
K = 5  # HEALPix order at which the dataset is partitioned


def wise_get_lightcurves(sample_table, *, radius=1.0, bandlist=["WISE_W1", "WISE_W2"]):
    """
    Loads WISE data by searching the unWISE light curve catalog (Meisner et al., 2023AJ....165...36M).
    This is the MAIN function

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table containing the source sample. The following columns must be present:
            coord : astropy.coordinates.SkyCoord
                Sky position of each source.
            objectid : int
                Unique identifier for each source in the sample.
            label : str
                Literature label for tracking source provenance.
    radius : float
        Cone-search radius **in arcseconds** used to match unWISE detections
        to each source. Only detections within this radius of the source
        position are returned.
    bandlist: list of strings
        which WISE bands to search for, example: ['W1', 'W2']

    Returns
    -------
    df_lc : MultiIndexDFObject
        indexed by [objectid, label, band, time]. The resulting internal pandas DataFrame
        contains the following columns:
            flux : float
                Flux in millijansky (mJy).
            err : float
                Flux uncertainty in millijansky (mJy).
            time : float
                Time of observation in MJD.
            objectid : int
                Input sample object identifier.
            band : str
                WISE band label ('W1' or 'W2').
            label : str
                Literature label associated with each source.

    Notes
    -----
    * The unWISE time-domain catalog is stored in parquet format on AWS S3 and
      partitioned by HEALPix index at order K=5. Only relevant partitions
      are loaded, based on the SkyCoord cone search.

    * Fluxes are converted from unWISE native units (DN/s) into millijansky
      using published zero-points.

    * Only detections with positive flux are included.

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
    """
    Identify the HEALPix (order K=5) partitions intersecting a cone of radius
    `radius` around each source. Helps determine which unWISE parquet files
    must be loaded.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table containing the source sample. The following columns must be present:
            coord : astropy.coordinates.SkyCoord
                Sky position of each source.
            objectid : int
                Unique identifier for each source in the sample.
            label : str
                Literature label for tracking source provenance.   
    radius : astropy.units.Quantity
        Angular radius used to determine which HEALPix pixels overlap each
        source. Must be in angular units (e.g., arcsec or deg).
        
    Returns
    -------
    locations : pandas.DataFrame
        Expanded dataframe containing one row per (object, HEALPix pixel)
        combination. Columns include:

            objectid : int  
            label : str  
            coord.ra : float (deg)  
            coord.dec : float (deg)  
            pixel : int
                HEALPix pixel number at K=5 that overlaps the cone.

    Notes
    -----
    * Each source may correspond to multiple HEALPix pixels.
    * The resulting dataframe is used to selectively load only the relevant
      portions of the unWISE catalog.

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
    """
    Load unWISE time-domain photometry from the AWS S3 parquet catalog.
        (Meisner et al., 2023AJ....165...36M).

    Parameters
    ----------
    locations : pandas.DataFrame
        Output of locate_objects() containing:
            objectid   : int
            coord.ra   : float (deg)
            coord.dec  : float (deg)
            label      : str
            pixel      : int
                HEALPix pixel indices to check.
    radius : astropy.units.Quantity (arcsec)
        Cone-search radius in arcseconds defining how close a detection must be 
        to a source to be considered a match.
    bandlist: list of strings
        which WISE bands to search for, example: ['W1', 'W2']

    Returns
    -------
    wise_df : pd.DataFrame
        dataframe of light curve data for all objects in `locations`
            flux : float
                Instrumental WISE flux (raw catalog units DN/s).
            dflux : float
                Instrumental flux uncertainty (DN/s).
            ra : float (deg)
            dec : float (deg)
            band : int
                Encoded WISE band (1=W1, 2=W2).
            MJDMEAN : float
                Mean MJD timestamp for the detection.
            objectid : int
            coord.ra : float (deg)
            coord.dec : float (deg)
            label : str

    Notes
    -----
    * Only detections within the `radius` of each source are returned.
    * If a source lies near a HEALPix boundary, detections from multiple
      partitions may be included.
    * Flux unit conversion to mJy is handled separately in `transform_lightcurves()`.
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
    """
    Clean and transform the data into the form needed for a `MultiIndexDFObject`.

    Parameters
    ----------
    wise_df : pd.DataFrame
        Raw detection table returned by load_lightcurves(), with columns:
            flux       : float (DN/s)
            dflux      : float (DN/s)
            ra         : float
            dec        : float
            band       : int
            MJDMEAN    : float
            objectid   : int
            coord.ra   : float
            coord.dec  : float
            label      : str

    Returns
    -------
    wise_df : pd.DataFrame
        the input dataframe, cleaned and transformed.
            flux : float
                Flux converted to millijansky (mJy).
            err  : float
                Flux uncertainty in millijansky (mJy).
            band : str
                WISE band name ('W1' or 'W2').
            time : float
                MJD timestamp.
            objectid : int
            label    : str
            coord.ra : float
            coord.dec: float
 
    Notes
    -----
    * Only positive-flux detections are retained.
    * Numeric band codes are converted to "W1" or "W2".
    * Flux conversions use `convert_wise_flux_to_millijansky()` and are applied
      separately to each band.
    """
    # rename columns to match a MultiIndexDFObject
    wise_df = wise_df.rename(columns={"MJDMEAN": "time", "dflux": "err"})
    # convert the band to its common name ("WISE_W1" or "WISE_W2"). need to invert the BANDMAP dict.
    wise_df["band"] = wise_df["band"].map({value: key for key, value in BANDMAP.items()})
    # filter for only positive fluxes
    wise_df = wise_df[wise_df["flux"] > 0]
    # convert units, per band
    grouped_wise_df = wise_df.groupby("band")
    wise_df["flux"] = grouped_wise_df["flux"].transform(convert_wise_flux_to_millijansky)
    wise_df["err"] = grouped_wise_df["err"].transform(convert_wise_flux_to_millijansky)
    return wise_df
