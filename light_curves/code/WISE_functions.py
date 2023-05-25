import astropy.units as u
import hpgeom as hp
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from astropy.coordinates import SkyCoord
from pyarrow.fs import S3FileSystem
from tqdm import tqdm

from data_structures import MultiIndexDFObject
from fluxconversions import convert_wise_flux_to_mag, convert_WISEtoJanskies
from lsst_formatters import arrow_to_astropy


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
    df_lc : pandas dataframe
        the main data structure to store all light curves
    """

    fs = S3FileSystem(region="us-west-1")
    # bucket = "nasa-irsa-wise"
    bucket = "irsa-parquet-raen-test"
    catalog_root = f"{bucket}/wise/neowiser/catalogs/meisner-etal-23/parquet/meisner-etal-23.parquet"
    dataset = ds.parquet_dataset(f"{catalog_root}/_metadata", filesystem=fs, partitioning="hive")

    df_lc = MultiIndexDFObject()
    for objectid, coord in tqdm(coords_list):
        lab = labels_list[objectid]

        pixels = hp.query_circle(nside=hp.order_to_nside(5), a=coord.ra.deg, b=coord.dec.deg,
                                 radius=radius.to(u.deg).value, nest=True, inclusive=True)
        src_tbl = dataset.to_table(columns=["ra", "dec", "band", "flux", "dflux", "MJDMEAN"], 
                                   filter=(pc.field("healpix_k5").isin(pixels)))  # pyarrow table
        src_tbl = arrow_to_astropy(src_tbl)  # astropy table
        wise_coords = SkyCoord(ra=src_tbl["ra"] * u.degree, dec=src_tbl["dec"] * u.degree)
        
        result_table = src_tbl[coord.separation(wise_coords) < radius]
        if (len(result_table) > 0):
            #got a live one
            for bcount, band in enumerate(bandlist):
                #sort by band
                mask = result_table['band'] == (bcount + 1)
                result_table_band = result_table[mask]

                mag, magerr = convert_wise_flux_to_mag(result_table_band['flux'], result_table_band['dflux'])

                wiseflux, wisefluxerr = convert_WISEtoJanskies(mag,magerr ,band)

                time_mjd = result_table_band['MJDMEAN']
        
                #plt.figure(figsize=(8, 4))
                #plt.errorbar(time_mjd, wiseflux, wisefluxerr)
                dfsingle = pd.DataFrame(dict(flux=wiseflux, err=wisefluxerr, time=time_mjd, 
                                             objectid=objectid, band=band,label=lab)
                                       ).set_index(["objectid","label", "band", "time"])

                #then concatenate each individual df together
                df_lc.append(dfsingle)
        else:        
            print("There is no WISE light curve for this object")
        
    return(df_lc)
