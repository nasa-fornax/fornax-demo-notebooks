import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.ipac.irsa import Irsa

from data_structures_spec import MultiIndexDFObject


def KeckDEIMOS_get_spec(sample_table, search_radius_arcsec):
    """
    Retrieve Keck DEIMOS on COSMOS spectra for a list of sources.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with the coordinates and journal reference labels of the sources.
    search_radius_arcsec : float
        Search radius in arcseconds.

    Returns
    -------
    MultiIndexDFObject
        The spectra returned from the archive.
    """

    # Initialize multi-index object:
    df_spec = MultiIndexDFObject()

    for stab in sample_table:

        search_coords = stab["coord"]
        tab = Irsa.query_region(coordinates=search_coords, catalog="cosmos_deimos",
                                spatial="Cone", radius=search_radius_arcsec * u.arcsec)
        print("Number of entries found: {}".format(len(tab)))

        if len(tab) == 0:
            continue

        # If multiple entries are found, pick the closest.
        if len(tab) > 1:
            print("More than 1 entry found - pick the closest. ")
            sep = [search_coords.separation(SkyCoord(tt["ra"], tt["dec"], unit=u.deg, frame='icrs')).to(
                u.arcsecond).value for tt in tab]
            id_min = np.where(sep == np.nanmin(sep))[0]
            tab_final = tab[id_min]
        else:
            tab_final = tab.copy()

        if "_acal" not in tab_final["fits1d"][0]:
            print("no calibration is available")
            continue

        # Now extract spectrum
        # ASCII 1d spectrum for file spec1d.cos0.034.VD_07891
        # wavelength in Angstroms and observed-frame
        # flux (f_lambda) in 1e-17 erg/s/cm2/A if calibrated, else in counts
        # ivar (inverse variance) in 1e-17 erg/s/cm2/A if calibrated, else in counts
        # wavelength flux ivar
        url = "https://irsa.ipac.caltech.edu{}".format(tab_final["fits1d"][0].split("\"")[1])
        spec = Table.read(url)

        # Prepare arrays
        wave = spec["LAMBDA"][0] * u.angstrom
        flux_cgs = spec["FLUX"][0] * 1e-17 * u.erg/u.second/u.centimeter**2/u.angstrom
        error_cgs = np.sqrt(1 / spec["IVAR"][0]) * 1e-17 * u.erg/u.second/u.centimeter**2/u.angstrom

        # Create MultiIndex object
        dfsingle = pd.DataFrame(dict(wave=[wave],
                                     flux=[flux_cgs],
                                     err=[error_cgs],
                                     label=[stab["label"]],
                                     objectid=[stab["objectid"]],
                                     mission=["Keck"],
                                     instrument=["DEIMOS"],
                                     filter=["optical"],
                                     )).set_index(["objectid", "label", "filter", "mission"])
        df_spec.append(dfsingle)

    return df_spec
