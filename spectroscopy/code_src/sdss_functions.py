import astropy.units as u
import numpy as np
import pandas as pd
from astroquery.sdss import SDSS

from data_structures_spec import MultiIndexDFObject


def SDSS_get_spec(sample_table, search_radius_arcsec, data_release):
    """
    Retrieve SDSS spectra for a list of sources. Note that no data will
    be directly downloaded. All will be saved in cache.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with the coordinates and journal reference labels of the sources.
    search_radius_arcsec : float
        Search radius in arcseconds.
    data_release : int
        SDSS data release (e.g., 17 or 18).

    Returns
    -------
    MultiIndexDFObject
        The spectra returned from the archive.
    """

    # Initialize multi-index object:
    df_spec = MultiIndexDFObject()

    for stab in sample_table:

        # Get Spectra for SDSS
        search_coords = stab["coord"]

        xid = SDSS.query_region(search_coords, radius=search_radius_arcsec *
                                u.arcsec, spectro=True, data_release=data_release)

        if str(type(xid)) == "<class 'NoneType'>":
            print("Source {} could not be found".format(stab["label"]))
            continue

        sp = SDSS.get_spectra(matches=xid, show_progress=True, data_release=data_release)

        # Get data
        # only one entry because we only search for one xid at a time. Could change that?
        wave = 10**sp[0]["COADD"].data.loglam * u.angstrom
        flux = sp[0]["COADD"].data.flux*1e-17 * u.erg/u.second/u.centimeter**2/u.angstrom
        err = np.sqrt(1/sp[0]["COADD"].data.ivar)*1e-17 * flux.unit

        # Add to df_spec.
        dfsingle = pd.DataFrame(dict(wave=[wave], flux=[flux], err=[err],
                                     label=[stab["label"]],
                                     objectid=[stab["objectid"]],
                                     mission=["SDSS"],
                                     instrument=["SDSS"],
                                     filter=["optical"],
                                     )).set_index(["objectid", "label", "filter", "mission"])
        df_spec.append(dfsingle)

    return df_spec
