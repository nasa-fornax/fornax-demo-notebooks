import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.ipac.irsa import Irsa

from data_structures_spec import MultiIndexDFObject


def SpitzerIRS_get_spec(sample_table, search_radius_arcsec, COMBINESPEC):
    """
    Retrieve HST spectra for a list of sources.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with the coordinates and journal reference labels of the sources.
    search_radius_arcsec : float
        Search radius in arcseconds.
    COMBINESPEC : bool
        If set to True, then, if multiple spectra are found, the spectra are
        mean-combined. If False, the closest spectrum is chosen and returned.

    Returns
    -------
    MultiIndexDFObject
        The spectra returned from the archive.
    """

    # Initialize multi-index object:
    df_spec = MultiIndexDFObject()

    for stab in sample_table:

        print("Processing source {}".format(stab["label"]))

        # Do search
        search_coords = stab["coord"]
        tab = Irsa.query_region(coordinates=search_coords, catalog="irs_enhv211",
                                spatial="Cone", radius=search_radius_arcsec * u.arcsec)
        print("Number of entries found: {}".format(len(tab)))

        if len(tab) == 0:
            continue

        # If multiple entries are found, pick the closest.
        # Or should we take the average instead??
        if len(tab) > 1:
            print("More than 1 entry found", end="")
            if not COMBINESPEC:
                print(" - pick the closest")
                sep = [search_coords.separation(SkyCoord(tt["ra"], tt["dec"], unit=u.deg, frame='icrs')).to(
                    u.arcsecond).value for tt in tab]
                id_min = np.where(sep == np.nanmin(sep))[0]
                tab_final = tab[id_min]
            else:
                print(" - Combine spectra")
                tab_final = tab.copy()
        else:
            tab_final = tab.copy()

        # Now extract spectra and put all in one array
        specs = []
        for tt in tab_final:
            url = "https://irsa.ipac.caltech.edu{}".format(tt["xtable"].split("\"")[1])
            spec = Table.read(url, format="ipac")  # flux_density in Jy
            specs.append(spec)

        # Create final spectrum (combine if necesary)
        # Note that spectrum is automatically combined if longer than length=1
        lengths = np.asarray([len(spec["wavelength"])
                              for spec in specs])  # get the lengths of the spectra
        # pick the spectra with the most wavelength as template
        id_orig = np.where(lengths == np.nanmax(lengths))[0]
        if len(id_orig) > 0:
            id_orig = id_orig[0]
        wave_orig = np.asarray(specs[id_orig]["wavelength"])  # original wavelength
        fluxes = np.asarray([np.interp(wave_orig, spec["wavelength"],
                            spec["flux_density"]) for spec in specs])
        errors = np.asarray(
            [np.interp(wave_orig, spec["wavelength"], spec["error"]) for spec in specs])
        flux_jy_mean = np.nanmean(fluxes, axis=0)
        errors_jy_combined = np.asarray(
            [np.sqrt(np.nansum(errors[:, ii]**2)) for ii in range(errors.shape[1])])

        # Change units
        wave_orig_A = (wave_orig*u.micrometer).to(u.angstrom)
        flux_cgs_mean = (flux_jy_mean*u.jansky).to(u.erg/u.second/u.centimeter **
                                                   2/u.hertz) * const.c.to(u.angstrom/u.second) / (wave_orig_A**2)
        errors_cgs_combined = (errors_jy_combined*u.jansky).to(u.erg/u.second /
                                                               u.centimeter**2/u.hertz) * const.c.to(u.angstrom/u.second) / (wave_orig_A**2)

        # Create MultiIndex object
        dfsingle = pd.DataFrame(dict(wave=[wave_orig_A],
                                     flux=[flux_cgs_mean],
                                     err=[errors_cgs_combined],
                                     label=[stab["label"]],
                                     objectid=[stab["objectid"]],
                                     mission=["Spitzer"],
                                     instrument=["IRS"],
                                     filter=["IR"],
                                     )).set_index(["objectid", "label", "filter", "mission"])
        df_spec.append(dfsingle)

    return df_spec
