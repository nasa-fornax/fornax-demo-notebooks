import astropy.units as u
import numpy as np
import pandas as pd
from astropy import nddata
from astropy.coordinates import SkyCoord
from astropy.table import Table
from sparcl.client import SparclClient
from specutils import Spectrum1D

from data_structures_spec import MultiIndexDFObject


def DESIBOSS_get_spec(sample_table, search_radius_arcsec):
    """
    Retrieve DESI and BOSS spectra for a list of sources.
    Note, that we can also retrieve SDSS-DR16 spectra here, which
    leads to similar results as SDSS_get_spec().

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with the coordinates and journal reference labels of the sources.
    search_radius_arcsec : float
        Search radius in arcseconds. Here its rather half a box size.

    Returns
    -------
    MultiIndexDFObject
        The spectra returned from the archive.
    """

    # Set up client
    client = SparclClient()
    # print(client.all_datasets) # print data sets

    # Initialize multi-index object:
    df_spec = MultiIndexDFObject()

    for stab in sample_table:

        # Search
        data_releases = ['DESI-EDR', 'BOSS-DR16']
        # data_releases = ['DESI-EDR','BOSS-DR16','SDSS-DR16'] # we want to use DR17 directly using SDSS query

        search_coords = stab["coord"]
        dra = (search_radius_arcsec*u.arcsec).to(u.degree)
        ddec = (search_radius_arcsec*u.arcsec).to(u.degree)
        out = ['sparcl_id', 'ra', 'dec', 'redshift', 'spectype', 'data_release', 'redshift_err']
        cons = {'spectype': ['GALAXY', 'STAR', 'QSO'],
                'data_release': data_releases,
                # 'redshift': [0.5, 0.9],
                'ra': [search_coords.ra.deg-dra.value, search_coords.ra.deg+dra.value],
                'dec': [search_coords.dec.deg-ddec.value, search_coords.dec.deg+ddec.value]
                }
        found_I = client.find(outfields=out, constraints=cons, limit=20)  # search
        # print(found_I)

        # Extract nice table and the spectra
        if len(found_I.records) > 0:
            result_tab = Table(names=found_I.records[0].keys(), dtype=[type(
                found_I.records[0][key]) for key in found_I.records[0].keys()])
            _ = [result_tab.add_row([f[key] for key in f.keys()]) for f in found_I.records]

            sep = [search_coords.separation(SkyCoord(tt["ra"], tt["dec"], unit=u.deg, frame='icrs')).to(
                u.arcsecond).value for tt in result_tab]
            result_tab["separation"] = sep

            # Retrieve Spectra
            inc = ['sparcl_id', 'specid', 'data_release', 'redshift', 'flux',
                   'wavelength', 'model', 'ivar', 'mask', 'spectype', 'ra', 'dec']
            results_I = client.retrieve(uuid_list=found_I.ids, include=inc)
            specs = [Spectrum1D(spectral_axis=r.wavelength*u.AA,
                                flux=np.array(r.flux) * 10**-17 * u.Unit('erg cm-2 s-1 AA-1'),
                                uncertainty=nddata.InverseVariance(np.array(r.ivar)),
                                redshift=r.redshift,
                                mask=r.mask)
                     for r in results_I.records]

            # Choose objects
            for dr in data_releases:

                sel = np.where(result_tab["data_release"] == dr)[0]
                if len(sel) > 0:  # found entry
                    idx_closest = sel[np.where(result_tab["separation"][sel] == np.nanmin(
                        result_tab["separation"][sel]))[0][0]]

                    # create MultiIndex Object
                    dfsingle = pd.DataFrame(dict(wave=[specs[idx_closest].spectral_axis],
                                                 flux=[specs[idx_closest].flux],
                                                 err=[
                                                     np.sqrt(1/specs[idx_closest].uncertainty.quantity)],
                                                 label=[stab["label"]],
                                                 objectid=[stab["objectid"]],
                                                 mission=[dr],
                                                 instrument=[dr],
                                                 filter=["optical"],
                                                 )).set_index(["objectid", "label", "filter", "mission"])
                    df_spec.append(dfsingle)

    return df_spec
