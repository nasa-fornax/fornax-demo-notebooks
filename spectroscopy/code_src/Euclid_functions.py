import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.io import fits
from astroquery.ipac.irsa import Irsa
from data_structures_spec import MultiIndexDFObject
from pyvo.dal import DALQueryError
import requests
import warnings

# We filter these specific warnings out as the end user can do nothing to about them, they have to be fixed at the data providers level.
# Remove the filters once they are not used in a newer data release
warnings.filterwarnings("ignore", "'Number' did not parse as fits unit", u.UnitsWarning)
warnings.filterwarnings("ignore", "'erg/s/cm2/Angstrom' contains multiple slashes", u.UnitsWarning)
warnings.filterwarnings("ignore", "'erg2/s2/cm4/Angstrom2' contains multiple slashes", u.UnitsWarning)


def get_coord_from_objectid(object_id, table_mer="euclid_q1_mer_catalogue"):
    """
    Query IRSA's TAP service for object coordinates given object_id.

    Parameters
    ----------
    object_id : int or str
        The Euclid object ID to look up.

    table_mer : str, optional
        Name of the IRSA TAP-accessible MER catalog table.

    Returns
    -------
    coord : astropy.coordinates.SkyCoord or None
        SkyCoord of the object if found, None if not found.

    """
    adql_query = f"""
    SELECT ra, dec FROM {table_mer}
    WHERE object_id = {object_id}
    """
    try:
        result = Irsa.query_tap(adql_query).to_table()
    except (DALQueryError, requests.exceptions.RequestException) as e:
        raise RuntimeError(f"IRSA TAP query failed for object_id {object_id}: {e}")

    if len(result) == 0:
        print(f"No match found for object_id {object_id}")
        return None

    ra, dec = result[0]["ra"], result[0]["dec"]
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg)


def euclid_get_spec(sample_table, search_radius_arcsec):
    """
    Retrieve Euclid 1D spectra for sources in a sample table using IRSA TAP cloud services.

    This function performs a cone search on the Euclid MER catalog via IRSA (a cloud-based service),retrieves associated 1D spectral file URIs from a TAP-accessible association table,
    downloads the spectra FITS files, extracts flux and error, and packages them
    into a MultiIndexDFObject.

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
    df_spec = MultiIndexDFObject()
    table_mer = "euclid_q1_mer_catalogue"
    table_1dspectra = "euclid.objectid_spectrafile_association_q1"

    for stab in sample_table:
        print(f"Processing source {stab['label']}")
        coord = stab["coord"]

        try:
            tab = Irsa.query_region(
                coordinates=coord,
                catalog=table_mer,
                spatial="Cone",
                radius=search_radius_arcsec * u.arcsec
            )
        except (DALQueryError, requests.exceptions.RequestException) as e:
            print(f"IRSA cone search failed for {stab['label']}: {e}")
            continue

        if len(tab) == 0:
            print(f"No match found in Euclid MER catalog for {stab['label']}.")
            continue

        closest_idx = np.argmin([
            coord.separation(SkyCoord(ra=t["ra"] * u.deg, dec=t["dec"] * u.deg)).arcsec
            for t in tab
        ])
        object_id = tab[closest_idx]["object_id"]
        print(f"Found Euclid object_id: {object_id}")

        adql_query = f"""
        SELECT * FROM {table_1dspectra}
        WHERE objectid = {object_id}
        """
        try:
            result = Irsa.query_tap(adql_query).to_table()
        except (DALQueryError, requests.exceptions.RequestException) as e:
            print(f"IRSA spectrum query failed for object_id {object_id}: {e}")
            continue

        if len(result) == 0:
            print(f"No 1D spectrum found for object_id {object_id}")
            continue

        uri = result[0]["uri"]
        hdu_index = result[0]["hdu"]
        file_uri = f"https://irsa.ipac.caltech.edu/{uri}"

        try:
            with fits.open(file_uri, ignore_missing_simple=True) as hdul:
                spec = QTable.read(hdul[hdu_index], format="fits")
                header = hdul[hdu_index].header
        except (OSError, IndexError) as e:
            print(f"Failed to read spectrum FITS file for object_id {object_id}: {e}")
            continue

        try:
            fscale = header.get("FSCALE", 1.0)
            wave = np.asarray(spec["WAVELENGTH"]) * u.angstrom
            signal = np.asarray(spec["SIGNAL"])
            var = np.asarray(spec["VAR"])
            mask = np.asarray(spec["MASK"])

            if not (len(wave) == len(signal) == len(var) == len(mask)):
                raise ValueError(f"Spectrum array length mismatch for object_id {object_id}")

            valid = (mask % 2 == 0) & (mask < 64)
            wave = wave[valid]
            flux = signal[valid] * fscale * u.erg / u.s / u.cm**2 / u.angstrom
            error = np.sqrt(var[valid]) * fscale * flux.unit
        except (KeyError, ValueError) as e:
            print(f"[ERROR] Could not parse spectrum for object_id {object_id}: {e}")
            continue

        dfsingle = pd.DataFrame([{
            "wave": wave,
            "flux": flux,
            "err": error,
            "label": str(stab["label"]),
            "objectid": int(stab["objectid"]),
            "mission": "Euclid",
            "instrument": "NISP",
            "filter": "RGS"
        }]).set_index(["objectid", "label", "filter", "mission"])

        df_spec.append(dfsingle)

    return df_spec
