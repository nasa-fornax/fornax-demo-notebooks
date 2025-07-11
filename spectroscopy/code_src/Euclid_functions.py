import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.io import fits
from astroquery.ipac.irsa import Irsa

from data_structures_spec import MultiIndexDFObject
import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)


def get_coord_from_objectid(object_id, table_mer="euclid_q1_mer_catalogue"):
    """
    Query IRSA's cloud-based TAP service to retrieve the sky coordinates of a Euclid object.
    This function performs an ADQL query against the specified MER catalog hosted at IRSA
    and returns the right ascension and declination of the object as a SkyCoord instance.

    Parameters
    ----------
    object_id : int or str
        The Euclid object ID to look up.

    table_mer : str, optional
        The name of the IRSA TAP-accessible MER catalog table to query.
        Defaults to "euclid_q1_mer_catalogue".

    Returns
    -------
    coord : astropy.coordinates.SkyCoord or None
        The sky coordinates (RA, Dec) of the object, if found.
        Returns None if the query fails or if no match is found.

    Notes
    -----
    - This function queries a remote (cloud-based) TAP service and requires internet access.
    - If no result is found or if the query fails, a warning is printed and None is returned.
    """
    
    adql_query = f"""
    SELECT ra, dec FROM {table_mer}
    WHERE object_id = {object_id}
    
    """
    try:
        result = Irsa.query_tap(adql_query).to_table()
        if len(result) == 0:
            print(f"No match found for object_id {object_id}")
            return None
        ra, dec = result[0]["ra"], result[0]["dec"]
        return SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    except Exception as e:
        print(f"Query failed for object_id {object_id}: {e}")
        return None


def Euclid_get_spec(sample_table, search_radius_arcsec):
    """
    Retrieve Euclid 1D spectra for sources in a sample table using IRSA TAP cloud services.

    This function performs a cone search on the Euclid MER catalog via IRSA (a cloud-based service),
    retrieves associated 1D spectral file URIs from a TAP-accessible association table,
    downloads the spectra FITS files, extracts flux and error, and packages them
    into a MultiIndexDFObject.

    Parameters
    ----------
    sample_table : list of dict
        A list of source dictionaries, each containing:
        - 'label' : str
            A unique identifier for the source (used for indexing).
        - 'coord' : astropy.coordinates.SkyCoord
            The sky coordinates of the source.

    search_radius_arcsec : float
        Search radius (in arcseconds) for the initial cone search around each coordinate.

    Returns
    -------
    df_spec : MultiIndexDFObject
        A structured multi-indexed DataFrame containing:
        - wave : Quantity array
        - flux : Quantity array
        - err : Quantity array
        along with metadata fields ('label', 'objectid', 'mission', 'instrument', 'filter').

    Notes
    -----
    - This function accesses IRSA services and downloads FITS files from the cloud.
    - It assumes valid internet connectivity and may fail or be slow under network issues.
    - The function suppresses Astropy warnings during parsing.
    """
    df_spec = MultiIndexDFObject()
    table_mer = "euclid_q1_mer_catalogue"
    table_1dspectra = "euclid.objectid_spectrafile_association_q1"

    for stab in sample_table:
        print(f"Processing source {stab['label']}")
        coord = stab["coord"]

        # Step 1: Cone search in MER catalog
        try:
            tab = Irsa.query_region(
                coordinates=coord,
                catalog=table_mer,
                spatial="Cone",
                radius=search_radius_arcsec * u.arcsec
            )
        except Exception as e:
            print(f"Query failed for {stab['label']}: {e}")
            continue

        if len(tab) == 0:
            print("No match found in Euclid MER catalog.")
            continue

        # Pick the closest match
        closest_idx = np.argmin([coord.separation(SkyCoord(ra=t["ra"] * u.deg, dec=t["dec"] * u.deg)).arcsec for t in tab])
        object_id = tab[closest_idx]["object_id"]
        print(f"Found Euclid object_id: {object_id}")

        # Step 2: Query spectrum association TAP table
        adql_query = f"""
        SELECT * FROM {table_1dspectra}
        WHERE objectid = {object_id}
        """
        try:
            result = Irsa.query_tap(adql_query).to_table()
            if len(result) == 0:
                print(f"No 1D spectrum found for object_id {object_id}")
                continue

            uri = result[0]["uri"]
            hdu_index = result[0]["hdu"]
            file_uri = f"https://irsa.ipac.caltech.edu/{uri}"

            with fits.open(file_uri, ignore_missing_simple=True) as hdul:
                spec = QTable.read(hdul[hdu_index], format="fits")
                header = hdul[hdu_index].header

        except Exception as e:
            print(f"Failed to retrieve/read spectrum for object_id {object_id}: {e}")
            continue

        # Step 3: Extract and scale spectrum
        try:
            fscale = header.get("FSCALE", 1.0)

            wave = np.asarray(spec["WAVELENGTH"]) * u.angstrom
            signal = np.asarray(spec["SIGNAL"])
            var = np.asarray(spec["VAR"])
            mask = np.asarray(spec["MASK"])

            if not (len(wave) == len(signal) == len(var) == len(mask)):
                raise ValueError("Spectrum array length mismatch")

            # Good data mask: even values and <64
            valid = (mask % 2 == 0) & (mask < 64)

            wave = wave[valid]
            flux = signal[valid] * fscale * u.erg / u.s / u.cm**2 / u.angstrom
            error = np.sqrt(var[valid]) * fscale * flux.unit

        except Exception as e:
            print(f"Could not parse spectrum for object_id {object_id}: {e}")
            continue

        # Step 4: Package into MultiIndexDFObject
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
