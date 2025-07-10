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