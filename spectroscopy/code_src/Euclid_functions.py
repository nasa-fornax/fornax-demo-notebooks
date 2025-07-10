import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.io import fits
from astroquery.ipac.irsa import Irsa

from data_structures_spec import MultiIndexDFObject


def Euclid_get_spec(sample_table, search_radius_arcsec):
    """
    Retrieve Euclid NISP 1D spectra from IRSA using the Q1 MER catalog and TAP access.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with coordinates, labels, and object IDs of the sources.
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
        if len(tab) > 1:
            sep = [coord.separation(SkyCoord(ra=tt["ra"] * u.deg, dec=tt["dec"] * u.deg)).arcsec for tt in tab]
            closest_idx = np.argmin(sep)
            tab_final = tab[[closest_idx]]
        else:
            tab_final = tab.copy()

        object_id = tab_final[0]["object_id"]
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

            # Step 3: Open FITS and read spectrum from correct HDU
            with fits.open(file_uri, ignore_missing_simple=True) as hdul:
                spec = QTable.read(hdul[hdu_index], format="fits")
                spec_header = hdul[hdu_index].header

        except Exception as e:
            print(f"Failed to retrieve/read spectrum for object_id {object_id}: {e}")
            continue

        # Step 4: Extract and clean spectrum
        # Step 4: Extract and convert data
        try:
            wave = spec["WAVELENGTH"] * u.angstrom  # already angstrom, do not convert again
            fscale = spec_header.get("FSCALE", 1.0)
            flux = spec["SIGNAL"] * fscale * u.erg / u.second / (u.centimeter**2) / u.angstrom
            error = np.sqrt(spec["VAR"]) * fscale * flux.unit

            # Mask filtering (optional)
            mask = np.array(spec["MASK"])  # extract as plain int array
            valid = (mask % 2 == 0) & (mask < 64)

            wave = wave[valid]
            flux = flux[valid]
            error = error[valid]

        except Exception as e:
            print(f"Could not parse spectrum for object_id {object_id}: {e}")
            continue

        # Step 5: Wrap into MultiIndexDFObject
        dfsingle = pd.DataFrame(dict(
            wave=[wave],
            flux=[flux],
            err=[err],
            label=[stab["label"]],
            objectid=[stab["objectid"]],
            mission=["Euclid"],
            instrument=["NISP"],
            filter=["YJH"]
        )).set_index(["objectid", "label", "filter", "mission"])

        df_spec.append(dfsingle)

    return df_spec
