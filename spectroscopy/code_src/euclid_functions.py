import warnings

import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astroquery.ipac.irsa import Irsa
from pyvo.dal import DALQueryError

from data_structures_spec import MultiIndexDFObject

# We filter these specific warnings out as the end user can do nothing to about them, they have to be fixed at the data providers level.
# Remove the filters once they are not used in a newer data release
warnings.filterwarnings("ignore", "The unit 'Angstrom' has been deprecated ", u.UnitsWarning)
warnings.filterwarnings("ignore", "The unit 'erg' has been deprecated ", u.UnitsWarning)


def euclid_get_spec(sample_table, search_radius_arcsec, verbose=True):
    """
    Retrieve Euclid 1D spectra for sources in a sample table using IRSA's
    Simple Spectral Access (SSA) service.

    This function queries IRSA’s SSA endpoint for Euclid NISP 1D spectra
    around each input sky position, selects the single spectrum whose
    on-sky position is closest to the query coordinate, downloads the
    corresponding spectral file via its SSA access URL, and packages the
    result into a MultiIndexDFObject.


    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with the coordinates and journal reference labels of the sources.

    search_radius_arcsec : float
        Search radius in arcseconds.

    verbose : bool, optional
        If True, print status messages when spectra are found or not found.

    Returns
    -------
    MultiIndexDFObject
        The spectra returned from the archive.

    """
    # Container for all spectra returned from the archive
    df_spec = MultiIndexDFObject()
    
    # Hard-coded SSA collection name 
    euclid_ssa_collection = "euclid_DpdSirCombinedSpectra"

    # Convert search radius once
    radius = u.Quantity(search_radius_arcsec, unit=u.arcsec)

    # Expected Euclid spectrum column names (same as original query_region path)
    required_cols = ["WAVELENGTH", "SIGNAL", "VAR", "MASK"]

    # Loop over each source in the input sample table
    for stab in sample_table:
        
        # Sky position and label of the source
        coord = stab["coord"]
        label = str(stab["label"])
        
        # Query the IRSA SSA service for spectra near this position
        try:
            ssa_result = Irsa.query_ssa(
                pos=coord,
                radius=radius,
                collection=euclid_ssa_collection,
            )
        except (DALQueryError, requests.exceptions.RequestException) as e:
            # SSA query-level failure (service down, bad response, network error)
            warnings.warn(
                f"SSA query failed for {label}: {e}",
                RuntimeWarning,
            )
            continue

        # If no spectra are returned
        if ssa_result is None or len(ssa_result) == 0:
            if verbose:
                print(
                    f"No Euclid SSA spectra found for {label} within "
                    f"{radius.to_value(u.arcsec)} arcsec."
                )
            continue

        # Report success if requested
        if verbose:
            print(f"Found Euclid SSA spectra for {label}")

        # Pick the single nearest SSA row to the query coordinate using SSA-provided sky positions
        # SSA standard commonly provides these as 's_ra' and 's_dec' in degrees.

        # Build SkyCoord array for all returned spectra positions
        ssa_coords = SkyCoord(ssa_result["s_ra"], ssa_result["s_dec"], unit=u.deg)

        # Compute separations and choose the closest SSA row
        seps = coord.separation(ssa_coords)
        closest_idx = int(np.argmin(seps))
        row = ssa_result[closest_idx]
 
        # Read the spectrum file from the SSA access URL
        spectrum_url = row["access_url"]
        spec = QTable.read(spectrum_url)

        #filter out bad or conatminated parts of the Euclid spectrum using the bitmask
        valid = (spec["MASK"] % 2 == 0) & (spec["MASK"] < 64)

        dfsingle = pd.DataFrame([{
            "wave": spec["WAVELENGTH"][valid],
            "flux": spec["SIGNAL"][valid],
            "err": np.sqrt(spec["VAR"][valid]),
            "label": label,
            "objectid": int(stab["objectid"]),
            "mission": "Euclid",
            "instrument": "NISP",
            "filter": "RGS"
        }]).set_index(["objectid", "label", "filter", "mission"])

        df_spec.append(dfsingle)

    return df_spec