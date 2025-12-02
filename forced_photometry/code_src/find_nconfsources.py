import numpy as np
from determine_source_type import determine_source_type


def find_nconfsources(raval, decval, gal_type, fluxval, x1, y1, cutout_width, subimage_wcs, df):
    """
    Identify the target source and any nearby contaminating sources within a
    cutout, and construct Tractor-ready source descriptions for all of them.

    This function assembles a list of objects that should be included in the
    Tractor photometry fit for a given position. It first creates the source
    description for the target object itself, then searches the input catalog
    for neighboring sources with real fluxes that fall inside the same cutout
    region. For each nearby “confusing” source, the function determines its
    pixel coordinates within the cutout using the provided WCS and constructs
    the appropriate Tractor source object using ``determine_source_type()``.

    Parameters
    ----------
    raval : float
        Right Ascension of the target source in degrees.
    decval : float
        Declination of the target source in degrees.
    gal_type : int
        Source type code for the target (e.g., 0 = galaxy, 1 = star).
    fluxval : float
        Ks-band aperture flux for the target source. Used to set the Tractor
        initial flux estimate.
    x1, y1 : float
        Pixel coordinates of the target source within the cutout image.
    cutout_width : float
        Width of the cutout in arcseconds. Used to identify neighboring sources.
    subimage_wcs : astropy.wcs.WCS
        WCS object for the cutout image. Used to convert RA/Dec of confusing
        sources into cutout pixel coordinates.
    df : pandas.DataFrame
        Full catalog containing RA, Dec, Ks-band fluxes, and source types
        for all objects in the region.

    Returns
    -------
    objsrc : list
        A list of Tractor-ready source objects, with the target source first,
        followed by any confusing sources within the cutout.
    nconfsrcs : int
        Number of confusing sources identified within the cutout region.

    Notes
    -----
    A “confusing source” is any catalog object with positive Ks-band flux that
    falls inside the cutout (excluding the target itself) and lies farther than
    0.2 arcseconds away from the target position. These sources are included so
    that Tractor can model blended or nearby objects correctly during the
    forced-photometry fit.

    """

    # Setup to collect sources
    objsrc = []

    # Keep the main source
    objsrc.append(determine_source_type(raval, decval, gal_type, fluxval, x1, y1))

    # Find confusing sources with real fluxes
    radiff = (df.ra - raval) * np.cos(decval)
    decdiff = df.dec - decval
    posdiff = np.sqrt(radiff**2 + decdiff**2) * 3600.
    det = df.ks_flux_aper2 > 0  # make sure they have fluxes

    # Make an index into the dataframe for those objects within the same cutout
    good = (abs(radiff * 3600.) < cutout_width / 2) & (abs(decdiff * 3600.) < cutout_width / 2) & (posdiff > 0.2) & det
    nconfsrcs = np.size(posdiff[good])

    # Add confusing sources
    # if there are any confusing sources
    if nconfsrcs > 0:
        ra_conf = df.ra[good].values
        dec_conf = df.dec[good].values
        flux_conf = df.ks_flux_aper2[good].values  # should all be real fluxes
        type_conf = df.type[good].values

        for n in range(nconfsrcs):
            # Now need to set the values of x1, y1 at the location of the target *in the cutout*
            xn, yn = subimage_wcs.all_world2pix(ra_conf[n], dec_conf[n], 1)
            objsrc.append(determine_source_type(ra_conf[n], dec_conf[n], type_conf[n], flux_conf[n], xn, yn))

    return objsrc, nconfsrcs
