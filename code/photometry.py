import sys
import os
import warnings
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import numpy as np

from cutout import make_cutouts

# temporarily let the notebook start without tractor as dependency
try:
    from tractor import (Tractor, PixelizedPSF, NullWCS,
                         NullPhotoCal, ConstantSky, Image)

    from find_nconfsources import find_nconfsources

except ImportError:
    print("tractor is missing")
    pass


def setup_for_tractor(*, band_configs, ra, dec, stype, ks_flux_aper2, infiles, df):
    """Create image cutouts, calculate sky background statistics, and find nearby sources.

    Parameters:
    -----------
    band_configs: BandConfigs
        Settings for a single band. See above for the definition of BandConfigs.
    ra, dec: float or double
        celestial coordinates for measuring photometry
    stype: int
        0, 1, 2, -9 for star, galaxy, x-ray source
    ks_flux_aper_2: float
        flux in aperture 2
    infiles: Tuple[str, str]
        Paths to FITS files to be used as the input science and sky-background images respectively.
    df: pd.DataFrame
        <need a description of this parameter>.
        Previous arguments (ra, dec, stype, ks_flux_aper_2) come from a single row of this df.
        However, we must also pass the entire dataframe in order to find nearby sources which are possible contaminates.

    Returns:
    --------
    subimage:
        Science cutout.
    objsrc: List[tractor.ducks.Source]
        List of tractor Source objects for the target and nearby sources.
    nconfsrcs: int
        Number of nearby confusing sources
    skymean: float
        Mean of sigma-clipped background
    skynoise: float
        Standard deviation of sigma-clipped background
    """
    # tractor doesn't need the entire image, just a small region around the object of interest
    subimage, x1, y1, subimage_wcs, bgsubimage = make_cutouts(
        ra, dec, infiles=infiles, cutout_width=band_configs.cutout_width, mosaic_pix_scale=band_configs.mosaic_pix_scale
    )

    # set up the source list by finding neighboring sources
    objsrc, nconfsrcs = find_nconfsources(
        ra, dec, stype, ks_flux_aper2, x1, y1, band_configs.cutout_width, subimage_wcs, df
    )

    # measure sky noise and mean level
    # suppress warnings about nans in the calculation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        skymean, skymedian, skynoise = sigma_clipped_stats(
            bgsubimage, sigma=3.0)

    return subimage, objsrc, nconfsrcs, skymean, skynoise


def run_tractor(*, subimage, prf, objsrc, skymean, skynoise):
    """Make the tractor image and perform forced photometry.

    Parameters:
    -----------
    subimage:
        Science cutout.
    prf:
        <need parameter definition>
    objsrc: List[tractor.ducks.Source]
        List of tractor Source objects for the target and nearby sources.
    skymean: float
        Mean of sigma-clipped background
    skynoise: float
        Standard deviation of sigma-clipped background

    Returns:
    -------
    flux_var:
        <need description>. NaN if the tractor optimization failed.
    fit_fail: bool
        Whether the tractor optimization failed.
    """
    # make the tractor image
    tim = Image(
        data=subimage,
        invvar=np.ones_like(subimage) / skynoise**2,
        psf=PixelizedPSF(prf),
        wcs=NullWCS(),
        photocal=NullPhotoCal(),
        sky=ConstantSky(skymean),
    )

    # make tractor object combining tractor image and source list
    tractor = Tractor([tim], objsrc)  # [src]

    # freeze the parameters we don't want tractor fitting
    tractor.freezeParam("images")  # now fits 2 positions and flux
    # tractor.freezeAllRecursive()#only fit for flux
    # tractor.thawPathsTo('brightness')

    # run the tractor optimization (do forced photometry)
    # Take several linearized least squares steps
    fit_fail = False
    try:
        tr = 0
        with suppress_stdout():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*divide by zero.*")
                # warnings.simplefilter('ignore')
                for tr in range(20):
                    dlnp, X, alpha, flux_var = tractor.optimize(variance=True)
                    # print('dlnp',dlnp)
                    if dlnp < 1e-3:
                        break

    # catch exceptions and bad fits

    except Exception:
        fit_fail = True
        flux_var = np.nan

    return flux_var, fit_fail


def interpret_tractor_results(*, flux_var, flux_conv, fit_fail, objsrc, nconfsrcs):
    """Convert the tractor results to a flux and uncertainty.

    <needs brief description of the logic>

    Parameters:
    -----------
    flux_var:
        <need description>. NaN if the tractor optimization failed.
    flux_conv: float
        factor used to convert tractor result to microjanskies
    fit_fail: bool
        Whether the tractor optimization failed.
    objsrc: List[tractor.ducks.Source]
        List of tractor Source objects for the target and nearby sources.
    nconfsrcs: int
        Number of nearby confusing sources

    Returns:
    --------
    flux: float
        measured flux in microJansky, NaN if unmeasurable
    unc: float
        measured uncertainty in microJansky, NaN if not able to estimate
    """
    # record the photometry results
    if fit_fail:
        # tractor fit failed
        # set flux and uncertainty as nan and move on
        return (np.nan, np.nan)

    if flux_var is None:
        # fit worked, but flux variance did not get reported
        params_list = objsrc[0].getParamNames()
        bindex = params_list.index("brightness.Flux")
        flux = objsrc[0].getParams()[bindex]
        # convert to microjanskies
        microJy_flux = flux * flux_conv
        return (microJy_flux, np.nan)

    # if we get here, fit and variance worked
    params_list = objsrc[0].getParamNames()
    bindex = params_list.index("brightness.Flux")
    flux = objsrc[0].getParams()[bindex]

    # determine flux uncertainty
    # which value of flux_var is for the flux variance?
    # assumes we are fitting positions and flux
    fv = ((nconfsrcs + 1) * 3) - 1
    # fv = ((nconfsrcs+1)*1) - 1  #assumes we are fitting only flux

    tractor_std = np.sqrt(flux_var[fv])

    # convert to microjanskies
    microJy_flux = flux * flux_conv
    microJy_unc = tractor_std * flux_conv

    return microJy_flux, microJy_unc


@contextmanager
def suppress_stdout():
    """Supress output of tractor.

    Seems to be the only way to make it be quiet and not output every step of optimization
    https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
