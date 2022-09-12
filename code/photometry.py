from contextlib import contextmanager
import os
import sys
import warnings

from astropy.stats import sigma_clipped_stats
import numpy as np

# temporarily let the notebook start without tractor as dependency
try:
    from tractor import (Tractor, PixelizedPSF, NullWCS,
                         NullPhotoCal, ConstantSky, Image)

except ImportError:
    print("tractor is missing")
    pass


def calc_background(*, bkgsubimage):
    """Measure sky noise and mean level.

    Parameters:
    -----------
    bkgsubimage: <need type>
        Background image cutout

    Returns:
    --------
    skymean: float
        Mean of sigma-clipped background
    skynoise: float
        Standard deviation of sigma-clipped background
    """
    # suppress warnings about nans in the calculation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        skymean, skymedian, skynoise = sigma_clipped_stats(
            bkgsubimage, sigma=3.0)

    return skymean, skynoise


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
    """Report tractor results for flux and uncertainty in microJy; NaN if tractor results not reported.

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
