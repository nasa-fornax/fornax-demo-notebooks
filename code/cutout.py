# function to extract cutout image
import math
from astropy.nddata import Cutout2D
import astropy.io.fits as fits
from astropy.wcs import WCS

from exceptions import CutoutError


def extract_pair(ra, dec, *, img_pair, cutout_width, mosaic_pix_scale):
    """Extract cutouts from the science and background images in `img_pair`.

    If `img_pair` contains only one element, extract the cutout once and return it for both.

    Parameters:
    -----------
    ra, dec : float
        celestial coordinates for measuring photometry
    img_pair : tuple
        Pair of images for science and background respectively.
        If the tuple only contains one element, the same image will be used for both cutouts.
        A tuple element can be a fits.ImageHDU or the path to a FITS file as a string.
    cutout_width : int
        width of desired cutout in arcseconds
    mosaic_pix_scale : float
        Pixel scale of the image

    Returns:
    --------
    cutout:
        Cropped science image data.
    bkg_cutout:
        Cropped background image data.
    x, y:
        Pixel coordinates of the target in the science cutout.
    cutout_wcs:
        `WCS` for the science cutout.
    """
    # extract science image cutout
    img = img_pair[0]
    subimage, x1, y1, subimage_wcs = extract(
        ra, dec, hdu=img, cutout_width=cutout_width, mosaic_pix_scale=mosaic_pix_scale
    )

    # extract sky background cutout
    # if there's only 1 HDU in the "pair", it doubles for the background
    if len(img_pair) == 1:
        bkgsubimage = subimage
    else:
        img = img_pair[1]
        bkgsubimage, _, _, _ = extract(
            ra, dec, hdu=img, cutout_width=cutout_width, mosaic_pix_scale=mosaic_pix_scale
        )

    return subimage, bkgsubimage, x1, y1, subimage_wcs


def extract(ra, dec, *, hdu, cutout_width, mosaic_pix_scale):
    '''Extract an image cutout from `hdu` at `ra` and `dec`.

    Parameters:
    -----------
    ra, dec : float
        celestial coordinates for measuring photometry
    hdu : fits.ImageHDU or str
        Image data to extract the cutout from.
        If this is a `str`, it should be the path to a FITS file -- the primary HDU
        will be loaded and used.
    cutout_width : int
        width of desired cutout in arcseconds
    mosaic_pix_scale : float
        Pixel scale of the image

    Returns:
    --------
    cutout:
        Cropped image data.
    x, y:
        Pixel coordinates of the target in the cutout.
    cutout_wcs:
        `WCS` for the cutout.

    Raises:
    -------
    CutoutError : If the cutout cannot be extracted.
    '''
    if isinstance(hdu, str):
        # hdu is a file path. load the primary HDU
        hdu = fits.open(hdu)[0]
    wcs_info = WCS(hdu)

    try:
        # convert ra and dec into x, y
        x0, y0 = wcs_info.all_world2pix(ra, dec, 1)
        position = (x0, y0)

        # make size array in pixels
        # how many pixels are in cutout_width
        size = (cutout_width / mosaic_pix_scale)
        size = int(math.ceil(size))  # round up the nearest integer

        # make the cutout
        cutout = Cutout2D(hdu.data, position, size,
                          copy=True, mode="trim", wcs=wcs_info)
        subimage = cutout.data.copy()
        subimage_wcs = cutout.wcs.copy()

        # now need to set the values of x1, y1 at the location of the target *in the cutout*
        x1, y1 = subimage_wcs.all_world2pix(ra, dec, 1)
        #print('x1, y1', x1, y1)

    except Exception as e:
        raise CutoutError("Cutout could not be extracted") from e

    return subimage.data, x1, y1, subimage_wcs
