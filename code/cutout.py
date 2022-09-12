# function to extract cutout image
import math
from astropy.nddata import Cutout2D
import astropy.io.fits as fits
from astropy.wcs import WCS

from exceptions import CutoutError


def extract_pair(ra, dec, *, fitsfiles, cutout_width, mosaic_pix_scale):
    """Load the science and background images from `fitsfiles` and extract cutouts from both.

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
    imgfile, skybkgfile = fitsfiles

    # extract science image cutout
    subimage, x1, y1, subimage_wcs = extract(
        ra, dec, fitsfile=imgfile, cutout_width=cutout_width, mosaic_pix_scale=mosaic_pix_scale
    )

    # extract sky background cutout
    # if input is same as above, don't redo the cutout
    if skybkgfile == imgfile:
        bkgsubimage = subimage
    else:
        bkgsubimage, _, _, _ = extract(
            ra, dec, fitsfile=skybkgfile, cutout_width=cutout_width, mosaic_pix_scale=mosaic_pix_scale
        )

    return subimage, bkgsubimage, x1, y1, subimage_wcs


def extract(ra, dec, *, fitsfile, cutout_width, mosaic_pix_scale):
    '''Extract an image cutout from fitsfile.

    Raise a `CutoutError` if the cutout cannot be extracted.

    Returns:
    --------
    cutout:
        Cropped image data.
    x, y:
        Pixel coordinates of the target in the cutout.
    cutout_wcs:
        `WCS` for the cutout.
    '''
    hdulist = fits.open(fitsfile)[0]
    wcs_info = WCS(hdulist)

    try:

        # convert ra and dec into x, y
        x0, y0 = wcs_info.all_world2pix(ra, dec, 1)
        position = (x0, y0)
        #subimage = hdulist[0].data

        # make size array in pixels
        # how many pixels are in cutout_width
        size = (cutout_width / mosaic_pix_scale)
        size = int(math.ceil(size))  # round up the nearest integer

        # make the cutout
        cutout = Cutout2D(hdulist.data, position, size,
                          copy=True, mode="trim", wcs=wcs_info)
        subimage = cutout.data.copy()
        subimage_wcs = cutout.wcs.copy()

        # now need to set the values of x1, y1 at the location of the target *in the cutout*
        x1, y1 = subimage_wcs.all_world2pix(ra, dec, 1)
        #print('x1, y1', x1, y1)

        # hdulist.close()

    except Exception as e:
        raise CutoutError("Cutout could not be extracted") from e

    return subimage.data, x1, y1, subimage_wcs
