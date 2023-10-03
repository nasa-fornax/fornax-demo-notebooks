from astropy.coordinates import SkyCoord

from alerce.core import Alerce

def TDE_id2coord(object_ids, coords, labels, verbose=1):
    """ To find and append coordinates of objects with only ZTF obj name

    Parameters
    ----------
    object_ids: list of strings
        eg., [ "ZTF18accqogs", "ZTF19aakyhxi", "ZTF19abyylzv", "ZTF19acyfpno"]
    coords : list of astropy skycoords
        the coordinates of the targets for which a user wants light curves
    labels: list of strings
        journal articles associated with the target coordinates
    verbose: int
        print out debugging info (1) or not(0)
    """

    alerce = Alerce()
    objects = alerce.query_objects(oid=object_ids, format="pandas")
    tde_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(objects['meanra'], objects['meandec'])]
    tde_labels = ['ZTF-Objname' for _ in objects['meanra']]
    coords.extend(tde_coords)
    labels.extend(tde_labels)
    if verbose:
        print('number of ztf coords added by Objectname:', len(objects['meanra']))