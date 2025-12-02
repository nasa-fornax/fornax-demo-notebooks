import numpy as np

def parse_polygon(s_region):
    """
    Extract coordinates from an s_region string.

    Parameters
    ----------
    s_region : string - s_region from cross-mission archive
               As written, this s_region must be a single polygon,
               which is the case for all JWST and SOFIA observations,
               but not all missions in the Fornax archives.

    Returns
    ----------
    coord_array : np.ndarray
        Parsed representation of the s_region. Last vertex is identical
        to first vertex.
    """

    if 'POLYGON ICRS' in s_region:
        coords = list(map(float, s_region.replace("POLYGON ICRS", "").strip().split()))
    elif 'POLYGON' in s_region:
        coords = list(map(float, s_region.replace("POLYGON", "").strip().split()))

    # Check if the polygon is closed, with the last vertex identical to the first vertex.
    # If not, append a copy of the first vertex to the end of the list.
    if (coords[-2], coords[-1]) != (coords[0], coords[1]):
        coords.append(coords[0])
        coords.append(coords[1])

    # Create a numpy array listing a coordinate tuple for each vertex
    coord_array = np.array([(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)])

    return(coord_array)