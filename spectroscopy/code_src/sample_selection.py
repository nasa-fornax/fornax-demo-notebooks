from astropy.table import Table, join, join_skycoord, unique


def clean_sample(coords_list, labels_list, precision, verbose=1):
    """
    Make a unique sample of sky coordinates and labels with no repeats.
    Attaches an object ID to the coordinates.

    Parameters
    ----------
    coords_list : list
        List of Astropy SkyCoord objects derived from literature sources.
    labels_list : list
        List of the first author name and publication year for tracking the sources.
    precision : float
        Precision of matching/removing duplicates. For example, 0.5 * u.arcsecond.
    verbose : int, optional
        Print out the length of the sample after applying this function.

    Returns
    -------
    astropy.table.Table
        Sample cleaned of duplicates, with an object ID attached.
    """

    sample_table = Table([coords_list, labels_list], names=['coord', 'label'])

    # now join the table with itself within a defined radius.
    # We keep one set of original column names to avoid later need for renaming
    tjoin = join(sample_table, sample_table, keys='coord',
                 join_funcs={'coord': join_skycoord(precision)},
                 uniq_col_name='{col_name}{table_name}', table_names=['', '_2'])

    # this join will return 4 entries for each redundant coordinate:
    # 1 for the match with itself and 1 for the match with the similar
    # enough coord target then the same thing again for the match, but all of
    # these 4 will have the same id in the new 'coords_id' column

    # keep only those entries in the resulting table which are unique
    uniqued_table = unique(tjoin, keys='coord_id')['coord_id', 'coord', 'label']
    uniqued_table.rename_column('coord_id', 'objectid')

    if verbose:
        print(f'after duplicates removal, sample size: {len(uniqued_table)}')

    return uniqued_table
