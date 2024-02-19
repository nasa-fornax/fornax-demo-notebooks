import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, join, join_skycoord, unique
from astroquery.ipac.ned import Ned
from astroquery.sdss import SDSS
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier


#lamassa et al., 2015  1 source
def get_lamassa_sample(coords, labels, *, verbose=1):
    """Automatically grabs changing look AGN from LaMassa et al 2015 sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    lables : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    """
    lamassa_CLQ = Ned.query_refcode('2015ApJ...800..144L')
    lamassa_CLQ= lamassa_CLQ[0]  #dont know what those other targets are.
    coords.append(SkyCoord(lamassa_CLQ['RA'], lamassa_CLQ['DEC'], frame='icrs', unit='deg'))
    labels.append('LaMassa 15')
    if verbose:
        print('Changing Look AGN- Lamassa et al: ',len(labels))


#MacLeod et al., 2016
def get_macleod16_sample(coords, labels, *, verbose=1):
    """Automatically grabs changing look AGN from MacLeod et al., 2016 sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    lables : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    """
    macleod_CSQ = Table.read('https://academic.oup.com/mnras/article/457/1/389/989199', htmldict={'table_id': 5}, format='ascii.html')

    #get coords from "name" column for this
    for i in range(len(macleod_CSQ)):
        coord_str = macleod_CSQ['Name\n            .'][i]
        test_str = coord_str[0:2]+ " "+ coord_str[2:4]+ " " + coord_str[4:9] + " " + coord_str[9:12] + " " + coord_str[12:14]+ " " + coord_str[14:]
        coords.append(SkyCoord(test_str, unit=(u.hourangle, u.deg)))
        labels.append('MacLeod 16')
    if verbose:
        print('Changing Look AGN- MacLeod et al 2016: ',len(macleod_CSQ))


#Ruan et al., 2016  3 sources
def get_ruan_sample(coords, labels, *, verbose=1):
    """Automatically grabs changing look AGN from Ruan et al., 2016 sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    lables : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    """
    ruan_CSQ = Ned.query_refcode('2016ApJ...826..188R')

    ruan_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(ruan_CSQ['RA'], ruan_CSQ['DEC'])]
    ruan_labels = ['Ruan 16' for ra in ruan_CSQ['RA']]
    coords.extend(ruan_coords)
    labels.extend(ruan_labels)
    #one of these is a repeat of lamassa et al.
    if verbose:
        print('Changing Look AGN- Ruan et al 2016: ',len(ruan_CSQ['RA']))



#MacLeod et al., 2019 17 sources
def get_macleod19_sample(coords, labels, *, verbose=1):
    """Automatically grabs changing look AGN from MacLeod et al., 2019 sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    lables : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    """
    #try vizier
    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('2019ApJ...874....8M')
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    table2 = catalogs[0]  #more than one table

    #filter for the actual CLQs
    macleod19_CSQ = table2[(table2['CLQ_'] > 0) & (table2['Nsigma'] > 3)]

    macleod19_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(macleod19_CSQ['_RA'], macleod19_CSQ['_DE'])]
    macleod19_labels = ['MacLeod 19' for ra in macleod19_CSQ['_RA']]
    coords.extend(macleod19_coords)
    labels.extend(macleod19_labels)
    if verbose:
        print('Changing Look AGN- MacLeod et al 2017: ',len(macleod19_CSQ['_RA']))


#sheng et al., 2020
def get_sheng_sample(coords, labels, *, verbose=1):
    """Automatically grabs changing look AGN from sheng et al., 2020 sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    lables : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    """
    CLQ = Ned.query_refcode('2020ApJ...889...46S')
    sheng_CLQ = CLQ[[0,1,3]]#need the first 3 objects in their table,
    sheng_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(sheng_CLQ['RA'], sheng_CLQ['DEC'])]
    sheng_labels = ['Sheng 20' for ra in sheng_CLQ['RA']]

    coords.extend(sheng_coords)
    labels.extend(sheng_labels)
    if verbose:
        print('Changing Look AGN- Sheng et al 2020: ',len(sheng_CLQ['RA']))


#green et al., 2022  19 sources
def get_green_sample(coords, labels, *, verbose=1):
    """Automatically grabs changing look AGN from green et al., 2022 sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    lables : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    """

    #try vizier
    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/ApJ/933/180')
    catalogs = Vizier.get_catalogs(catalog_list.keys())
    table2 = catalogs[0]

    #go to pandas to manipulate the table
    green_CSQ = table2.to_pandas()

    #filter only those that are confirmed CLQ in the notes column
    green_CSQ = green_CSQ[green_CSQ['Notes'].str.contains("CLQ", na = False)]

    #pick out the coordinates from the 'SDSS' column
    coord_str = green_CSQ['SDSS']
    coord_str.astype('string')
    test_str = coord_str.str[1:3]+ " "+ coord_str.str[3:5]+ " " + coord_str.str[5:10] + " " + coord_str.str[10:13] + " " + coord_str.str[13:15]+ " " + coord_str.str[15:]
    green_labels = ['Green 22' for ra in green_CSQ['SDSS']]

    coords.extend(SkyCoord(test_str.values.tolist() , unit=(u.hourangle, u.deg)))#convert from pandas series to list as input to SkyCoord
    labels.extend(green_labels)
    if verbose:
        print('Changing Look AGN- Green et al 2022: ',len(green_labels))

#Lyu et al., 2021  lists a known sample of 68 sources to date!!!
def get_lyu_sample(coords, labels, *, verbose=1):
    """Automatically grabs changing look AGN from Lyu et al., 2021 sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    lables : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    """
    CLQ = Ned.query_refcode('2022ApJ...927..227L')
    lyu_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(CLQ['RA'], CLQ['DEC'])]
    lyu_labels = ['Lyu 22' for ra in CLQ['RA']]

    coords.extend(lyu_coords)
    labels.extend(lyu_labels)
    if verbose:
        print('Changing Look AGN- Lyu et al 2021: ',len(CLQ['RA']))


#Lopez-navas et al., 2022
def get_lopeznavas_sample(coords, labels, *, verbose=1):
    """Automatically grabs changing look AGN from Lopez-navas et al., 2022 sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    lables : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    """
    result_table = Simbad.query_bibobj('2022MNRAS.513L..57L')
    result_table = result_table[[0,1,2,3]]  #pick the correct sources by hand

    ln_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(result_table['RA'], result_table['DEC'])]
    ln_labels = ['Lopez-Navas 22' for ra in result_table['RA']]

    coords.extend(ln_coords)
    labels.extend(ln_labels)
    if verbose:
        print('Changing Look AGN- Lopez-navas et al 2022: ',len(result_table['RA']))


#Hon et al., 2022
def get_hon_sample(coords, labels, *, verbose=1):
    """Automatically grabs changing look AGN from Hon et al., 2022 sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    lables : list
    
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    """
    CLQ = Ned.query_refcode('2022MNRAS.511...54H')
    hon_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(CLQ['RA'], CLQ['DEC'])]
    hon_labels = ['Hon 22' for ra in CLQ['RA']]

    coords.extend(hon_coords)
    labels.extend(hon_labels)
    if verbose:
        print('Changing Look AGN- Hon et al 2022: ',len(CLQ['RA']))

#yang et al., 2018
def get_yang_sample(coords, labels, *, verbose=1):
    """Automatically grabs changing look AGN from yang et al., 2018 sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    lables : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    """
    CLQ = Ned.query_refcode('2018ApJ...862..109Y')
    yang_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(CLQ['RA'], CLQ['DEC'])]
    yang_labels = ['Yang 18' for ra in CLQ['RA']]

    coords.extend(yang_coords)
    labels.extend(yang_labels)
    if verbose:
        print('Changing Look AGN- Yang et al: ',len(CLQ['RA']))


#Here are additional CLAGN samples
#but not spectroscopically confirmed
#Sanchez Saez et al., 2021
def get_sanchezsaez_sample(coords, labels, *, verbose=1):
    """Automatically grabs changing look AGN from Sanchez Saez et al., 2021 sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    lables : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    """

    CSAGN = Ned.query_refcode('2021AJ....162..206S') # from Sanchez-Saez 2021

    ss_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(CSAGN['RA'], CSAGN['DEC'])]
    ss_labels = ['Sanchez-Saez 21' for ra in CSAGN['RA']]
    coords.extend(ss_coords)
    labels.extend(ss_labels)
    if verbose:
        print('Changing Look AGN- Sanchez et al: ',len(CSAGN['RA']))

#Graham et al., 2019
def get_graham_sample(coords, labels, *, verbose=1):
    """Automatically grabs changing look AGN from Graham et al., 2019  sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    lables : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    """
    #use astropy table to get larger sample that neither NED nor astropy can access.
    CSQ = Table.read('https://academic.oup.com/mnras/article/491/4/4925/5634279', htmldict={'table_id': 5}, format='ascii.html')

    #get coords from "name" column for this
    for i in range(len(CSQ)):
        coord_str = CSQ['Name\n            .'][i]
        test_str = coord_str[6:8]+ " "+ coord_str[8:10]+ " " + coord_str[10:14] + " " + coord_str[14:17] + " " + coord_str[17:19]+ " " + coord_str[19:]
        coords.append(SkyCoord(test_str, unit=(u.hourangle, u.deg)))
        labels.append('Graham 19')
    if verbose:
        print('Changing Look AGN- Graham et al: ',len(CSQ))


#SDSS QSO sample of any desired number
#These are "normal" QSOs to use in the classifier
def get_sdss_sample(coords, labels, *, num=10, zmin=0, zmax=10, verbose=1):
    """Automatically grabs SDSS quasar sample.

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    lables : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    num : int < 500K
        How many quasars for which to return the coords and labels
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    """
    # Define the query
    query = "SELECT TOP " + str(num) + " specObjID, ra, dec, z FROM SpecObj \
    WHERE ( z > " + str(zmin) + "AND z < " + str(zmax) + " AND class='QSO' AND zWARNING=0 )"

    #making up redshift range here, but should look at redshift distribution of the CLQ

    #use astroquery to return an astropy table of results
    if num>0:
        res = SDSS.query_sql(query, data_release = 16)

        SDSS_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(res['ra'], res['dec'])]
        SDSS_labels = ['SDSS' for ra in res['ra']]

        coords.extend(SDSS_coords)
        labels.extend(SDSS_labels)

    if verbose:
        print('SDSS Quasar: '+str(num))


def get_paper_sample(coords, labels, *, paper_link="2019A&A...627A..33D", label="Cicco19", verbose=1):
    """Looks for RA,DEC in a paper using Ned query and returns list of coords and lables

    Parameters
    ----------
    coords : list
        list of Astropy SkyCoords derived from literature sources, shared amongst functions
    lables : list
        List of the first author name and publication year for tracking the sources, shared amongst functions
    verbose : int, optional
        Print out the length of the sample derived from this literature source
    """
    paper = Ned.query_refcode(paper_link)

    paper_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(paper['RA'], paper['DEC'])]
    paper_labels = [label for ra in paper['RA']]
    coords.extend(paper_coords)
    labels.extend(paper_labels)
    if verbose:
        print("number of sources added from "+str(label)+" :"+str(len(paper_coords)))


def clean_sample(coords_list, labels_list, verbose=1):
    """Makes a unique sample of skycoords and labels with no repeats. Attaches an object ID to the coords.

    Parameters
    ----------
    coords_list : list
        list of Astropy SkyCoords derived from literature sources
    labels_list : list
        List of the first author name and publication year for tracking the sources
    verbose : int, optional
        Print out the length of the sample after applying this function

    Returns
    -------
    sample_table : `~astropy.table.Table`
        sample cleaned of duplicates, with an object ID attached.
    """

    sample_table = Table([coords_list, labels_list], names=['coord', 'label'])

    # now join the table with itself within a defined radius.
    # We keep one set of original column names to avoid later need for renaming
    tjoin = join(sample_table, sample_table, keys='coord',
                 join_funcs={'coord': join_skycoord(0.005 * u.deg)},
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


def nonunique_sample(skycoordslist, labels, verbose=1):
    """Changes the structure of the coordinates to a list of SkyCoords and a list of labels.

    Parameters
    ----------
    skycoordslist : list
        list of Astropy SkyCoords derived from literature sources
    lables : list
        List of the first author name and publication year for tracking the sources
    verbose : int, optional
        Print out the length of the sample after applying this function

    Returns
    -------
    coords_list : list of tuples
        coords input cleaned of duplicates, with an object ID attached. Tuples contain (objectid, skycoords).
    labels_list : list
        labels associated with coords_list
    """
    #first turn the skycoord list into a table to be able to access table functions in astropy
    t = Table([skycoordslist, labels, np.arange(0, len(skycoordslist), 1)], names=['sc', 'label', 'idx'])
    uniquerows = t#table.unique(tjoin, keys = 'sc_id')
    raw_coords_list = list(t['sc'])
    labels_list = list(t['label'])
    if verbose:
        print('without duplicates removal, sample size: '+str(len(raw_coords_list)))
    coords_list = list(enumerate(raw_coords_list))  # list of tuples (objectid, skycoords)
    return coords_list, labels_list
