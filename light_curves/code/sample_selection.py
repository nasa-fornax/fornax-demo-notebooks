from astroquery.ipac.ned import Ned
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.sdss import SDSS

from astropy.table import unique, Table, join, join_skycoord
from astropy.coordinates import SkyCoord, name_resolve
import astropy.units as u
from astropy import table

import numpy as np

#lamassa et al., 2015  1 source

def get_lamassa_sample(coords, labels):
    lamassa_CLQ = Ned.query_refcode('2015ApJ...800..144L')
    lamassa_CLQ= lamassa_CLQ[0]  #dont know what those other targets are.
    coords.append(SkyCoord(lamassa_CLQ['RA'], lamassa_CLQ['DEC'], frame='icrs', unit='deg'))
    labels.append('LaMassa 15')

#MacLeod et al., 2016
def get_macleod16_sample(coords, labels):
    macleod_CSQ = Table.read('https://academic.oup.com/mnras/article/457/1/389/989199', htmldict={'table_id': 5}, format='ascii.html')

    #get coords from "name" column for this
    for i in range(len(macleod_CSQ)):
        coord_str = macleod_CSQ['Name\n            .'][i]
        test_str = coord_str[0:2]+ " "+ coord_str[2:4]+ " " + coord_str[4:9] + " " + coord_str[9:12] + " " + coord_str[12:14]+ " " + coord_str[14:]
        coords.append(SkyCoord(test_str, unit=(u.hourangle, u.deg)))
        labels.append('MacLeod 16')
        
#Ruan et al., 2016  3 sources
def get_ruan_sample(coords, labels):
    ruan_CSQ = Ned.query_refcode('2016ApJ...826..188R')

    ruan_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(ruan_CSQ['RA'], ruan_CSQ['DEC'])]
    ruan_labels = ['Ruan 16' for ra in ruan_CSQ['RA']]
    coords.extend(ruan_coords)
    labels.extend(ruan_labels)
    #one of these is a repeat of lamassa et al.


#MacLeod et al., 2019 17 sources
def get_macleod19_sample(coords, labels):
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

#sheng et al., 2020
def get_sheng_sample(coords, labels):
    CLQ = Ned.query_refcode('2020ApJ...889...46S') 
    sheng_CLQ = CLQ[[0,1,3]]#need the first 3 objects in their table, 
    sheng_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(sheng_CLQ['RA'], sheng_CLQ['DEC'])]
    sheng_labels = ['Sheng 20' for ra in sheng_CLQ['RA']]

    coords.extend(sheng_coords)
    labels.extend(sheng_labels)

#green et al., 2022  19 sources
def get_green_sample(coords, labels):
    green_CSQ = Table.read('https://cfn-live-content-bucket-iop-org.s3.amazonaws.com/journals/0004-637X/933/2/180/revision2/apjac743ft2_mrt.txt?AWSAccessKeyId=AKIAYDKQL6LTV7YY2HIK&Expires=1678235090&Signature=sRu3zJOwgegGTF2iCZvkRxwjT44%3D', format='ascii')
    green_CSQ = green_CSQ.to_pandas()

    #filter only those that are confirmed CLQ in the notes column
    green_CSQ = green_CSQ[green_CSQ['Notes'].str.contains("CLQ", na = False)]

    #pick out the coordinates from the 'SDSS' column
    coord_str = green_CSQ['SDSS']
    coord_str.astype('string')
    test_str = coord_str.str[1:3]+ " "+ coord_str.str[3:5]+ " " + coord_str.str[5:10] + " " + coord_str.str[10:13] + " " + coord_str.str[13:15]+ " " + coord_str.str[15:]
    green_labels = ['Green 22' for ra in green_CSQ['Notes']]

    coords.extend(SkyCoord(test_str.values.tolist() , unit=(u.hourangle, u.deg)))#convert from pandas series to list as input to SkyCoord
    labels.extend(green_labels)

#Lyu et al., 2021  lists a known sample of 68 sources to date!!!
def get_lyu_sample(coords, labels):
    CLQ = Ned.query_refcode('2022ApJ...927..227L') 
    lyu_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(CLQ['RA'], CLQ['DEC'])]
    lyu_labels = ['Lyu 21' for ra in CLQ['RA']]

    coords.extend(lyu_coords)
    labels.extend(lyu_labels)

#Lopez-navas et al., 2022
def get_lopeznavas_sample(coords, labels):
    result_table = Simbad.query_bibobj('2022MNRAS.513L..57L')
    result_table = result_table[[0,1,2,3]]  #pick the correct sources by hand

    ln_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(result_table['RA'], result_table['DEC'])]
    ln_labels = ['Lopez-Navas 22' for ra in result_table['RA']]

    coords.extend(ln_coords)
    labels.extend(ln_labels)

#Hon et al., 2022
def get_hon_sample(coords, labels):
    CLQ = Ned.query_refcode('2022MNRAS.511...54H') 
    hon_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(CLQ['RA'], CLQ['DEC'])]
    hon_labels = ['Hon 22' for ra in CLQ['RA']]

    coords.extend(hon_coords)
    labels.extend(hon_labels)

def get_yang_sample(coords, labels):
    CLQ = Ned.query_refcode('2018ApJ...862..109Y')
    yang_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(CLQ['RA'], CLQ['DEC'])]
    yang_labels = ['Yang 18' for ra in CLQ['RA']]

    coords.extend(yang_coords)
    labels.extend(yang_labels)
                  
#Here are additional CLAGN samples 
#but not spectroscopically confirmed
def get_sanchezsaez_sample(coords, labels):
                  
    CSAGN = Ned.query_refcode('2021AJ....162..206S') # from Sanchez-Saez 2021

    ss_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(CSAGN['RA'], CSAGN['DEC'])]
    ss_labels = ['Sanchez-Saez 21' for ra in CSAGN['RA']]
    coords.extend(ss_coords)
    labels.extend(ss_labels)
                  
def get_graham_sample(coords, labels):
    #use astropy table to get larger sample that neither NED nor astropy can access.
    CSQ = Table.read('https://academic.oup.com/mnras/article/491/4/4925/5634279', htmldict={'table_id': 5}, format='ascii.html')

    #get coords from "name" column for this
    for i in range(len(CSQ)):
        coord_str = CSQ['Name\n            .'][i]
        test_str = coord_str[6:8]+ " "+ coord_str[8:10]+ " " + coord_str[10:14] + " " + coord_str[14:17] + " " + coord_str[17:19]+ " " + coord_str[19:]
        coords.append(SkyCoord(test_str, unit=(u.hourangle, u.deg)))
        labels.append('Graham 19')
        
        
        
#and now a function to remove duplicates from the coordinate list 
def remove_duplicate_coords(skycoordslist, labels):
    #first turn the skycoord list into a table to be able to access table functions in astropy
    t = Table([skycoordslist, labels, np.arange(0, len(skycoordslist), 1)], names=['sc', 'label', 'idx'])

    #now join the table with itself within a defined radius
    tjoin = join(t, t, keys='sc', join_funcs={'sc': join_skycoord(0.005 * u.deg)})
    #this join will return 4 entries for each redundant coordinate = 
    #1 for the match with itself and 1 for the match with the similar enough coord target then the same thing again for the match
    #but all of these 4 will have the same id in the new 'sc_id' column made by the join function

    #keep only those entries in the resulting table which are unique 
    uniquerows = table.unique(tjoin, keys = 'sc_id')

    #turn back into a list
    sample_CLQ = uniquerows['sc_1']
    labels_CLQ = uniquerows['label_1']
    return(sample_CLQ, labels_CLQ)