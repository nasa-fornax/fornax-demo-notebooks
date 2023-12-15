import numpy as np
import pandas as pd
import requests
from astropy.io import ascii
from astropy.table import Table
from tqdm import tqdm

from data_structures import MultiIndexDFObject

# code partially taken from https://ps1images.stsci.edu/ps1_dr2_api.html
def ps1cone(ra,dec,radius,table="mean",release="dr1",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    """Do a cone search of the PS1 catalog
    
    Parameters
    ----------
    ra: float (degrees) 
        J2000 Right Ascension
    dec: float (degrees) 
        J2000 Declination
    radius: float (degrees) 
        Search radius (<= 0.5 degrees)
    table: string
        mean, stack, or detection
    release: string
        dr1 or dr2
    format: string
        csv, votable, json
    columns: list of strings 
        list of column names to include (None means use defaults)
    baseurl: stirng
        base URL for the request
    verbose: int
        print info about request
    **kw: 
        other parameters (e.g., 'nDetections.min':2)
    
    Returns
    -------
    cone search results
    """
    
    data = kw.copy()
    data['ra'] = ra
    data['dec'] = dec
    data['radius'] = radius
    return ps1search(table=table,release=release,format=format,columns=columns,
                    baseurl=baseurl, verbose=verbose, **data)


def ps1search(table="mean",release="dr1",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    """Do a general search of the PS1 catalog (possibly without ra/dec/radius)
    
    Parameters
    ----------
    table: string 
        mean, stack, or detection
    release: string 
        dr1 or dr2
    format: string
        csv, votable, json
    columns: list of strings
        list of column names to include (None means use defaults)
    baseurl: string
        base URL for the request
    verbose: int
        print info about request
    **kw: 
        other parameters (e.g., 'nDetections.min':2).  Note this is required!
    
    Returns
    -------
    search results
    """
    
    data = kw.copy()
    if not data:
        raise ValueError("You must specify some parameters for search")
    checklegal(table,release)
    if format not in ("csv","votable","json"):
        raise ValueError("Bad value for format")
    url = f"{baseurl}/{release}/{table}.{format}"
    if columns:
        # check that column values are legal
        # create a dictionary to speed this up
        dcols = {}
        for col in ps1metadata(table,release)['name']:
            dcols[col.lower()] = 1
        badcols = []
        for col in columns:
            if col.lower().strip() not in dcols:
                badcols.append(col)
        if badcols:
            raise ValueError('Some columns not found in table: {}'.format(', '.join(badcols)))
        # two different ways to specify a list of column values in the API
        # data['columns'] = columns
        data['columns'] = '[{}]'.format(','.join(columns))

    # either get or post works
#    r = requests.post(url, data=data)
    r = requests.get(url, params=data)

    if verbose:
        print(r.url)
    r.raise_for_status()
    if format == "json":
        return r.json()
    else:
        return r.text


def checklegal(table,release):
    """Checks if this combination of table and release is acceptable
    Raises a ValueError exception if there is problem
    
    Parameters
    ----------
    table: string
        mean, stack, or detection
    release: string
        dr1 or dr2
    """
    
    releaselist = ("dr1", "dr2")
    if release not in ("dr1","dr2"):
        raise ValueError("Bad value for release (must be one of {})".format(', '.join(releaselist)))
    if release=="dr1":
        tablelist = ("mean", "stack")
    else:
        tablelist = ("mean", "stack", "detection")
    if table not in tablelist:
        raise ValueError("Bad value for table (for {} must be one of {})".format(release, ", ".join(tablelist)))


def ps1metadata(table="mean",release="dr1",
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"):
    """Return metadata for the specified catalog and table
    
    Parameters
    ----------
    table: string 
        mean, stack, or detection
    release: string 
        dr1 or dr2
    baseurl: string 
        base URL for the request
    
    Returns
    -------
    astropy table with columns name, type, description
    """
    
    checklegal(table,release)
    url = f"{baseurl}/{release}/{table}/metadata"
    r = requests.get(url)
    r.raise_for_status()
    v = r.json()
    # convert to astropy table
    tab = Table(rows=[(x['name'],x['type'],x['description']) for x in v],
               names=('name','type','description'))
    return tab


def addfilter(dtab):
    """Add filter name as column in detection table by translating filterID
    
    This modifies the table in place.  If the 'filter' column already exists,
    the table is returned unchanged.
    
    Parameters
    ----------
    dtab: table
        detection table
        
    Returns
    -------
    dtab: table
    """
    if 'filter' not in dtab.colnames:
        # the filterID value goes from 1 to 5 for grizy
        #id2filter = np.array(list('grizy'))
        id2filter = np.array(['panstarrs g','panstarrs r','panstarrs i','panstarrs z','panstarrs y'])
        dtab['filter'] = id2filter[(dtab['filterID']-1).data]
    return dtab

def improve_filter_format(tab):
    """Add filter string to column name
    Parameters
    ----------
    tab:table
    
    Returns
    -------
    tab: table
    """
    for filter in 'grizy':
        col = filter+'MeanPSFMag'
        tab[col].format = ".4f"
        tab[col][tab[col] == -999.0] = np.nan
            
    return(tab)

def search_lightcurve(objid):
    """setup to pull light curve info
    
    Parameters
    ----------
    objid: string
    
    Returns
    -------
    dresults: search results
    
    """
    dconstraints = {'objID': objid}
    dcolumns = ("""objID,detectID,filterID,obsTime,ra,dec,psfFlux,psfFluxErr,psfMajorFWHM,psfMinorFWHM,
            psfQfPerfect,apFlux,apFluxErr,infoFlag,infoFlag2,infoFlag3""").split(',')
    # strip blanks and weed out blank and commented-out values
    dcolumns = [x.strip() for x in dcolumns]
    dcolumns = [x for x in dcolumns if x and not x.startswith('#')]


    #get the actual detections and light curve info for this target
    dresults = ps1search(table='detection',release='dr2',columns=dcolumns,**dconstraints)
    
    return(dresults)


#Do a panstarrs search
def Panstarrs_get_lightcurves(sample_table, radius):
    """Searches panstarrs for light curves from a list of input coordinates.  This is the MAIN function.
    
    Parameters
    ----------
    sample_table : `~astropy.table.Table`
        Table with the coordinates and journal reference labels of the sources
    radius : float
        search radius, how far from the source should the archives return results

    Returns
    -------
    df_lc : MultiIndexDFObject
        the main data structure to store all light curves
    """
        
    df_lc = MultiIndexDFObject()
    
    #for all objects in our catalog
    for row in tqdm(sample_table):
        #doesn't take SkyCoord, convert to floats
        ra = row["coord"].ra.deg
        dec = row["coord"].dec.deg
        lab = row["label"]
        objectid = row["objectid"]

        #sometimes there isn't actually a light curve for the target???
        try:
            #see if there is an object in panSTARRS at this location
            results = ps1cone(ra,dec,radius,release='dr2')
            tab = ascii.read(results)
    
            # improve the format of the table
            tab = improve_filter_format(tab)
        
            #in case there is more than one object within 1 arcsec, sort them by match distance
            tab.sort('distance')
    
            #if there is an object at that location
            if len(tab) > 0:   
                #got a live one
                #print( 'for object', ccount + 1, 'there is ',len(tab), 'match in panSTARRS', tab['objID'])

                #take the closest match as the best match
                objid = tab['objID'][0]
        
                #get the actual detections and light curve info for this target
                dresults = search_lightcurve(objid)
        
            ascii.read(dresults)
       
    
        #fix the column names to include filter names
    
            dtab = addfilter(ascii.read(dresults))
    
            dtab.sort('obsTime')

            #here is the light curve mixed from all 5 bands
            t_panstarrs = dtab['obsTime']
            flux_panstarrs = dtab['psfFlux']*1E3  # in mJy
            err_panstarrs = dtab['psfFluxErr'] *1E3
            filtername = dtab['filter']
            
            #put this single object light curves into a pandas multiindex dataframe
            dfsingle = pd.DataFrame(dict(flux=flux_panstarrs, err=err_panstarrs, time=t_panstarrs, objectid=objectid, band=filtername, label=lab)).set_index(["objectid","label", "band", "time"])
            #then concatenate each individual df together
            df_lc.append(dfsingle)
        except FileNotFoundError:
            #print("There is no light curve")
            pass
            
    return(df_lc)
