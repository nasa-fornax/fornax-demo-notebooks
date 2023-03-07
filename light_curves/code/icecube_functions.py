## Functions related to IceCube matching
from astropy.table import Table, vstack
from astropy.io import ascii
import astropy.units as u
import os

def get_icecube_catalog(path):
    '''
    Creates the combined IceCube catalog based on the yearly catalog
    
    INPUT:
        - path: path to the directory where the cataogs are saved. Must be the main directory,
                such as /my/path/icecube_10year_ps/ which includes the "events" directory.
    
    OUTPUT:
        - returns combined catalog with columns ["mjd","energy_logGeV","AngErr","ra","dec","az","zen"]
        - returns event file names (for convenience)
    '''
    
    event_names = ["IC40_exp.csv",
                    "IC59_exp.csv",
                    "IC79_exp.csv",
                    "IC86_III_exp.csv",
                    "IC86_II_exp.csv",
                    "IC86_IV_exp.csv",
                    "IC86_I_exp.csv",
                    "IC86_VII_exp.csv",
                    "IC86_VI_exp.csv",
                    "IC86_V_exp.csv"
                  ]
    
    EVENTS = Table(names=["mjd","energy_logGeV","AngErr","ra","dec","az","zen"] ,
                   units=[u.d , u.electronvolt*1e9 , u.degree , u.degree , u.degree , u.degree , u.degree ])
    for event_name in event_names:
        print("Loading: ", event_name)
        tmp = ascii.read(os.path.join(path , "events" , event_name))
        tmp.rename_columns(names=tmp.keys() , new_names=EVENTS.keys() )

        EVENTS = vstack([EVENTS , tmp])
    print("done")
    return(EVENTS , event_names)