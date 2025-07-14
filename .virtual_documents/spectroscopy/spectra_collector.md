





# Uncomment the next line to install dependencies if needed.
# %pip install -r requirements_spectra_generator.txt


import os
import sys

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

sys.path.append('code_src/')
from data_structures_spec import MultiIndexDFObject
#from desi_functions import DESIBOSS_get_spec
from herschel_functions import Herschel_get_spec
from keck_functions import KeckDEIMOS_get_spec
from mast_functions import HST_get_spec, JWST_get_spec
from plot_functions import create_figures
from sample_selection import clean_sample
from sdss_functions import SDSS_get_spec
from spitzer_functions import SpitzerIRS_get_spec
from euclid_functions import euclid_get_spec





coords = []
labels = []

coords.append(SkyCoord("{} {}".format("09 54 49.40", "+09 16 15.9"), unit=(u.hourangle, u.deg)))
labels.append("NGC3049")

coords.append(SkyCoord("{} {}".format("12 45 17.44 ", "27 07 31.8"), unit=(u.hourangle, u.deg)))
labels.append("NGC4670")

coords.append(SkyCoord("{} {}".format("14 01 19.92", "−33 04 10.7"), unit=(u.hourangle, u.deg)))
labels.append("Tol_89")

coords.append(SkyCoord(233.73856, 23.50321, unit=u.deg))
labels.append("Arp220")

coords.append(SkyCoord(150.091, 2.2745833, unit=u.deg))
labels.append("COSMOS1")

coords.append(SkyCoord(150.1024475, 2.2815559, unit=u.deg))
labels.append("COSMOS2")

coords.append(SkyCoord("{} {}".format("150.000", "+2.00"), unit=(u.deg, u.deg)))
labels.append("COSMOS3")

coords.append(SkyCoord("{} {}".format("+53.15508", "-27.80178"), unit=(u.deg, u.deg)))
labels.append("JADESGS-z7-01-QU")

coords.append(SkyCoord("{} {}".format("+53.15398", "-27.80095"), unit=(u.deg, u.deg)))
labels.append("TestJWST")

coords.append(SkyCoord("{} {}".format("268.48058743", "64.78064676"), unit=(u.deg, u.deg)))
labels.append("TestEuclid")

coords.append(SkyCoord("{} {}".format("+150.33622", "+55.89878"), unit=(u.deg, u.deg)))
labels.append("Twin Quasar")

sample_table = clean_sample(coords, labels, precision=2.0 * u.arcsecond, verbose=1)





if not os.path.exists("./data"):
    os.mkdir("./data")
sample_table.write('data/input_sample.ecsv', format='ascii.ecsv', overwrite=True)





sample_table = Table.read('data/input_sample.ecsv', format='ascii.ecsv')





df_spec = MultiIndexDFObject()








%%time
# Get Keck Spectra (COSMOS only)
df_spec_DEIMOS = KeckDEIMOS_get_spec(sample_table=sample_table, search_radius_arcsec=1)
df_spec.append(df_spec_DEIMOS)


%%time
# Get Spitzer IRS Spectra
df_spec_IRS = SpitzerIRS_get_spec(sample_table, search_radius_arcsec=1, COMBINESPEC=False)
df_spec.append(df_spec_IRS)


%%time
# Get Euclid Spectra
df_spec_Euclid = euclid_get_spec(sample_table=sample_table, search_radius_arcsec=1)
df_spec.append(df_spec_Euclid)





%%time
# Get Spectra for HST
df_spec_HST = HST_get_spec(
    sample_table,
    search_radius_arcsec=0.5,
    datadir="./data/",
    verbose=False,
    delete_downloaded_data=True
)
df_spec.append(df_spec_HST)


%%time
# Get Spectra for JWST
df_jwst = JWST_get_spec(
    sample_table,
    search_radius_arcsec=0.5,
    verbose=False,
    max_spectra_per_source = 5
)
df_spec.append(df_jwst)





# Herschel PACS & SPIRE from ESA TAP using astroquery
# This search is fully functional, but is commented out because it takes
# ~4 hours to run to completion
herschel_radius = 1.1
herschel_download_directory = 'data/herschel'

# if not os.path.exists(herschel_download_directory):
#    os.makedirs(herschel_download_directory, exist_ok=True)
# df_spec_herschel =  Herschel_get_spec(sample_table, herschel_radius, herschel_download_directory, delete_downloaded_data=True)
# df_spec.append(df_spec_herschel)





%%time
# Get SDSS Spectra
df_spec_SDSS = SDSS_get_spec(sample_table, search_radius_arcsec=5, data_release=17)
df_spec.append(df_spec_SDSS)





#%%time
## Get DESI and BOSS spectra with SPARCL
#df_spec_DESIBOSS = DESIBOSS_get_spec(sample_table, search_radius_arcsec=5)
#df_spec.append(df_spec_DESIBOSS)





### Plotting ####
create_figures(df_spec=df_spec,
               bin_factor=1,
               show_nbr_figures=10,
               save_output=False,
               )
