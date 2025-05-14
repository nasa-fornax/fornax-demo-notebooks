---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: python3
  display_name: notebook
  language: python
---

### Add Rubin data
warning you need a Rubin science platform (RSP) login to access both the data previews and real data

Instructions:

1. If you need a RSP login, follow these [instructions](https://rsp.lsst.io/guides/getting-started/get-an-account.html) to get that setup. 

2. Setup an RSP token using these [instructions](code_src/rsp_token_instructions.txt).  A token is a way of saving your access credentials in your home directory which lets RSP know that you are allowed to access data through this code cell instead of their web interface. 

# to do
- figure out pyvo and tap to do a query of data preview for light curves of our targets
- is there some better way to write the queries that we could introduce to people?

```{code-cell} ipython3
!pip install -r requirements_light_curve_generator.txt
```

```{code-cell} ipython3
import multiprocessing as mp
import sys
import time

import astropy.units as u
import pandas as pd
from astropy.table import Table

import pyvo
import os, getpass

# local code imports
sys.path.append('code_src/')
from data_structures import MultiIndexDFObject
```

```{code-cell} ipython3
#need to query rubin for light curve data based on position.
```

```{code-cell} ipython3
#get the token
my_username = getpass.getuser()
token_filename = os.getenv('HOME')+'/.rsp-tap.token'

#if this line does not return an error than the token file exists
assert os.path.exists(token_filename)

with open(token_filename, 'r') as f:
    token = f.readline().strip()
    
#if this line does not return an error than the token has been successfully read
assert token is not None

#get credentials
cred = pyvo.auth.CredentialStore()
cred.set_password("x-oauth-basic", token)
session = cred.get("ivo://ivoa.net/sso#BasicAA")

#Instantiate TAPService
rsp_tap_url = 'https://data.lsst.cloud/api/tap'
rsp_tap = pyvo.dal.TAPService(rsp_tap_url, session = session)

assert rsp_tap is not None
assert rsp_tap.baseurl == rsp_tap_url

#The only way to test the token has authorized access to the RSP is to try to run a query.
#but no error messages up to this point is a good sign
```

```{code-cell} ipython3
#general plan for the syntax

# query = "SELECT * FROM tap_schema.schemas WHERE <constraints>"
# results = rsp_tap.run_sync(query).to_table()
```

```{code-cell} ipython3
# can I run the example from RSP?
use_center_coords = "62, -37"
my_adql_query = "SELECT TOP 10 "+ \
                "coord_ra, coord_dec, detect_isPrimary, " + \
                "r_calibFlux, r_cModelFlux, r_extendedness " + \
                "FROM dp02_dc2_catalogs.Object " + \
                "WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), " + \
                "CIRCLE('ICRS', " + use_center_coords + ", 0.01)) = 1 "
results = rsp_tap.run_sync(my_adql_query).to_table()
```

```{code-cell} ipython3
results
```

```{code-cell} ipython3
#following tutorial for light curve
#first find the objectid corresponding to the position
ra_known_rrl = 62.1479031
dec_known_rrl = -35.799138
query = "SELECT TOP 10 "\
        "coord_ra, coord_dec, objectId "\
        "FROM dp02_dc2_catalogs.Object "\
        "WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), "\
        "CIRCLE('ICRS'," + str(ra_known_rrl) + ", "\
        + str(dec_known_rrl) + ", 0.001)) = 1 "\
        "AND detect_isPrimary = 1"
objs = rsp_tap.run_sync(query).to_table()
```

```{code-cell} ipython3
objs
```

```{code-cell} ipython3
#extract measurements from the forced source table
sel_objid = objs[0]['objectId']  #grab the objectID

#setup query
#join is required because that second catalog has the times 

query = "SELECT src.band, src.ccdVisitId, src.coord_ra, src.coord_dec, "\
        "src.objectId, src.psfFlux, src.psfFluxErr, "\
        "scisql_nanojanskyToAbMag(psfFlux) as psfMag, "\
        "visinfo.ccdVisitId, visinfo.band, "\
        "visinfo.expMidptMJD, visinfo.zeroPoint "\
        "FROM dp02_dc2_catalogs.ForcedSource as src "\
        "JOIN dp02_dc2_catalogs.CcdVisit as visinfo "\
        "ON visinfo.ccdVisitId = src.ccdVisitId "\
        "WHERE src.objectId = "+str(sel_objid)+" "
srcs = rsp_tap.run_sync(query).to_table()
srcs
```

```{code-cell} ipython3
#now try with a skycoord list
savename_sample = f"output/yang_CLAGN_sample.ecsv"
sample_table = Table.read(savename_sample, format='ascii.ecsv')
```

```{code-cell} ipython3
sample_table
```

```{code-cell} ipython3
#table upload not currently supported by RSP
#trying to OR-chain
clauses = [
    f"CONTAINS(POINT('ICRS', coord_ra, coord_dec), "
    f"CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, 0.001))=1"
    for ra, dec in zip(sample_table['coord'].ra.deg, sample_table['coord'].dec.deg)
]
adql = f"""
SELECT
coord_ra, coord_dec, objectId
FROM dp02_dc2_catalogs.Object
WHERE ({' OR '.join(clauses)})
  AND detect_isPrimary=1
"""
results = rsp_tap.run_sync(adql).to_table()

#this takes 18min to run
```

```{code-cell} ipython3
results
```

```{code-cell} ipython3
#what about looping?
#check to see if this is faster
from astropy.table import vstack

all_tables = []
for coord in sample_table['coord']:
    # extract RA/Dec in degrees
    ra = coord.ra.deg
    dec = coord.dec.deg

    # build & run the ADQL cone‐search for this one position
    query = f"""
    SELECT
      coord_ra,
      coord_dec,
      objectId,
      detect_isPrimary
    FROM dp02_dc2_catalogs.Object
    WHERE CONTAINS(
            POINT('ICRS', coord_ra, coord_dec),
            CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, 0.001)
          ) = 1
      AND detect_isPrimary = 1
    """
    
    tbl = rsp_tap.run_sync(query).to_table()
    all_tables.append(tbl)

# combine everything into one big table
combined = vstack(all_tables)
```

```{code-cell} ipython3
combined
```

```{code-cell} ipython3
# 1. Extract the unique objectIds from your OR‐chain results
objids = sorted(set(results['objectId']))    

# 2. Build an ADQL‐friendly tuple string
#    - if only one ID, add a trailing comma so ADQL parses it as a tuple: (12345,)
if len(objids) == 1:
    id_tuple_str = f"({objids[0]},)"
else:
    id_tuple_str = "(" + ",".join(str(i) for i in objids) + ")"

# 3. Plug that into your second query
query_lc = f"""
SELECT 
  src.band,
  src.ccdVisitId,
  src.coord_ra,
  src.coord_dec,
  src.objectId,
  src.psfFlux,
  src.psfFluxErr,
  scisql_nanojanskyToAbMag(src.psfFlux) AS psfMag,
  visinfo.ccdVisitId   AS visitId,
  visinfo.band         AS visitBand,
  visinfo.expMidptMJD,
  visinfo.zeroPoint
FROM dp02_dc2_catalogs.ForcedSource AS src
JOIN dp02_dc2_catalogs.CcdVisit    AS visinfo
  ON visinfo.ccdVisitId = src.ccdVisitId
WHERE src.objectId IN {id_tuple_str}
  AND src.detect_isPrimary = 1
"""

# 4. Run it
srcs = rsp_tap.run_sync(query_lc).to_table()
print(srcs)
```

```{code-cell} ipython3
#actually check if ra and dec are in light curve table:

for tbl in rsp_tap.tables:
    print(tbl.name)
```

```{code-cell} ipython3
for tbl in rsp_tap.tables:
    # tbl.name might be 'dp02_dc2_catalogs.ForcedSource'
    if tbl.name.lower().endswith('.forcedsource'):
        cols = [col.name for col in tbl.columns]
        print(cols)
        break
```

```{code-cell} ipython3

```
