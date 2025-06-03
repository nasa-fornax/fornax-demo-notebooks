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

```{code-cell} ipython3
%pip install -U astroquery
```

```{code-cell} ipython3
from astroquery.mast import MastMissions, Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from astropy.table import Table
```

```{code-cell} ipython3

```

```{code-cell} ipython3
class MultiIndexDFObject:
    """
    Pandas MultiIndex data frame to store & manipulate spectra.

    Examples
    --------
    # Initialize Pandas MultiIndex data frame for storing the spectra
    df_spec = MultiIndexDFObject()

    # Make a single multiindex dataframe
    df_single = pd.DataFrame(dict(wave=[0.1], flux=[0.1], err=[0.1], instrument=[instrument_name],
                                  objectid=[ccount + 1], filter=[filter_name],
                                  mission=[mission_name], label=[lab]))
    df_single = df_single.set_index(["objectid", "label", "filter", "mission"])

    # Append to existing MultiIndex object
    df_spec.append(dfsingle)

    # Show the contents
    df_spec.data
    """

    def __init__(self, data=None):
        """
        Create a MultiIndex DataFrame that is empty if data is None, else contains the data.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Dataframe to store in the `data` attribute.
        """

        index = ["objectid", "label", "filter", "mission"]
        columns = ["wave", "flux", "err", "instrument"]
        self.data = pd.DataFrame(columns=index + columns).set_index(index)
        if data is not None:
            self.append(data)
    @property
    def empty(self):
        """
        Boolean flag: True if no spectra are stored.
        """
        return self.data.empty

    def __len__(self):
        """
        Number of spectra entries stored.
        """
        return len(self.data)

    def append(self, x):
        """
        Add a new spectra data to the dataframe.

        Parameters
        ----------
        x : Pandas dataframe
            Contains columns ["wave", "flux", "err", "instrument"]
            and multi-index ["objectid", "label", "filter", "mission"].
        """

        if isinstance(x, self.__class__):
            # x is a MultiIndexDFObject. extract the DataFrame
            new_data = x.data
        else:
            # assume x is a pd.DataFrame
            new_data = x

        # if either new_data or self.data is empty we should not try to concat
        if new_data.empty:
            # leave self.data as is
            return
        if self.data.empty:
            # replace self.data with new_data
            self.data = new_data
            return

        # if we get here, both new_data and self.data contain data, so concat
        self.data = pd.concat([self.data, new_data])

    def remove(self, x):
        """
        Drop a row from the dataframe.

        Parameters
        ----------
        x : list of values
            Index values identifying rows to be dropped.
        """

        self.data = self.data.drop(x)
```

```{code-cell} ipython3
def JWST_get_spec_helper(sample_table, search_radius_arcsec, datadir, verbose=False, delete_downloaded_data=True):
    """
    Retrieve JWST spectra for sources in sample_table, returning a MultiIndexDFObject.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Must have columns 'coord' (SkyCoord), 'label', and 'objectid'.
    search_radius_arcsec : float
        Radius for query_region (arcseconds).
    datadir : str
        Base directory; spectra saved under datadir/JWST/.
    verbose : bool
        If True, print progress.
    delete_downloaded_data : bool
        If True, remove the download folder after processing.

    Returns
    -------
    df_spec : MultiIndexDFObject
        Combined spectra for all sources.
    """
    missions = MastMissions(mission='jwst')
    obs = Observations()
    df_spec = MultiIndexDFObject()

    # prepare download dir
    jwst_dir = os.path.join(datadir, 'JWST')
    os.makedirs(jwst_dir, exist_ok=True)

    for row in sample_table:
        coord = row['coord']
        label = row['label']
        if verbose:
            print(f"Querying {label} at {coord.to_string('hmsdms')}")

        # query metadata
        radius_deg = (search_radius_arcsec * u.arcsec).to(u.deg).value
        meta = missions.query_region(coord, radius=radius_deg)
        if len(meta) == 0:
            if verbose:
                print(f"  No data for {label}")
            continue

        # products and filtering
        prods = obs.get_product_list(meta)
        spec_prods = obs.filter_products(
            prods,
            productType=['SCIENCE'], extension='fits',
            calib_level=[2,3,4], productSubGroupDescription=['X1D'],
            dataRights=['PUBLIC']
        )
        if verbose:
            print(f"  {len(spec_prods)} spectra found")
        if len(spec_prods) == 0:
            continue

        # download
        manifest = obs.download_products(spec_prods, download_dir=jwst_dir, verbose=False)
        paths = manifest.get('Local Path', [])

        # read and append
        for prod, path in zip(spec_prods, paths):
            tbl = Table.read(path, hdu=1)
            wave = tbl['WAVELENGTH'].data * tbl['WAVELENGTH'].unit
            flux = tbl['FLUX'].data * tbl['FLUX'].unit
            err = tbl['FLUX_ERROR'].data * tbl['FLUX_ERROR'].unit
            inst = prod['instrument_name']
            filt = prod['filters']

            df = pd.DataFrame({
                'wave': [wave], 'flux': [flux], 'err': [err], 'instrument': [inst],
                 'label': [label], 'filter': [filt], 'mission': ['JWST']
            }).set_index(['objectid','label','filter','mission'])
            df_spec.append(df)

        if delete_downloaded_data:
            shutil.rmtree(jwst_dir)
            os.makedirs(jwst_dir, exist_ok=True)

    return df_spec


def JWST_group_spectra(df_spec, quickplot=False):
    """
    Stack spectra by (objectid,label,filter,mission), computing median flux.

    Parameters
    ----------
    df_spec : MultiIndexDFObject
        Raw spectra.
    quickplot : bool
        If True, plot each stacked spectrum.

    Returns
    -------
    df_grouped : MultiIndexDFObject
        One median spectrum per unique (objectid,label,filter,mission).
    """
    data = df_spec.data.reset_index()
    df_out = MultiIndexDFObject()

    grouped = data.groupby(['objectid','label','filter','mission'])
    for idx, grp in grouped:
        wave = grp['wave'].iloc[0]
        arrs = np.vstack([arr.value for arr in grp['flux']])
        median_flux = np.nanmedian(arrs, axis=0) * grp['flux'].iloc[0].unit
        err = np.zeros_like(median_flux.value) * median_flux.unit
        inst = grp['instrument'].iloc[0]

        df = pd.DataFrame({
            'wave': [wave], 'flux': [median_flux], 'err': [err], 'instrument': [inst],
            'objectid': [idx[0]], 'label': [idx[1]], 'filter': [idx[2]], 'mission': [idx[3]]
        }).set_index(['objectid','label','filter','mission'])
        df_out.append(df)

        if quickplot:
            plt.plot(wave, median_flux)
            plt.xlabel('Wavelength')
            plt.ylabel('Flux')
            plt.title(f"{idx[1]} - {idx[2]}")
            plt.show()

    return df_out
```

```{code-cell} ipython3
def JWST_get_spec(sample_table, search_radius_arcsec, datadir,
                  verbose=False, delete_downloaded_data=True, quickplot=False):
    """
    Retrieve JWST spectra for a list of sources and group/stack them.
    This wrapper runs two sub-functions:
    - `JWST_get_spec_helper()` which searches, downloads, and retrieves spectra.
    - `JWST_group_spectra()` which groups and stacks the spectra.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with 'coord', 'label', and 'objectid' for each source.
    search_radius_arcsec : float
        Search radius in arcseconds.
    datadir : str
        Directory under which a 'JWST' subfolder will store downloaded files.
    verbose : bool, optional
        If True, print progress messages. Default is False.
    delete_downloaded_data : bool, optional
        If True, delete downloaded files after processing. Default is True.
    quickplot : bool, optional
        If True, plot each stacked spectrum. Default is False.

    Returns
    -------
    MultiIndexDFObject
        Consolidated and grouped JWST spectra.
    """
    # Fetch individual spectra
    print("Searching and downloading spectra...")
    df_jwst_all = JWST_get_spec_helper(
        sample_table,
        search_radius_arcsec,
        datadir,
        verbose=verbose,
        delete_downloaded_data=delete_downloaded_data
    )
    print("Done fetching spectra.")

    # Group and stack
    print("Grouping spectra...")
    df_jwst_group = JWST_group_spectra(df_jwst_all, quickplot=quickplot)
    print("Done grouping spectra.")

    return df_jwst_group
```

```{code-cell} ipython3
def create_sample_table_from_skycoords(coords, object_ids=None):
    """
    Build a simple Astropy Table from SkyCoord positions.

    Parameters
    ----------
    coords : SkyCoord
        An Astropy SkyCoord instance or array of instances.
    object_ids : array-like, optional
        IDs corresponding to each coord. If provided, must match length of `coords`.

    Returns
    -------
    sample_table : astropy.table.Table
        Table with columns:
        - coord : SkyCoord
        - ra    : float (deg)
        - dec   : float (deg)
        - object_id : same type as `object_ids` (if provided)
    """
    sample_table = Table()
    sample_table['coord'] = coords
    sample_table['ra']    = coords.ra.deg
    sample_table['dec']   = coords.dec.deg
    if object_ids is not None:
        sample_table['object_id'] = object_ids
    return sample_table
```

```{code-cell} ipython3

```

```{code-cell} ipython3
#file location
csv_path = "/home/jovyan/fornax-demo-notebooks/spectroscopy/data/SPICY_C1_2.csv"
SPICY = pd.read_csv(csv_path)
# Create a vectorized SkyCoord from the RA and Dec columns
ra_str = SPICY["RA"].astype(str).str.strip()
dec_str = SPICY["DEC"].astype(str).str.strip()

SPICY_skycoords = coord.SkyCoord(
        ra=ra_str,
        dec=dec_str,
        unit=(u.hourangle, u.deg),
        frame='icrs')

sample_table = create_sample_table_from_skycoords(SPICY_skycoords)
sample_table['label'] = 'SPICY'
```

```{code-cell} ipython3
search_radius_arcsec = 0.5
datadir = 'jwst'
df_jwst= JWST_get_spec(sample_table, search_radius_arcsec, datadir,
                  verbose=False, delete_downloaded_data=True, quickplot=False)
```

```{code-cell} ipython3
df_spec = MultiIndexDFObject()
df_spec.append(df_jwst)
```

```{code-cell} ipython3
df_spec.data
```

```{code-cell} ipython3
SPICY_skycoords
```

```{code-cell} ipython3
missions = MastMissions(mission='jwst')
```

```{code-cell} ipython3
# Create coordinate object
coords = SkyCoord(269.31, 66.47, unit=('deg'), 
                  exp_type='MIR_LRS-FIXEDSLIT,MIR_LRS-SLITLESS,MIR_MRS,NRC_GRISM,NRC_WFSS,NIS_SOSS,NIS_WFSS,NRS_FIXEDSLIT,NRS_MSASPEC',  # Spectroscopy data
            instrume='!FGS',  # Any instrument except FGS
                 )

# Query for results within 10 arcminutes of coords
results = missions.query_region(coords, radius=0.5*u.deg)

# Display results
print(f'Total number of results: {len(results)}')
results[:5]
```

```{code-cell} ipython3
col_table = missions.get_column_list()
```

```{code-cell} ipython3
nlines = len(col_table) + 2
col_table.pprint(nlines) 
```

```{code-cell} ipython3
from astroquery.mast import MastMissions, Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table, vstack
import numpy as np


targets = [
    (SkyCoord("09h54m49.40s +09d16m15.9s", frame='icrs'), "NGC3049"),
    (SkyCoord("12h45m17.44s +27d07m31.8s", frame='icrs'), "NGC4670"),
    (SkyCoord("14h01m19.92s -33d04m10.7s", frame='icrs'), "Tol_89"),
    (SkyCoord(150.091 * u.deg,  2.2745833 * u.deg),         "COSMOS1"),
]

radius_arcsec = 10

# MAST missions interface
m = MastMissions(mission='jwst')

# loop over targets
for coord, label in targets:
    print(f"MM Querying JWST around {label} ({coord.to_string('hmsdms')})")

    # cone search for JWST datasets
    datasets = m.query_region(
        coord,
        radius=(radius_arcsec * u.arcsec).to(u.deg).value, 
        exp_type='MIR_LRS-FIXEDSLIT,MIR_LRS-SLITLESS,MIR_MRS,NRC_GRISM,NRC_WFSS,NIS_SOSS,NIS_WFSS,NRS_FIXEDSLIT,NRS_MSASPEC',  # Spectroscopy data
        instrume='!FGS',  # Any instrument except FGS
        access='PUBLIC'            # public data
    )

    print(len(datasets), "length(datasets)")
    
    if len(datasets) == 0:
        print(f"  No datasets found for {label}")
        continue

    
    # get all products for these datasets
    print("getting product list")
    #products = m.get_product_list(datasets)  #timeout problem
    #Get products in groups of 250
    batch_size = 250
    batches = [datasets[i:i + batch_size] for i in range(0, len(datasets), batch_size)]
 
    products = Table()
    for batch in batches:
        # Get the products for each batch
        print(f"Getting products for batch of {len(batch)} products")
        batch_prods = m.get_product_list(batch)
 
        # Append the results to the products table
        products = vstack([products, batch_prods])


    # filter down to calibrated 1D science spectra
    print("filtering products")
    filtered = m.filter_products(
        products,
        type='science',         # only science products
        extension='fits',                # FITS files
        #calib_level=[2, 3, 4],           # calibrated data
        #productSubGroupDescription=['X1D'],  # 1D spectra
    )

    #then I would go on to downloading, but timeout happens above for COSMOS1
```

```{code-cell} ipython3
filtered = m.filter_products(
    products,
    type='science',         # only science products
    extension='fits',                # FITS files
    file_suffix=['_c1d','x1d'],  # calibrated 1D spectra
    access=['PUBLIC']            # public data
)

len(filtered)
```

```{code-cell} ipython3
tab = products.get_column_list()
```

```{code-cell} ipython3
filtered
```

```{code-cell} ipython3
col_tab = m.get_column_list()
```

```{code-cell} ipython3
len(col_tab)
```

```{code-cell} ipython3
for row in col_tab:
    print(row)
```

```{code-cell} ipython3

```
