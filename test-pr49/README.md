# Testing for PR \#49

PR \#49 requests to merge `troyraen:issue28` -> `fornax-navo:main`

Test outline:

-   [Run both notebooks](#run-the-notebooks-and-collect-data) on fornaxdev, then manually collect some key variables and dump them to a file. Variables:
    -   band params:
        -   band, prf, cutout_width, flux_conv, mosaic_pix_scale
        -   (prf is an array, so I'll just check their hashes)
    -   source params:
        -   row.Index, band, row.ra, row.dec, row.type, row.ks_flux_aper2, imgpath, bkgpath
    -   tractor output for a subset of sources
-   [Compare the outputs](#compare-output)
    -   tests pass if the corresponding outputs from each notebook are equal

## Run the notebooks and collect data

Do this on [daskhub.fornaxdev.mysmce.com](daskhub.fornaxdev.mysmce.com), once for each branch.

`troyraen:issue28` should be done first because the other branch will need to read/write files from it's directory.

Get the branch:

```bash
# --- use one of:
gitorg="troyraen"
branch="issue28"
# --- or:
gitorg="fornax-navo"
branch="main"

git clone https://github.com/${gitorg}/fornax-demo-notebooks.git "${branch}"
cd "${branch}"
git fetch
git checkout "${branch}"
```

-   Run the notebook force_photometery/multiband_photometery.ipynb
    -   using the `conda env:tractor`
    -   run at least through/including the cell that creates the `paramlist` in section "Calculate forced photomery" (ok to exclude section "A little Data Exploration").
-   Then add+execute cells with the following to collect output:

### Setup

```python
branch28 = "troyraenissue28"
branchmain = "fornaxnavomain"
# --- use one of:
branch = branch28
branch = branchmain

fbase = "../../issue28/test-pr49"

# files we'll write, plus their columns
def fband(branch): return f"{fbase}/bandparams-{branch}.csv"
band_cols = ["band", "prf", "cutout_width", "flux_conv", "mosaic_pix_scale"]
def fsource(branch): return f"{fbase}/sourceparams-{branch}.csv"
source_cols = [
    "row.Index", "band", "row.ra", "row.dec", "row.type", "row.ks_flux_aper2", "imgpath", "bkgpath"
]
def ftractor(branch): return f"{fbase}/tractoroutput-{branch}.csv"
tractor_cols = ["band", "microJy_flux", "microJy_unc"]

# random sample of sources to get tractor output on
findexes_for_tractor = f"{fbase}/indexes_for_tractor.csv"
if branch == "troyraenissue28":
    from random import randint
    nsources_in_15arcmin = 459486  # total number of sources (initial IRSA radius is 15 arcmin)
    nsources_test = 100  # number of sources to actually get tractor output on
    indexes_for_tractor = [randint(0, nsources_in_15arcmin-1) for i in range(nsources_test)]
    with open(findexes_for_tractor, 'w') as fout:
        np.savetxt(fout, indexes_for_tractor, fmt="1.1x")
else:
    with open(findexes_for_tractor, 'r') as fin:
        indexes_for_tractor = list(np.genfromtxt(fin))
```

### Collect band params and dump to file

```python
# collect band params
# prf is a multi-dimensional array, so I'll collect and compare their hashes
fout = fband(branch)
cols = band_cols

def collect_band_params_issue28(all_band_params):
    paramdf = pd.DataFrame(all_band_params, columns=cols).set_index(cols[0])
    # hash the prf
    paramdf['prf_hash'] = paramdf['prf'].apply(lambda prf: hash(tuple(prf.flatten())))
    del paramdf['prf']
    paramdf.to_csv(fout)

def collect_band_params_main(
    prfs, cutout_width_list, flux_conv_list, mosaic_pix_scale_list
):
    all_band_params = [
        [
            band,
            prfs[band],
            cutout_width_list[band],
            flux_conv_list[band],
            mosaic_pix_scale_list[band],
        ]
        for band in range(6)
    ]
    collect_band_params_issue28(all_band_params)

# --- use one of:
collect_band_params_issue28(all_band_params)
collect_band_params_main(prfs, cutout_width_list, flux_conv_list, mosaic_pix_scale_list)
```

### Collect source params and dump to file

```python
# collect source params
fout = fsource(branch)
cols = source_cols

def collect_source_params_issue28(paramlist):
    outlist = []
    for (index, band_params, ra, dec, stype, ks_flux_aper2, infiles, df) in paramlist:
        outlist.append([
            index, band_params.band, ra, dec, stype, ks_flux_aper2, *infiles
        ])
    dfout = pd.DataFrame(outlist, columns=cols).set_index(cols[0])
    dfout.to_csv(fout)

def collect_source_params_main(paramlist):
    outlist = []
    for (index, band, ra, dec, stype, ks_flux_aper2, g_band) in paramlist:
        outlist.append([
            index, band, ra, dec, stype, ks_flux_aper2, infiles[g_band], skybgfiles[g_band]
        ])
    dfout = pd.DataFrame(outlist, columns=cols).set_index(cols[0])
    dfout.to_csv(fout)

# --- use one of:
collect_source_params_issue28(paramlist)
collect_source_params_main(paramlist)
```

### Collect tractor output and dump to file

```python
fout = ftractor(branch)
cols = tractor_cols

def collect_tractor_results(paramlist, idxs)
    outlist = []
    for idx in idxs:
        outlist.append(calc_instrflux(*paramlist[idx][1:]))
    tractordf = pd.DataFrame(outlist, columns=cols).set_index(cols[0])
    tractordf.to_csv(fout)

# same for both branches
collect_tractor_results(paramlist, indexes_for_tractor)
```

## Compare output

Do this in the `troyraen:issue28` notebook, after running the above in both notebooks.

Load the output files and test for equality by running the [results.ipynb](results.ipynb) notebook that is in this directory.
