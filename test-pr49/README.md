# Testing for PR \#49

```{important}
See [results.ipynb](results.ipynb) (in this directory) for test results.

See [collect_results.py](collect_results.py) (in this directory) for the code that collects the output from each notebook.
```

PR \#49 requests to merge `troyraen:issue28` -> `fornax-navo:main`

Test outline:

-   [Run both notebooks](#run-the-notebooks) on fornaxdev
-   [Collect](#collect-the-output) some key variables and dump them to a file. Variables:
    -   band params:
        -   band, prf, cutout_width, flux_conv, mosaic_pix_scale
        -   (prf is an array, so I'll just check their hashes)
    -   source params:
        -   row.Index, band, row.ra, row.dec, row.type, row.ks_flux_aper2, imgpath, bkgpath
    -   tractor output for a subset of sources
        -   row.Index, band, flux, flux_unc
-   [Compare the outputs](#compare-output)
    -   tests pass if the corresponding outputs from each notebook are equal


## Run the notebooks

Do this on [daskhub.fornaxdev.mysmce.com](daskhub.fornaxdev.mysmce.com), once for each branch.

`troyraen:issue28` should be done first because the other branch will need to read/write files from it's directory.

Clone the repo and checkout the branch (from a terminal):

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
git pull
```

-   Run the notebook force_photometery/multiband_photometery.ipynb
    -   using the `conda env:tractor`
    -   run at least through/including the cell that creates the `paramlist` in section "Calculate forced photomery" (ok to exclude section "A little Data Exploration").

## Collect the output

-   Collect the results by executing the following in a cell in the multiband_photometery notebook. (This uses [collect_results.py](collect_results.py), which is in this directory.)

```python
branch28 = "troyraenissue28"
branchmain = "fornaxnavomain"
# --- CHOOSE ONE OF THE FOLLOWING:
branch = branch28
branch = branchmain

# directory where collect_results.py is
# output files will be written to this same directory
fbase = "../../issue28/test-pr49"

sys.path.append(fbase)
import collect_results

collect_results.run(fbase, branch)
```

## Compare output

Load the output files and test for equality by running the [results.ipynb](results.ipynb) notebook that is in this directory.
