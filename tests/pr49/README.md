# Testing for PR \#49

PR \#49 requests to merge ``troyraen:issue28`` -> ``fornax-navo:main``

This page contains a description of the tests written for that PR, and instructions for running them.

These tests have already been run -- see [RESULTS.out](RESULTS.out).

See [collect_and_test.py](collect_and_test.py) for the code used to
collect the output from each notebook and run the tests.

## Test outline:

-   [Run notebooks](#run-the-notebooks) from both branches on fornaxdev
-   [Collect](#collect-the-output) some key variables and write them to disk. Variables will include:
    -   band params:
        -   band, prf, cutout_width, flux_conv, mosaic_pix_scale
        -   (prf is an array, so I'll just check their hashes)
    -   source params:
        -   row.Index, band, row.ra, row.dec, row.type, row.ks_flux_aper2, imgpath, bkgpath
    -   ``calc_instrflux`` output for a subset of sources (same subset for both branches)
        -   row.Index, band, flux, flux_unc
-   [Compare the outputs](#compare-output) from the "Collect" step
    -   tests pass if the corresponding outputs from each notebook are equal


## Run the notebooks

Do this on [daskhub.fornaxdev.mysmce.com](daskhub.fornaxdev.mysmce.com), once for each branch.

``troyraen:issue28`` should be done first because the other branch will need to read/write files from it's ``tests`` directory.

Clone the repo and checkout the branch:

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

-   Run the notebook `force_photometery/multiband_photometery.ipynb`
    -   use the kernel ``conda env:tractor``
    -   run at least through/including the cell that creates the ``paramlist`` in section "Calculate forced photomery" (ok to exclude section "A little Data Exploration").
        -   **When creating the `paramlist` in branch=issue28, use ``sci_bkg_pairs`` instead of ``hdu_pairs`` when creating the `img_pair`.**
            This will use file paths (rather than pre-loaded HDUs), which we need to compare with the other branch.

## Collect the output

In a new cell in the notebook that you ran in the last step, do the following.
Note: Fill in the ``issue28dir`` and uncomment a ``branch`` before executing.

```python
# TODO: enter the path to the root dir for issue28 branch
issue28dir = "path/to/issue28"

# source the output collection code into the environment
# use the same .py from issue28 for both branches
%run -i "$issue28dir/tests/pr49/collect_and_test.py"

# TODO: uncomment ONE of the following
# branch = branch28
# branch = branchmain

# collect output
run_collectors(branch)
```

## Compare output

After doing the previous two steps in both branches,

Load the output files and test for equality by running the following from within this directory:

```bash
# TODO: enter the path to the root dir for issue28 branch
issue28dir="path/to/issue28"

# run the tests
cd "${issue28dir}/tests/pr49"
python -c "import collect_and_test; collect_and_test.run_equality_tests()"
```
