# # Testing for PR \#49
#
# This script collects output from a notebook and dumps it to local files.
#
# Usage:
#   See the README.md in this directory.
#   In short:
#       - run the multiband_photometry.ipynb notebook
#       - then, copy and paste the contents of this file into the notebook
#           - uncomment the appropriate `branch` variable
#           - execute it
#       - then set `branch` and `fbase` and run the test by calling `collect_output(fbase, branch)`
from random import randint


# branch names
branch28 = "troyraenissue28"
branchmain = "fornaxnavomain"


# define file paths that we'll write to, plus their columns

def fband(fbase, branch): return f"{fbase}/bandparams-{branch}.csv"


band_cols = ["idx", "prf", "cutout_width", "flux_conv", "mosaic_pix_scale"]
def fsource(fbase, branch): return f"{fbase}/sourceparams-{branch}.csv"


source_cols = [
    "row.Index", "band.idx", "row.ra", "row.dec", "row.type", "row.ks_flux_aper2", "imgpath", "bkgpath"
]
def ftractor(fbase, branch): return f"{fbase}/tractoroutput-{branch}.csv"


tractor_cols = ["row.Index", "band.idx", "flux", "flux_unc"]


def get_indexes_for_tractor(fbase, branch):
    findexes_for_tractor = f"{fbase}/indexes_for_tractor.csv"
    if branch == branch28:
        # generate the list
        # total number of sources (initial IRSA radius is 15 arcmin)
        nsources_in_15arcmin = 459486
        nsources_test = 100  # number of sources to actually get tractor output on
        indexes_for_tractor = [randint(0, nsources_in_15arcmin-1)
                               for i in range(nsources_test)]
        # write it to file so the other branch can use the same list
        with open(findexes_for_tractor, 'w') as fout:
            np.savetxt(fout, indexes_for_tractor, fmt="%.1u")
    else:
        # read in the pre-generated list
        with open(findexes_for_tractor, 'r') as fin:
            indexes_for_tractor = list(int(i) for i in np.genfromtxt(fin))
    return indexes_for_tractor


def collect_bands_issue28(all_bands, fout, cols):
    paramdf = pd.DataFrame(
        all_bands, columns=cols).set_index(cols[0])
    # hash the prf
    paramdf['prf_hash'] = paramdf['prf'].apply(
        lambda prf: hash(tuple(prf.flatten())))
    del paramdf['prf']
    paramdf.to_csv(fout)


def collect_bands_main(
    prfs, cutout_width_list, flux_conv_list, mosaic_pix_scale_list, fout, cols
):
    all_bands = [
        [
            bandidx,
            prfs[bandidx],
            cutout_width_list[bandidx],
            flux_conv_list[bandidx],
            mosaic_pix_scale_list[bandidx],
        ]
        for bandidx in range(6)
    ]
    collect_bands_issue28(all_bands, fout, cols)


def collect_source_params_issue28(paramlist, fout, cols):
    outlist = []
    for (index, band, ra, dec, stype, ks_flux_aper2, infiles, df) in paramlist:
        outlist.append([
            index, band.idx, ra, dec, stype, ks_flux_aper2, *infiles
        ])
    dfout = pd.DataFrame(outlist, columns=cols).set_index(cols[0])

    def populate_path(row):
        # the code will use the imgpath for the bkgpath, if bkgpath is null
        if row.bkgpath is not None:
            return row.bkgpath
        return row.imgpath
    # populate the null paths so we can compare with the main branch
    dfout["bkgpath"] = dfout[["imgpath", "bkgpath"]].apply(
        lambda row: row.bkgpath if row.bkgpath is not None else row.imgpath,
        axis=1,
    )
    # populate_path, axis=1)
    dfout.to_csv(fout)


def collect_source_params_main(paramlist, fout, cols):
    outlist = []
    for (index, bandidx, ra, dec, stype, ks_flux_aper2, g_band) in paramlist:
        outlist.append([
            index, bandidx, ra, dec, stype, ks_flux_aper2, infiles[g_band], skybgfiles[g_band]
        ])
    dfout = pd.DataFrame(outlist, columns=cols).set_index(cols[0])
    dfout.to_csv(fout)


def collect_tractor_results(paramlist, idxs, fout, cols):
    outlist = []
    for idx in idxs:
        rowIndex = paramlist[idx][0]
        tractor_results = calc_instrflux(*paramlist[idx][1:])
        outlist.append([rowIndex, *tractor_results])
    tractordf = pd.DataFrame(outlist, columns=cols).set_index(cols[0])
    tractordf.to_csv(fout)


def collect_output(fbase, branch):
    # get indexes of a random sample of sources to get tractor output for
    indexes_for_tractor = get_indexes_for_tractor(fbase, branch)

    # ### Collect band params and dump to file
    fout = fband(fbase, branch)
    cols = band_cols
    print(f"Collecting Band params and dumping to file at {fout}")
    if branch == branch28:
        collect_bands_issue28(all_bands, fout, cols)
    else:
        collect_bands_main(prfs, cutout_width_list,
                           flux_conv_list, mosaic_pix_scale_list, fout, cols)

    # ### Collect source params and dump to file
    fout = fsource(fbase, branch)
    cols = source_cols
    print(f"Collecting Source params and dumping to file at {fout}")
    if branch == branch28:
        collect_source_params_issue28(paramlist, fout, cols)
    else:
        collect_source_params_main(paramlist, fout, cols)

    # ### Collect tractor output and dump to file
    fout = ftractor(fbase, branch)
    cols = tractor_cols
    print(
        f"Running tractor, collecting results, and dumping to file at {fout}")
    # same for both branches
    collect_tractor_results(paramlist, indexes_for_tractor, fout, cols)
