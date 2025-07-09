import os
import shutil
import warnings
import fsspec

import astropy.constants as const
import astropy.units as u
from astropy.io import fits
import astroquery.exceptions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.table import vstack
from astroquery.mast import Observations
from specutils import Spectrum1D

from data_structures_spec import MultiIndexDFObject

def JWST_get_spec(sample_table, search_radius_arcsec, verbose, max_spectra_per_source=None):
    """
    Retrieve JWST spectra for a list of sources and groups/stacks them.
    This main function runs two sub-functions:
    - `JWST_get_spec_helper()` which searches and retrieves the spectra from
      the cloud.
    - `JWST_group_spectra()` which groups and stacks the spectra.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with the coordinates and journal reference labels of the sources.
    search_radius_arcsec : float
        Search radius in arcseconds.
    verbose : bool
        Verbosity level. Set to True for extra talking.
    max_spectra_per_source : int or None, optional
        Maximum number of spectra to retrieve per source. If None, retrieve all available spectra.

    Returns
    -------
    MultiIndexDFObject
        The spectra returned from the archive.
    """

    # Get the spectra
    print("Searching Spectra in the cloud... ")
    df_jwst_all = JWST_get_spec_helper(
        sample_table, search_radius_arcsec, verbose, max_spectra_per_source = max_spectra_per_source)
    print("done")

    # Group
    print("Grouping Spectra... ")
    df_jwst_group = JWST_group_spectra(df_jwst_all, verbose=verbose, quickplot=False)
    print("done")

    return df_jwst_group


def JWST_get_spec_helper(sample_table, search_radius_arcsec, verbose, max_spectra_per_source=None):
    """
    Retrieve JWST spectra for a list of sources.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with the coordinates and journal reference labels of the sources.
    search_radius_arcsec : float
        Search radius in arcseconds.
    verbose : bool
        Verbosity level. Set to True for extra talking.
    max_spectra_per_source : int or None, optional
        Maximum number of spectra to retrieve per source. If None, retrieve all available spectra.

    Returns
    -------
    MultiIndexDFObject
        The spectra returned from the archive.
    """

    # Enable cloud data access for MAST once
    Observations.enable_cloud_dataset()


    # Initialize multi-index object:
    df_spec = MultiIndexDFObject()

    for stab in sample_table:

        print("Processing source {}".format(stab["label"]))

        # Query results
        search_coords = stab["coord"]
        # If no results are found, this will raise a warning. We explicitly handle the no-results
        # case below, so let's suppress the warning to avoid confusing notebook users.
        warnings.filterwarnings("ignore", message='Query returned no results.',
                                category=astroquery.exceptions.NoResultsWarning,
                                module="astroquery.mast.discovery_portal")
        query_results = Observations.query_criteria(
            coordinates=search_coords, radius=search_radius_arcsec * u.arcsec,
            dataproduct_type=["spectrum"], obs_collection=["JWST"], intentType="science",
            calib_level=[2, 3, 4], instrument_name=['NIRSPEC/MSA', 'NIRSPEC/SLIT'],
            dataRights=['PUBLIC'])
        print("Number of search results: {}".format(len(query_results)))

        if len(query_results) == 0:
            print("Source {} could not be found".format(stab["label"]))
            continue

        # Retrieve spectra
        data_products = [Observations.get_product_list(obs) for obs in query_results]
        data_products_list = vstack(data_products)
        
        # Filter
        data_products_list_filter = Observations.filter_products(
            data_products_list, productType=["SCIENCE"], extension="fits",
            calib_level=[2, 3, 4],  # only calibrated data
            productSubGroupDescription=["X1D"],  # only 1D spectra
            dataRights=['PUBLIC'])  # only public data
        print("Number of files found: {}".format(len(data_products_list_filter)))

        #no need to continue if there aren't any spectra
        if len(data_products_list_filter) == 0:
            print("No spectra found for source {}.".format(stab["label"]))
            continue
            
        #This code takes a long time if you let it get all spectra, this
        # next part filters out some of the results if not all spectra are needed.
        # change max_spectra_per_source to None if you want all of them
        if max_spectra_per_source is not None and len(data_products_list_filter) > max_spectra_per_source:
            data_products_list_filter = data_products_list_filter[:max_spectra_per_source]
            print(f"Limiting to first {max_spectra_per_source} spectra for source {stab['label']}.")
            
        # Get cloud access URIs (returns a list of strings)
        cloud_uris_dict = Observations.get_cloud_uris(data_products_list_filter, return_uri_map=True)

        # Create table with metadata from data_priducts_list_filter and cloud URIs
        keys = ["filters", "obs_collection", "instrument_name", "calib_level",
                "t_obs_release", "proposal_id", "obsid", "objID", "distance"]
        tab = Table(names=keys + ["productFilename", "clouduri"],
                    dtype=[str, str, str, int, float, int, int, int, float, str, str])

        for row in data_products_list_filter:
            filename = str(row["productFilename"])
            lookup_key = f"mast:JWST/product/{filename}"
            uri = cloud_uris_dict.get(lookup_key)
            if uri is None:
                print(f"Skipping {filename}: not available in cloud.")
                continue

            obsid = row["parent_obsid"]
            idx_cross = np.where(query_results["obsid"] == obsid)[0]

            tmp = query_results[idx_cross][keys]
            tab.add_row(list(tmp[0]) + [filename, uri])

            
        # Create multi-index object
        for jj in range(len(tab)):

            # open spectrum directly from the cloud
            filepath = tab["clouduri"][jj]
            
            with fsspec.open(filepath, mode='rb', anon=True) as f:  # Open the file from S3
                with fits.open(f) as hdul:  # Open the FITS file
                    spec1d = Table(hdul[1].data)
                    columns = hdul[1].columns
                    # Explicitly extract units
                    wave_unit = u.Unit(getattr(columns["WAVELENGTH"], 'unit', None))
                    flux_unit = u.Unit(getattr(columns["FLUX"], 'unit', None))
                    err_unit = u.Unit(getattr(columns["FLUX_ERROR"], 'unit', None))

            dfsingle = pd.DataFrame(dict(
                wave=[spec1d["WAVELENGTH"].data * wave_unit],
                flux=[spec1d["FLUX"].data * flux_unit],
                err=[spec1d["FLUX_ERROR"].data * err_unit],
                label=[stab["label"]],
                objectid=[stab["objectid"]],
                mission=[tab["obs_collection"][jj]],
                instrument=[tab["instrument_name"][jj]],
                filter=[tab["filters"][jj]],
            )).set_index(["objectid", "label", "filter", "mission"])
            df_spec.append(dfsingle)


    return df_spec

def JWST_group_spectra(df, verbose, quickplot):
    """
    Group the JWST spectra and removes entries that have no spectra.
    Stack spectra that are similar and create a new DataFrame.

    Parameters
    ----------
    df : MultiIndexDFObject
        Raw JWST multi-index object (output from JWST_get_spec()).
    verbose : bool
        Flag for verbosity: True or False.
    quickplot : bool
        If True, quick plots are made for each spectral group.

    Returns
    -------
    MultiIndexDFObject
        Consolidated and grouped data structure storing the spectra.
    """

    # Initialize multi-index object:
    df_spec = MultiIndexDFObject()

    # Create data table from DF.
    tab = df.data.reset_index()

    # Get objects
    objects_unique = np.unique(tab["label"])

    for obj in objects_unique:
        print("Grouping object {}".format(obj))

        # Get filters
        filters_unique = np.unique(tab["filter"])
        if verbose:
            print("List of filters in data frame: {}".format(" | ".join(filters_unique)))

        for filt in filters_unique:
            if verbose:
                print("Processing {}: ".format(filt), end=" ")

            sel = np.where((tab["filter"] == filt) & (tab["label"] == obj))[0]
            tab_sel = tab.iloc[sel]
            if verbose:
                print("Number of items: {}".format(len(sel)), end=" | ")

            # get good ones
            sum_flux = np.asarray(
                [np.nansum(tab_sel.iloc[iii]["flux"]).value for iii in range(len(tab_sel))])
            idx_good = np.where(sum_flux > 0)[0]
            if verbose:
                print("Number of good spectra: {}".format(len(idx_good)))

            if len(idx_good) == 0:
                continue

            # Create wavelength grid
            wave_grid = tab_sel.iloc[idx_good[0]]["wave"]  # NEED TO BE MADE BETTER LATER

            # Interpolate fluxes
            fluxes_int = np.asarray(
                [np.interp(wave_grid, tab_sel.iloc[idx]["wave"], tab_sel.iloc[idx]["flux"]) for idx in idx_good])
            fluxes_units = [tab_sel.iloc[idx]["flux"].unit for idx in idx_good]

            # Sometimes fluxes are all NaN. We'll leave these in and ignore the RuntimeWarning.
            warnings.filterwarnings("ignore", message='All-NaN slice encountered', category=RuntimeWarning)
            fluxes_stack = np.nanmedian(fluxes_int, axis=0)
            if verbose:
                print("Units of fluxes for each spectrum: {}".format(
                    ",".join([str(tt) for tt in fluxes_units])))

            # Unit conversion to erg/s/cm2/A
            # (note fluxes are nominally in Jy. So have to do the step with dividing by lam^2)
            fluxes_stack_cgs = (fluxes_stack * fluxes_units[0]).to(u.erg / u.second / (
                u.centimeter**2) / u.hertz) * (const.c.to(u.angstrom/u.second)) / (wave_grid.to(u.angstrom)**2)
            fluxes_stack_cgs = fluxes_stack_cgs.to(
                u.erg / u.second / (u.centimeter**2) / u.angstrom)

            # Add to data frame
            dfsingle = pd.DataFrame(dict(
                wave=[wave_grid.to(u.micrometer)], flux=[fluxes_stack_cgs],
                err=[np.repeat(0, len(fluxes_stack_cgs))], label=[tab_sel["label"].iloc[0]],
                objectid=[tab_sel["objectid"].iloc[0]], mission=[tab_sel["mission"].iloc[0]],
                instrument=[tab_sel["instrument"].iloc[0]], filter=[tab_sel["filter"].iloc[0]]))
            dfsingle = dfsingle.set_index(["objectid", "label", "filter", "mission"])
            df_spec.append(dfsingle)

            # Quick plot
            if quickplot:
                tmp = np.percentile(fluxes_stack, q=(1, 50, 99))
                plt.plot(wave_grid, fluxes_stack)
                plt.ylim(tmp[0], tmp[2])
                plt.xlabel(r"Observed Wavelength [$\mu$m]")
                plt.ylabel(r"Flux [Jy]")
                plt.show()

    return df_spec


def HST_get_spec(sample_table, search_radius_arcsec, datadir, verbose,
                 delete_downloaded_data=True):
    """
    Retrieve HST spectra for a list of sources.

    Parameters
    ----------
    sample_table : astropy.table.Table
        Table with the coordinates and journal reference labels of the sources.
    search_radius_arcsec : float
        Search radius in arcseconds.
    datadir : str
        Data directory where to store the data. Each function will create a
        separate data directory (for example "[datadir]/HST/" for HST data).
    verbose : bool
        Verbosity level. Set to True for extra talking.
    delete_downloaded_data : bool, optional
        If True, delete the downloaded data files. Default is True.

    Returns
    -------
    MultiIndexDFObject
        The spectra returned from the archive.
    """

    # Create directory
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    this_data_dir = os.path.join(datadir, "HST/")

    # Initialize multi-index object:
    df_spec = MultiIndexDFObject()

    for stab in sample_table:

        print("Processing source {}".format(stab["label"]))

        # Query results
        search_coords = stab["coord"]
        # If no results are found, this will raise a warning. We explicitly handle the no-results
        # case below, so let's suppress the warning to avoid confusing notebook users.
        warnings.filterwarnings("ignore", message='Query returned no results.',
                                category=astroquery.exceptions.NoResultsWarning,
                                module="astroquery.mast.discovery_portal")
        query_results = Observations.query_criteria(
            coordinates=search_coords, radius=search_radius_arcsec * u.arcsec,
            dataproduct_type=["spectrum"], obs_collection=["HST"], intentType="science",
            calib_level=[3, 4],)
        print("Number of search results: {}".format(len(query_results)))

        if len(query_results) == 0:
            print("Source {} could not be found".format(stab["label"]))
            continue

        # Retrieve spectra
        data_products_list = Observations.get_product_list(query_results)

        # Filter
        data_products_list_filter = Observations.filter_products(
            data_products_list, productType=["SCIENCE"], extension="fits",
            calib_level=[3, 4],  # only fully reduced or contributed
            productSubGroupDescription=["SX1"])  # only 1D spectra
        print("Number of files to download: {}".format(len(data_products_list_filter)))

        if len(data_products_list_filter) == 0:
            print("Nothing to download for source {}.".format(stab["label"]))
            continue

        # Download
        download_results = Observations.download_products(
            data_products_list_filter, download_dir=this_data_dir, verbose=False)

        # Create table
        # NOTE: `download_results` has NOT the same order as `data_products_list_filter`.
        # We therefore have to "manually" get the product file names here and then use
        # those to open the files.
        keys = ["filters", "obs_collection", "instrument_name", "calib_level",
                "t_obs_release", "proposal_id", "obsid", "objID", "distance"]
        tab = Table(names=keys + ["productFilename"], dtype=[str,
                    str, str, int, float, int, int, int, float]+[str])
        for jj in range(len(data_products_list_filter)):
            idx_cross = np.where(query_results["obsid"] ==
                                 data_products_list_filter["obsID"][jj])[0]
            tmp = query_results[idx_cross][keys]
            tab.add_row(list(tmp[0]) + [data_products_list_filter["productFilename"][jj]])

        # Create multi-index object
        for jj in range(len(tab)):

            # find correct path name:
            # Note that `download_results` does NOT have same order as `tab`!!
            file_idx = np.where([tab["productFilename"][jj] in download_results["Local Path"][iii]
                                for iii in range(len(download_results))])[0]

            # open spectrum
            filepath = download_results["Local Path"][file_idx[0]]
            spec1d = Spectrum1D.read(filepath)

            # Note: this should be in erg/s/cm2/A and any wavelength unit.
            dfsingle = pd.DataFrame(dict(
                wave=[spec1d.spectral_axis], flux=[spec1d.flux], err=[
                    spec1d.uncertainty.array * spec1d.uncertainty.unit],
                label=[stab["label"]],
                objectid=[stab["objectid"]],
                mission=[tab["obs_collection"][jj]],
                instrument=[tab["instrument_name"][jj]],
                filter=[tab["filters"][jj]],
            )).set_index(["objectid", "label", "filter", "mission"])
            df_spec.append(dfsingle)

        if delete_downloaded_data:
            shutil.rmtree(this_data_dir)

    return df_spec
