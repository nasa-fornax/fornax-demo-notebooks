import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from astropy.table import Table

HELPERS_DIR = Path(__file__).parent
sys.path.append(str(HELPERS_DIR.parent))  # put code_src dir on the path

# Lazy-load all other imports to avoid depending on modules that will not actually be used.

ARCHIVE_NAMES = {
    "all": ["Gaia", "HCV", "HEASARC", "IceCube", "PanSTARRS", "TESS_Kepler", "WISE", "ZTF"],
    "scaled": ["Gaia", "HEASARC", "IceCube", "WISE", "ZTF"],  # these are expected to run successfully at scale
}
DEFAULTS = {
    "run_id": "my-run",
    "get_samples": ["Yang"],
    "consolidate_nearby_objects": True,
    "overwrite_existing_sample": True,
    "archives": ARCHIVE_NAMES["all"],
    "overwrite_existing_lightcurves": True,
    "use_yaml": False,
    "yaml_filename": "kwargs.yml",
    "sample_filename": "object_sample.ecsv",
    "parquet_dataset_name": "lightcurves.parquet",
}


def run(*, build, **kwargs_dict):
    """Run the light_curve_generator step indicated by `build`.

    Parameters
    ==========

    build : str
        Which step to run. Generally either "sample" or "lightcurves". Can also be either "kwargs"
        or any of the `kwargs_dict` keys, and then the full set of kwargs is built from `kwargs_dict`
        and either the whole dictionary is returned (if "kwargs") or the value of this key is returned.

    kwargs_dict
        Key/value pairs for the build function. This can include any key in the dict
        `helpers.scale_up.DEFAULTS` plus "archive". These are described below.

        run_id : str
            ID for this run. This is used to name the output subdirectory ("base_dir") where the
            scale_up helper will read/write files.

        get_samples : list or dict[dict]
            Names of get_<name>_sample functions from which to gather the object sample.
            To send keyword arguments for any of the named functions, use a dict with key=name
             value=dict of keyword arguments for the named function). Defaults will be
            used for any parameter not provided.

        consolidate_nearby_objects : bool
            Whether to consolidate nearby objects in the sample. Passed to the clean_sample function.

        overwrite_existing_sample : bool
            Whether to overwrite an existing .ecsv file. If false and the file exists, the sample will simply
            be loaded from the file and returned.

        archive : str
            Name of a <name>_get_lightcurves archive function to call when building light curves.

        archives : list or dict[dict]
            Names of <name>_get_lightcurves functions. Use a dict (key=name, value=dict of kwargs for the named
            function) to override defaults for the named function. **Note that this does not determine which
            archives are actually called** (unlike `get_samples` with sample names). This is because the
            scale_up helper can only run one function at a time, so a single archive name must be passed separately
            when building light curves.

        overwrite_existing_lightcurves : bool
            Whether to overwrite an existing .parquet file. If false and the file exists, the light curve
            data will simply be loaded from the file and returned.

        use_yaml : bool
            Whether to load additional kwargs from a yaml file.

        yaml_filename : str
            Name of the yaml to read/write, relative to the base_dir.

        sample_filename: str
            Name of the `sample_table` .ecsv file to read/write, relative to the base_dir.

        parquet_dataset_name : str
            Name of the directory to read/write the parquet dataset, relative to the base_dir.
            The dataset will contain one .parquet file for each archive that returned light curve data.
    """
    my_kwargs_dict = _construct_kwargs_dict(**kwargs_dict)

    if build == "sample":
        return _build_sample(**my_kwargs_dict)

    if build == "lightcurves":
        return _build_lightcurves(**my_kwargs_dict)

    return _build_other(keyword=build, **my_kwargs_dict)


# ---- build functions ---- #


def _build_sample(*, get_samples, consolidate_nearby_objects, sample_file, overwrite_existing_sample, **_):
    """
    Build an AGN (Active Galactic Nuclei) sample using coordinates from different papers.

    This function orchestrates the process of creating an AGN sample by fetching data from various sources,
    cleaning the samples, and consolidating nearby objects. The resulting sample is saved to a specified file.

    Parameters:
    - get_samples (dict): A dictionary where keys are sample names and values are dictionaries of 
    keyword arguments to be passed to the corresponding sample retrieval functions.
    - consolidate_nearby_objects (bool): If True, nearby objects will be consolidated.
    - sample_file (pathlib.Path): The file path where the sample will be saved.
    - overwrite_existing_sample (bool): If True, the existing sample file will be overwritten if it exists.

    Returns:
    - astropy.table.Table: The resulting sample table containing the AGN objects.

    Notes:
    - If the sample file already exists and `overwrite_existing_sample` is False, the function will read 
    and return the existing sample without rebuilding it.
    - The function prints progress messages to the console to indicate the current state of the sample 
    building process.
    """
    _init_worker(job_name="build=sample")

    # if a sample file currently exists and the user elected not to overwrite, just return it
    if sample_file.is_file() and not overwrite_existing_sample:
        print(f"Using existing object sample at: {sample_file}", flush=True)
        return Table.read(sample_file, format="ascii.ecsv")

    # else continue fetching the sample
    print(f"Building object sample from: {list(get_samples.keys())}", flush=True)

    import sample_selection

    # list of tuples. tuple contains: (get-sample function, kwargs dict for that function)
    get_sample_functions = [
        (getattr(sample_selection, f"get_{name}_sample"), kwargs) for name, kwargs in get_samples.items()
    ]

    # iterate over the functions and get the samples
    coords, labels = [], []
    for get_sample_fnc, kwargs in get_sample_functions:
        get_sample_fnc(coords, labels, **kwargs)

    # create an astropy Table of objects
    sample_table = sample_selection.clean_sample(coords, labels, consolidate_nearby_objects=consolidate_nearby_objects)

    # save and return the Table
    sample_table.write(sample_file, format="ascii.ecsv", overwrite=True)
    print(f"Object sample saved to: {sample_file}")
    print(_now(), flush=True)
    return sample_table


def _build_lightcurves(*, archive, archive_kwargs, sample_file, parquet_dir, overwrite_existing_lightcurves, **_):
    """
    Fetch data from the specified archive and build light curves for objects in the given sample file.

    This function retrieves light curve data for objects listed in a sample file from a specified archive.
    The resulting light curves are saved to a Parquet file in a specified directory. If the Parquet file 
    already exists and overwriting is not requested, the existing file is used.

    Parameters:
    - archive (str): The name of the archive to fetch data from. This should correspond to a module containing 
      functions to interact with the archive.
    - archive_kwargs (dict): A dictionary of keyword arguments to be passed to the archive's light curve 
    retrieval function.
    - sample_file (pathlib.Path): The path to the sample file containing the objects for which light curves 
    are to be retrieved.
    - parquet_dir (pathlib.Path): The directory where the Parquet file containing the light curves will be saved.
    - overwrite_existing_lightcurves (bool): If True, the existing Parquet file will be overwritten if it exists.

    Returns:
    - MultiIndexDFObject: The resulting light curve data, or None if no data was returned from the archive.

    Notes:
    - If the Parquet file already exists and `overwrite_existing_lightcurves` is False, the function will 
    read and return the existing data without fetching new data from the archive.
    - The function prints progress messages to the console to indicate the current state of the light curve 
    building process.
    """
    _init_worker(job_name=f"build=lightcurves, archive={archive}")
    parquet_filepath = parquet_dir / f"archive={archive}" / "part0.snappy.parquet"

    # if a sample file currently exists and the user elected not to overwrite, just return it
    if parquet_filepath.is_file() and not overwrite_existing_lightcurves:
        from data_structures import MultiIndexDFObject

        print(f"Using existing light curve data at: {parquet_filepath}", flush=True)
        return MultiIndexDFObject(data=pd.read_parquet(parquet_filepath))

    # Load the sample.
    sample_table = Table.read(sample_file, format="ascii.ecsv")

    # Import only the archive module that will actually be used to avoid unnecessary dependencies.
    archive_functions = importlib.import_module(f"{archive}_functions")
    get_lightcurves_fnc = getattr(archive_functions, f"{archive}_get_lightcurves")
    # archive_kwargs = archive_kwargs.get(f"{archive}_get_lightcurves", {})

    # Query the archive and load light curves.
    lightcurve_df = get_lightcurves_fnc(sample_table, **archive_kwargs)

    # Save and return the light curve data or tell the user there is no data.
    if len(lightcurve_df.data.index) == 0:
        print(f"No light curve data was returned from {archive}.")
        return
    parquet_filepath.parent.mkdir(parents=True, exist_ok=True)
    lightcurve_df.data.to_parquet(parquet_filepath)
    print(f"Light curves saved to:\n\tparquet_dir={parquet_dir}\n\tfile={parquet_filepath.relative_to(parquet_dir)}")
    print(_now(), flush=True)
    return lightcurve_df


def _build_other(keyword, **kwargs_dict):
    """
    Process a keyword and return its corresponding value from the provided dictionary or predefined constants.

    This function handles specific keywords and returns their associated values. It also has special handling for 
    keywords that end with "+" or "+l" to optionally print the values in a specific format.

    Parameters:
    - keyword (str): The keyword to process. If the keyword ends with "+", the value will be printed. 
      If it ends with "+l", the value will be printed as a space-separated list.
    - kwargs_dict (dict): A dictionary containing keyword-value pairs. These values are used if the keyword 
    does not match predefined constants.

    Returns:
    - The value associated with the keyword. The exact type depends on the keyword and the value in the 
    dictionary or predefined constants.

    Notes:
    - If the keyword is "kwargs", the function will return the entire `kwargs_dict`.
    - If the keyword ends with "+", the function will print the value before returning it.
    - If the keyword ends with "+l", the function will print the value as a space-separated list before 
    returning it.
    - The predefined constants `ARCHIVE_NAMES` are used for keywords "archive_names_all" and 
    "archive_names_scaled".
    """

    if keyword == "kwargs":
        return kwargs_dict

    # if this was called from the command line, we need to print the value so it can be captured by the script
    # this is indicated by a "+" flag appended to the keyword
    print_scalar, print_list = keyword.endswith("+"), keyword.endswith("+l")
    my_keyword = keyword.removesuffix("+l").removesuffix("+")

    # get the keyword value
    if my_keyword in ["archive_names_all", "archive_names_scaled"]:
        value = ARCHIVE_NAMES[my_keyword.split("_")[-1]]
    else:
        value = kwargs_dict[my_keyword]

    if print_scalar:
        print(value)
    if print_list:
        print(" ".join(value))

    return value


# ---- construct kwargs ---- #


def _construct_kwargs_dict(**kwargs_dict):
    """
    Construct a complete kwargs dictionary by combining default values, YAML configuration (if requested), 
    and provided keyword arguments, with precedence in that order.

    This function merges default values, optional YAML configurations, and user-provided keyword arguments 
    into a single dictionary. It ensures that all necessary keys are present and sets up paths and directories 
    for the run.

    Parameters:
    - kwargs_dict (dict): A dictionary of keyword arguments provided by the user. These values take the highest 
      precedence and will overwrite defaults and YAML configurations.

    Returns:
    - dict: A dictionary containing the combined and final keyword arguments for the run, sorted by key.

    Notes:
    - Default values are defined in the `DEFAULTS` dictionary.
    - If `use_yaml` is True in `kwargs_dict` or in the defaults, the function will load additional keyword 
      arguments from a YAML file specified by `yaml_filename`.
    - Keys "get_samples" and "archives" are deep updated to combine nested dictionaries.
    - The function sets up various path-related keyword arguments, including `base_dir`, `logs_dir`, 
      `sample_file`, `parquet_dir`, and `yaml_file`.
    - The base directory is created if it does not already exist.
    """
    run_id = kwargs_dict.get("run_id", DEFAULTS["run_id"])
    base_dir = HELPERS_DIR.parent.parent / f"output/lightcurves-{run_id}"

    # load kwargs from yaml if requested
    yaml_file = base_dir / kwargs_dict.get("yaml_filename", DEFAULTS["yaml_filename"])
    my_kwargs = _load_yaml(yaml_file) if kwargs_dict.get("use_yaml", DEFAULTS["use_yaml"]) else {}

    # update with kwargs_dict. both may contain dict values for the following keys, which need deep updates.
    for key in ["get_samples", "archives"]:
        my_kwargs[key] = _deep_update_kwargs_group(key, my_kwargs.pop(key, []), kwargs_dict.pop(key, []))
    my_kwargs.update(kwargs_dict)  # update with any additional keys in kwargs_dict

    # add any defaults that are still missing
    for key in set(DEFAULTS) - set(my_kwargs):
        my_kwargs[key] = DEFAULTS[key]

    # if a single archive is requested, lower-case it and pull out the right set of kwargs for _build_lightcurves
    if my_kwargs.get("archive"):
        my_kwargs["archive"] = my_kwargs["archive"].lower()
        my_kwargs["archive_kwargs"] = my_kwargs["archives"].get(my_kwargs["archive"], {})

    # set path kwargs and make the base dir if needed
    my_kwargs["run_id"] = run_id
    my_kwargs["base_dir"] = base_dir
    my_kwargs["logs_dir"] = base_dir / "logs"
    my_kwargs["sample_file"] = base_dir / my_kwargs["sample_filename"]
    my_kwargs["parquet_dir"] = base_dir / my_kwargs["parquet_dataset_name"]
    my_kwargs["yaml_file"] = yaml_file
    base_dir.mkdir(parents=True, exist_ok=True)

    # sort by key and return
    return {key: my_kwargs[key] for key in sorted(my_kwargs)}


def _deep_update_kwargs_group(key, group_a, group_b):
    """
    Deeply update a group of keyword arguments by combining two groups.

    This function merges two groups of keyword arguments, with `group_b` values taking precedence
    over `group_a` values. Both groups may be either lists or dictionaries.

    Parameters:
    - key (str): The key associated with the groups in the `DEFAULTS` dictionary.
    - group_a (list or dict): The first group of keyword arguments.
    - group_b (list or dict): The second group of keyword arguments, which takes precedence over `group_a`.

    Returns:
    - dict: A dictionary with the combined keyword arguments.

    Notes:
    - If both groups are empty, the function returns the default values for the given key.
    - The function converts both groups to dictionaries with keys as names and values as dictionaries of 
    keyword arguments.
    - It deeply updates the individual name/kwarg pairs.
    """
    # if both groups are empty, just return defaults
    if len(group_a) == 0 and len(group_b) == 0:
        return _kwargs_list_to_dict(DEFAULTS[key])

    # these groups may be either lists or dicts.
    # turn them both into dicts with key = name, value = dict of kwargs for name (empty dict if none supplied)
    my_group_a, my_group_b = _kwargs_list_to_dict(group_a), _kwargs_list_to_dict(group_b)

    # update a with b. first, descend one level to update individual name/kwarg pairs.
    for name in my_group_a:
        my_group_a[name].update(my_group_b.pop(name, {}))
    # add any keys from b that were not in a
    my_group_a.update(my_group_b)

    return my_group_a


def _kwargs_list_to_dict(list_or_dict):
    """
    Convert a list or dictionary of keyword arguments to a dictionary with lowercase keys.

    This function transforms a list of names or a dictionary of name/kwargs pairs into a 
    dictionary with lowercase keys and corresponding values.

    Parameters:
    - list_or_dict (list or dict): A list of names or a dictionary of name/kwargs pairs.

    Returns:
    - dict: A dictionary with lowercase keys and corresponding values.

    Notes:
    - If the input is a list, the function generates a dictionary with names as keys and empty dictionaries 
    as values.
    - If the input is already a dictionary, the function converts the keys to lowercase.
    """    
    if isinstance(list_or_dict, list):
        return {name.lower(): {} for name in list_or_dict}
    return {name.lower(): kwargs for name, kwargs in list_or_dict.items()}


# ---- other utils ---- #


def _init_worker(job_name="worker"):
    """
    Run generic start-up tasks for a job.

    This function performs initial setup tasks for a job, including printing the process ID
    for the current worker.

    Parameters:
    - job_name (str): The name of the job. Default is "worker".

    Notes:
    - The function prints the current date and time, process ID, and job name to the console.
    """
    # print the Process ID for the current worker so it can be killed if needed
    print(f"{_now()} | [pid={os.getpid()}] Starting {job_name}", flush=True)


def _now():
    """
    Return the current datetime as a string in the format '%Y/%m/%d %H:%M:%S %Z'.

    This function returns the current date and time in a specific string format, with the time zone included.

    Returns:
    - str: The current date and time as a formatted string.

    Example:
    >>> import dateutil
    >>> now = dateutil.parser.parse(_now())
    """
    date_format = "%Y/%m/%d %H:%M:%S %Z"
    return datetime.now(timezone.utc).strftime(date_format)


def _load_yaml(yaml_file):
    """
    Load a YAML file and return its contents as a dictionary.

    This function reads a YAML file and parses its contents into a dictionary.

    Parameters:
    - yaml_file (pathlib.Path): The path to the YAML file to load.

    Returns:
    - dict: The contents of the YAML file as a dictionary.

    Notes:
    - The function uses `yaml.safe_load` to parse the YAML file.
    """
    with open(yaml_file, "r") as fin:
        yaml_dict = yaml.safe_load(fin)
    return yaml_dict


def write_kwargs_to_yaml(**kwargs_dict) -> None:
    """
    Write the provided keyword arguments dictionary to a YAML file.

    This function writes the contents of `kwargs_dict` to a YAML file in the run's base directory.

    Parameters:
    - kwargs_dict (dict): The dictionary of keyword arguments to write to the YAML file.

    Returns:
    - None

    Notes:
    - The YAML file path is determined by the `yaml_file` key in `kwargs_dict`.
    - The function prints a message indicating the path to the written YAML file.
    """
    yaml_path = run(build="yaml_file", **kwargs_dict)
    with open(yaml_path, "w") as fout:
        yaml.safe_dump(kwargs_dict, fout)
    print(f"kwargs written to {yaml_path}")


# ---- helpers for __name__ == "__main__" ---- #


def _argparser():
    """
    Create and return an argument parser for command-line arguments.

    This function sets up an argument parser with several options for building samples, 
    light curves, and handling keyword arguments.

    Returns:
    - argparse.ArgumentParser: The configured argument parser.

    Notes:
    - The parser includes options for `--build`, `--kwargs_dict`, `--kwargs_json`, and `--archive`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        type=str,
        default="sample",
        help="Either 'sample', 'lightcurves', or a kwargs key.",
    )
    parser.add_argument(
        "--kwargs_dict",
        type=str,
        default=list(),
        nargs="*",
        help="Keyword arguments for the run. Input as a list of strings 'key=value'.",
    )
    parser.add_argument(
        "--kwargs_json",
        type=json.loads,
        default=r"{}",
        help="Kwargs as a json string, to be added to kwargs_dict.",
    )
    parser.add_argument(
        "--archive",
        type=str,
        default=None,
        help="Archive name to query for light curves, to be added to kwargs_dict.",
    )
    return parser


def _parse_args(args_list):
    """
    Parse command-line arguments into a build type and keyword arguments dictionary.

    This function processes command-line arguments to determine the build type and 
    construct a combined dictionary of keyword arguments.

    Parameters:
    - args_list (list): A list of command-line arguments to parse.

    Returns:
    - tuple: A tuple containing the build type (str) and the combined keyword arguments dictionary (dict).

    Notes:
    - The function starts with `kwargs_json` and updates it with `kwargs_dict`.
    - It converts `key=value` pairs to a dictionary and handles boolean values.
    - Keys "get_samples" and "archives" are deeply updated.
    - If an archive is provided, it is added to the keyword arguments dictionary.
    """
    args = _argparser().parse_args(args_list)

    # start with kwargs_json, then update
    my_kwargs_dict = args.kwargs_json

    # parse args.kwargs_dict 'key=value' pairs into a proper dict and convert true/false to bool
    bool_map = {"true": True, "false": False}
    args_kwargs_dict = {
        key: bool_map.get(val.lower(), val)
        for (key, val) in dict(kwarg.split("=") for kwarg in args.kwargs_dict).items()
        # for (key, val) in {kwarg.split("=")[0]: kwarg.split("=")[1] for kwarg in args.kwargs_dict}.items()
    }

    # update my_kwargs_dict with args_kwargs_dict
    # both may contain dict values for the following keys, which need deep updates.
    for key in list(k for k in ["get_samples", "archives"] if k in my_kwargs_dict):
        my_kwargs_dict[key] = _deep_update_kwargs_group(key, my_kwargs_dict.pop(key), args_kwargs_dict.pop(key, []))
    my_kwargs_dict.update(args_kwargs_dict)  # update with any additional keys in args_kwargs_dict

    # add the archive, if provided
    if args.archive:
        my_kwargs_dict["archive"] = args.archive

    return args.build, my_kwargs_dict


if __name__ == "__main__":
    build, kwargs_dict = _parse_args(sys.argv[1:])
    run(build=build, **kwargs_dict)
