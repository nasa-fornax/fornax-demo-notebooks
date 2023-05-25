"""Functions to convert a pyarrow table to an astropy table.

All functions were copied directly from LSST's daf_butler, specifically:
https://github.com/lsst/daf_butler/blob/main/python/lsst/daf/butler/formatters/parquet.py.
It would be better to use the LSST package directly, but it requires python >= 3.10
and we need to be less restrictive. Minor modifications have been made below to accommodate.
"""
import re
from typing import Optional

import numpy as np
import pyarrow as pa
from astropy.table import Table


def arrow_to_astropy(arrow_table: pa.Table) -> Table:
    """Convert a pyarrow table to an `astropy.Table`.

    Parameters
    ----------
    arrow_table : `pyarrow.Table`
        Input arrow table to convert. If the table has astropy unit
        metadata in the schema it will be used in the construction
        of the ``astropy.Table``.

    Returns
    -------
    table : `astropy.Table`
        Converted astropy table.
    """
    from astropy.table import Table

    astropy_table = Table(arrow_to_numpy_dict(arrow_table))

    metadata = arrow_table.schema.metadata if arrow_table.schema.metadata is not None else {}

    _apply_astropy_metadata(astropy_table, metadata)

    return astropy_table


def arrow_to_numpy_dict(arrow_table: pa.Table) -> dict[str, np.ndarray]:
    """Convert a pyarrow table to a dict of numpy arrays.

    Parameters
    ----------
    arrow_table : `pyarrow.Table`
        Input arrow table.

    Returns
    -------
    numpy_dict : `dict` [`str`, `numpy.ndarray`]
        Dict with keys as the column names, values as the arrays.
    """
    import numpy as np

    schema = arrow_table.schema
    metadata = schema.metadata if schema.metadata is not None else {}

    numpy_dict = {}

    for name in schema.names:
        t = schema.field(name).type

        if arrow_table[name].null_count == 0:
            # Regular non-masked column
            col = arrow_table[name].to_numpy()
        else:
            # For a masked column, we need to ask arrow to fill the null
            # values with an appropriately typed value before conversion.
            # Then we apply the mask to get a masked array of the correct type.

            if t in (pa.string(), pa.binary()):
                dummy = ""
            else:
                dummy = t.to_pandas_dtype()(0)

            col = np.ma.masked_array(
                data=arrow_table[name].fill_null(dummy).to_numpy(),
                mask=arrow_table[name].is_null().to_numpy(),
            )

        if t in (pa.string(), pa.binary()):
            col = col.astype(_arrow_string_to_numpy_dtype(schema, name, col))
        elif isinstance(t, pa.FixedSizeListType):
            if len(col) > 0:
                col = np.stack(col)
            else:
                # this is an empty column, and needs to be coerced to type.
                col = col.astype(t.value_type.to_pandas_dtype())

            shape = _multidim_shape_from_metadata(metadata, t.list_size, name)
            col = col.reshape((len(arrow_table), *shape))

        numpy_dict[name] = col

    return numpy_dict


def _arrow_string_to_numpy_dtype(
    schema: pa.Schema, name: str, numpy_column: Optional[np.ndarray] = None, default_length: int = 10
) -> str:
    """Get the numpy dtype string associated with an arrow column.

    Parameters
    ----------
    schema : `pyarrow.Schema`
        Arrow table schema.
    name : `str`
        Column name.
    numpy_column : `numpy.ndarray`, optional
        Column to determine numpy string dtype.
    default_length : `int`, optional
        Default string length when not in metadata or can be inferred
        from column.

    Returns
    -------
    dtype_str : `str`
        Numpy dtype string.
    """
    # Special-case for string and binary columns
    md_name = f"lsst::arrow::len::{name}"
    strlen = default_length
    metadata = schema.metadata if schema.metadata is not None else {}
    if (encoded := md_name.encode("UTF-8")) in metadata:
        # String/bytes length from header.
        strlen = int(schema.metadata[encoded])
    elif numpy_column is not None:
        if len(numpy_column) > 0:
            strlen = max(len(row) for row in numpy_column)

    dtype = f"U{strlen}" if schema.field(name).type == pa.string() else f"|S{strlen}"

    return dtype


def _multidim_shape_from_metadata(metadata: dict[bytes, bytes], list_size: int, name: str) -> tuple[int, ...]:
    """Retrieve the shape from the metadata, if available.

    Parameters
    ----------
    metadata : `dict` [`bytes`, `bytes`]
        Metadata dictionary.
    list_size : `int`
        Size of the list datatype.
    name : `str`
        Column name.

    Returns
    -------
    shape : `tuple` [`int`]
        Shape associated with the column.

    Raises
    ------
    RuntimeError
        Raised if metadata is found but has incorrect format.
    """
    md_name = f"lsst::arrow::shape::{name}"
    if (encoded := md_name.encode("UTF-8")) in metadata:
        groups = re.search(r"\((.*)\)", metadata[encoded].decode("UTF-8"))
        if groups is None:
            raise RuntimeError("Illegal value found in metadata.")
        shape = tuple(int(x) for x in groups[1].split(",") if x != "")
    else:
        shape = (list_size,)

    return shape


def _apply_astropy_metadata(astropy_table: Table, metadata: dict) -> None:
    """Apply any astropy metadata from the schema metadata.

    Parameters
    ----------
    astropy_table : `astropy.table.Table`
        Table to apply metadata.
    metadata : `dict` [`bytes`]
        Metadata dict.
    """
    from astropy.table import meta

    meta_yaml = metadata.get(b"table_meta_yaml", None)
    if meta_yaml:
        meta_yaml = meta_yaml.decode("UTF8").split("\n")
        meta_hdr = meta.get_header_from_yaml(meta_yaml)

        # Set description, format, unit, meta from the column
        # metadata that was serialized with the table.
        header_cols = {x["name"]: x for x in meta_hdr["datatype"]}
        for col in astropy_table.columns.values():
            for attr in ("description", "format", "unit", "meta"):
                if attr in header_cols[col.name]:
                    setattr(col, attr, header_cols[col.name][attr])

        if "meta" in meta_hdr:
            astropy_table.meta.update(meta_hdr["meta"])
