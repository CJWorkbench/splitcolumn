import numpy as np
import pandas as pd
import re


MaxNResultColumns = 100  # prevent out-of-memory


def _migrate_params_v0_to_v1(params):
    """
    v0: no 'method'

    v1: 'method' is an index into "Delimiter|Chars from Left|Chars from Right"
    """
    return {**params, 'method': 0, 'numchars': 1}


def _migrate_params_v1_to_v2(params):
    """
    v1: 'method' is an index into "Delimiter|Chars from Left|Chars from Right"

    v2: 'method' is one of 'delimiter', 'left', 'right'
    """
    return {**params, 'method': ['delimiter','left','right'][params['method']]}


def migrate_params(params):
    if 'method' not in params:
        params = _migrate_params_v0_to_v1(params)

    if isinstance(params['method'], int):
        params = _migrate_params_v1_to_v2(params)

    return params


# Take a string column, split according to user's chosen method
# Returns a table (multiple columns) of string category type, or None meaning NOP
def dosplit(coldata, *, method: str, delimiter: str, numchars: int):
    if method == 'delimiter':
        # pandas does not split by string (despite what its docs say). It
        # splits by regex. Turn our string into a regex so we can split it.
        return coldata.str.split(re.escape(delimiter), expand=True,
                                 n=MaxNResultColumns)
    else:
        # otherwise, split off left or right chars
        if numchars <= 0:
            return 'Please choose a positive number of characters.'

        if method == 'left':
            return pd.concat([coldata.str[:numchars], coldata.str[numchars:]], axis=1)
        else:
            # 'Characters from right
            return pd.concat([coldata.str[:-numchars], coldata.str[-numchars:]], axis=1)


def render(table, params):
    colname = params.pop('column')

    if colname == '':
        return table

    if params['method'] == 'delimiter' and not params['delimiter']:
        return table   # Empty delimiter, NOP

    newcols = dosplit(table[colname], **params)

    # NOP if input is bad or we didn't find the delimiter anywhere
    if isinstance(newcols, str):
        # Invalid form input
        return newcols
    if len(newcols.columns) == 1:
        # We didn't find the delimiter anywhere
        return table

    # preserve category-ness
    if hasattr(table[colname], 'cat'):
        newcols = newcols.astype('category')

    # Number the split columns
    newcols.columns = [colname + ' ' + str(x+1)
                       for x in range(len(newcols.columns))]

    if len(table.columns) > 1:
        # glue before, split, and after columns together
        # TODO test this
        colloc = table.columns.get_loc(colname)
        start = table.iloc[:, :colloc]
        end = table.iloc[:, colloc+1:]
        return pd.concat([start, newcols, end], axis=1)
    else:
        return newcols
