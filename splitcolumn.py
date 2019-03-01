import pandas as pd
import numpy as np



def migrate_params(params):
    # go from 'method' as integer to 'method' as string 

    if isinstance(params['method'], int):
        params = dict(params)  # copy
        params['method'] = ['delimiter','left','right'][params['method']]

    return params


# Take a string column, split according to user's chosen method
# Returns a table (multiple columns) of string category type, or None meaning NOP
def dosplit(coldata, params):
    # v1 params or method = Delimiter
    if 'method' not in params or params['method'] == 'delimiter':
        delim = params['delimiter']
        return coldata.str.split(delim, expand=True)

    # otherwise, split off left or right chars 
    numchars = params['numchars']
    if numchars<=0:
        return None    # NOP

    if params['method'] == 'left':
        return pd.concat([coldata.str[:numchars], coldata.str[numchars:]], axis=1)
    else:   
        # 'Characters from right
        return pd.concat([coldata.str[:-numchars], coldata.str[-numchars:]], axis=1)


# Convert float column to string, remove decimal if float is a whole number
# and fill nulls with '' to avoid 'NaN'
# (To be replaced when we have type format utilities)
def convert_float_to_str(col):
    ints = col[col % 1 == 0].astype(int)
    col = col.fillna('').astype(str)
    col.update(ints.astype(str))
    return col


# Converts all possible column types to simple strings, including categorical
def col_to_str(col):
    if col.dtype.name == 'category':
        if col.cat.categories.dtype == float:
            # Floats stored as categories
            return convert_float_to_str(col.astype(float))

        elif col.cat.categories.dtype == object: 
            # Strings stored as categories
            copy = col.copy()
            if '' not in copy.cat.categories:
                copy.cat.add_categories([''], inplace=True) # Must add '' to category if want to remove NaN
            return copy.fillna('')

        else:
            # Some other categorical type. Cast to string.
            return col.astype(str)

    elif col.dtype == float:
        # Floats
        return convert_float_to_str(col)

    else:
        # Everything else, like ints and strings
        return col.astype(str)


def render(table, params):
    colname = params['column']

    if colname == '':
      return table

    using_delimiter = ('method' not in params) or (params['method']=='delimiter')
    if using_delimiter and params['delimiter']=='':
        return table   # Empty delimiter, NOP

    coldata = col_to_str(table[colname])
    newcols = dosplit(coldata, params)

    # NOP if input is bad or we didn't find the delimiter anywhere
    if (newcols is None) or (len(newcols.columns) == 1):
      return table

    # preserve category-ness (cat string, cat float, etc.)
    if table[colname].dtype.name == 'category':
        if using_delimiter:
            newcols = newcols.astype('category')
        else:
            # We cannot convert to categories after taking left/right chars in Pandas 0.23.x
            # Remove this conditional after upgrading Workbench to 0.24
            # Minimal reproduction of bug:
            # import pandas as pd
            # a = pd.DataFrame(['The string is best','When in silence'])
            # b = a[0]
            # c = pd.concat([b.str[:5], b.str[5:]], axis=1)
            # c.astype('category')
            # --> Fatal Python error: Cannot recover from stack overflow.
            pass

    # Number the split columns
    newcols.columns = [colname + ' ' + str(x+1) for x in range(len(newcols.columns))]

    if len(table.columns)>1:
        # glue before, split, and after columns together
        colloc = table.columns.get_loc(colname)
        start = table.iloc[:, :colloc]
        end = table.iloc[:, colloc+1:]
        return pd.concat([start, newcols, end], axis=1)
    else:
        return newcols



