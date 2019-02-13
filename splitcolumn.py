import pandas as pd
import numpy as np


# Take a string column, split according to user's chosen method
# Returns a table (multiple columns) of string category type
def dosplit(coldata, params):
    if 'method' not in params or params['method'] == 'Delimiter':
        # v1 params, always split on delimiter
        delim = params['delimiter']
        return coldata.str.split(delim, expand=True)

    numchars = params['numchars']

    if params['method'] == 'Characters from left':
        return col 
    else:   # 'Characters from right
        return col

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
        # Everything else (e.g. normal, non categorical strings)
        return col.astype(str)


def render(table, params):
    colname = params['column']

    if colname == '' or params['delimiter'] == '':
      return table

    coldata = col_to_str(table[colname])
    newcols = dosplit(coldata, params)

    # NOP if we didn't find the delimiter anywhere
    if len(newcols.columns) == 1:
      return table

    # preserve category-ness (cat string, cat float, etc.)
    if table[colname].dtype.name == 'category':
        newcols = newcols.astype('category')

    # Number the split columns
    newcols.columns = [colname + ' ' + str(x+1) for x in newcols.columns]

    # glue before, split, and after columns together
    colloc = table.columns.get_loc(colname)
    start = table.iloc[:, :colloc]
    end = table.iloc[:, colloc+1:]
    return pd.concat([start, newcols, end], axis=1)




