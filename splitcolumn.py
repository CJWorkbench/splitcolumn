def render(table, params):
    delim = params['delimiter']
    col = params['column']

    if col == '' or delim == '':
      return table

    # the actual split, output to str and cat dtypes for now
    if table[col].dtype.name == 'category':
        if table[col].cat.categories.dtype == float:
            newcols = (convert_float(table[col].astype(float), delim)).astype('category')
        elif table[col].cat.categories.dtype == object:
            # Must add '' to category if want to remove NaN
            copy = table[col].copy()
            if '' not in copy.cat.categories:
                copy.cat.add_categories([''], inplace=True)
            newcols = copy.fillna('').str.split(delim, expand=True).astype('category')
        else:
            newcols = table[col].astype(str).str.split(delim, expand=True).astype('category')
    # fill nulls with '' to avoid 'NaN'
    elif table[col].dtype == float:
        newcols = convert_float(table[col], delim)

    else:
        newcols = table[col].astype(str).str.split(delim, expand=True)

    # we didn't find the delimiter anywhere
    if len(newcols.columns) == 1:
      return table

    newcols.columns = [col + ' ' + str(x+1) for x in newcols.columns]

    # now glue before, split, and after columns together
    colloc = table.columns.get_loc(col)
    start = table.iloc[:, :colloc]
    end = table.iloc[:, colloc+1:]
    return pd.concat([start, newcols, end], axis=1)

# Special case: remove decimal if float is a whole number
def convert_float(col, delim):
    # Special case: remove decimal if float is a whole number
    ints = col[col % 1 == 0].astype(int)
    col = col.fillna('').astype(str)
    col.update(ints.astype(str))
    return col.str.split(delim, expand=True)

