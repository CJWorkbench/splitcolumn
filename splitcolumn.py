def render(table, params):
  delim = params['delimiter']
  col = params['column']
  
  # the actual split
  newcols =  table[col].str.split(delim, expand=True)
  newcols.columns = [col + ' ' + str(x+1) for x in newcols.columns]
  
  # now glue before, split, and after columns together 
  colloc = table.columns.get_loc(col)
  start = table.iloc[:, :colloc]
  end = table.iloc[:, colloc+1:]
  return pd.concat([start, newcols, end], axis=1)

