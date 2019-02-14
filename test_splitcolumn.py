import unittest
import pandas as pd
import numpy as np
from splitcolumn import render


class TestSplitColumns(unittest.TestCase):

    def setUp(self):
        # Test data includes:
        #  - rows of numeric, string, and categorical types
        #  - zero entries (which should not be removed)
        #  - if either column categorical type, retain caegorical
        self.table = pd.DataFrame([
            ['a.b',     123,    2.1,    'b.c',      2.1,    '2018-08-03',   123],
            ['a.b',     123,    2,      'b.c',      2,      '2018-08-03',   123],
            ['a.b',     123,    2.1,    '',         2.1,    '2018-08-03',   123],
            ['a.b',     123,    2.1,    'b.c',      2.1,    '2018-08-03',   123],
            ['a.b',     123,    None,   None,       None,   '2018-08-03',   123]],
            columns=['stringcol','intcol', 'floatcol','catcol','floatcatcol', 'datecol', 'intcatcol'])

        # Pandas should infer these types anyway, but leave nothing to chance
        self.table['stringcol'] = self.table['stringcol'].astype(str)
        self.table['intcol'] = self.table['intcol'].astype(np.int64)
        self.table['floatcol'] = self.table['floatcol'].astype(np.float64)
        self.table['catcol'] = self.table['catcol'].astype('category')
        self.table['floatcatcol'] = self.table['floatcatcol'].astype('category')
        self.table['datecol'] = self.table['datecol'].astype(str)
        self.table['intcatcol'] = self.table['intcatcol'].astype('category')

    def construct_expected(self, ref, column, rows):
        idx = ref.columns.get_loc(column)
        ref.drop(columns=[column], inplace=True)
        for offset, col in enumerate(rows):
            ref.insert(idx + offset, f'{column} {offset + 1}', col)
        return ref

    def test_NOP(self):
        # should NOP when first applied (no col chosen or empty delim)
        params = {'delimiter': '-', 'column': ''}
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table))

        params = {'column': 'stringcol', 'delimiter': ''}
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table)) 


    def test_split_str(self):
        column = 'stringcol'
        params = {'delimiter': '.', 'column': column}
        out = render(self.table, params)

        # Assert old column removed
        self.assertTrue((set(self.table.columns)-set(out.columns) == set([column])))

        ref = self.construct_expected(self.table, column,
                   [['a']*5,
                    ['b']*5])

        pd.testing.assert_frame_equal(out, ref)

    def test_split_int(self):
        column = 'intcol'
        params = {'delimiter': '2', 'column': column}
        out = render(self.table, params)
        ref = self.table

        # Assert old column removed
        self.assertTrue((set(self.table.columns) - set(out.columns) == set([column])))

        ref = self.construct_expected(self.table, column,
                   [['1']*5,
                    ['3']*5])

        pd.testing.assert_frame_equal(out, ref)

    def test_split_float(self):
        column = 'floatcol'
        params = {'delimiter': '.', 'column': column}
        out = render(self.table, params)
        ref = self.table

        # Assert old column removed
        self.assertTrue((set(self.table.columns) - set(out.columns) == set([column])))

        ref = self.construct_expected(self.table, column,
                   [['2','2','2','2',''],
                    ['1', None, '1', '1', None]])

        pd.testing.assert_frame_equal(out, ref)

    def test_split_cat(self):
        column = 'catcol'
        params = {'delimiter': '.', 'column': column}
        out = render(self.table, params)
        ref = self.table

        # Assert old column removed
        self.assertTrue((set(self.table.columns) - set(out.columns) == set([column])))

        # .replace splits '' into '','nan'
        ref = self.construct_expected(self.table, column,
                   [['b','b','','b',''],
                    ['c','c',np.nan,'c',np.nan]])

        ref['catcol 1'] = ref['catcol 1'].astype('category')
        ref['catcol 2'] = ref['catcol 2'].astype('category')

        pd.testing.assert_frame_equal(out, ref)

        column = 'intcatcol'
        params = {'delimiter': '2', 'column': column}
        out = render(self.table, params)
        ref = self.table

        # Assert old column removed
        self.assertTrue((set(self.table.columns) - set(out.columns) == set([column])))

        ref = self.construct_expected(self.table, column,
                                      [['1'] * 5,
                                       ['3'] * 5])
        ref['intcatcol 1'] = ref['intcatcol 1'].astype('category')
        ref['intcatcol 2'] = ref['intcatcol 2'].astype('category')

        pd.testing.assert_frame_equal(out, ref)

    def test_split_cat_float(self):
        column = 'floatcatcol'
        params = {'delimiter': '.', 'column': column}
        out = render(self.table, params)
        ref = self.table

        # Assert old column removed
        self.assertTrue((set(self.table.columns) - set(out.columns) == set([column])))

        ref = self.construct_expected(self.table, column,
                                      [['2', '2', '2', '2', ''],
                                       ['1', None, '1', '1', None]])

        # Require float categorical type to be preserved. 
        ref['floatcatcol 1'] = ref['floatcatcol 1'].astype('category')
        ref['floatcatcol 2'] = ref['floatcatcol 2'].astype('category')

        pd.testing.assert_frame_equal(out, ref)

    def test_multiple_splits(self):
        column = 'datecol'
        params = {'delimiter': '-', 'column': column}
        out = render(self.table, params)
        ref = self.table

        # Assert old column removed
        self.assertTrue((set(self.table.columns) - set(out.columns) == set([column])))

        ref = self.construct_expected(self.table, column,
                   [['2018']*5,
                    ['08'] * 5,
                    ['03'] * 5])

        pd.testing.assert_frame_equal(out, ref)

    def test_no_del_found(self):
        params = {'delimiter': '!', 'column': 'stringcol'}
        out = render(self.table, params)
        ref = self.table
        self.assertTrue(out.equals(ref))


    def test_split_left(self):
        column = 'stringcol'
        params = {'column': column, 'method': 1, 'numchars':2 }
        out = render(self.table, params)

        # Assert old column removed
        self.assertTrue((set(self.table.columns)-set(out.columns) == set([column])))

        ref = self.construct_expected(self.table, column,
                   [['a.']*5,
                    ['b']*5])

        pd.testing.assert_frame_equal(out, ref)


    def test_split_right(self):
        column = 'stringcol'
        params = {'column': column, 'method': 2, 'numchars':2 }
        out = render(self.table, params)

        # Assert old column removed
        self.assertTrue((set(self.table.columns)-set(out.columns) == set([column])))

        ref = self.construct_expected(self.table, column,
                   [['a']*5,
                    ['.b']*5])

        pd.testing.assert_frame_equal(out, ref)


    def test_split_right_bad_param(self):
        # Negative split, do nothing
        column = 'stringcol'
        params = {'column': column, 'method': 2, 'numchars':-2 }
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table))


    def test_split_right_many_characters(self):
        # if we ask for more characters than there are, we should get all of them plus an empty column
        column = 'stringcol'
        params = {'column': column, 'method': 2, 'numchars':20 }
        out = render(self.table, params)

        # Assert old column removed
        self.assertTrue((set(self.table.columns)-set(out.columns) == set([column])))

        ref = self.construct_expected(self.table, column,
                   [['']*5,
                    ['a.b']*5])

        pd.testing.assert_frame_equal(out, ref)


if __name__ == '__main__':
    unittest.main()


