import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from splitcolumn import render,migrate_params


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
        params = {'column': '', 'method':'delimiter', 'delimiter': '-', 'numchars':1}
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table))

        params = {'column': 'stringcol', 'method':'delimiter', 'delimiter': '', 'numchars':1}
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table)) 

    def test_split_dot(self):
        table = pd.DataFrame({'A': ['abcd..ef', 'bcde..fg']})
        result = render(table, {
            'column': 'A',
            'method': 'delimiter',
            'delimiter': '..', 
            'numchars': 1,
        })
        # Pandas default is to treat '..' as a regex.
        assert_frame_equal(result, pd.DataFrame({
            'A 1': ['abcd', 'bcde'],
            'A 2': ['ef', 'fg'],
        }))

    def test_split_str(self):
        column = 'stringcol'
        params = {'column': column, 'method':'delimiter', 'delimiter': '.', 'numchars':1}
        out = render(self.table, params)

        # Assert old column removed
        self.assertTrue((set(self.table.columns)-set(out.columns) == set([column])))

        ref = self.construct_expected(self.table, column,
                   [['a']*5,
                    ['b']*5])

        pd.testing.assert_frame_equal(out, ref)

    def test_split_int(self):
        column = 'intcol'
        params = {'column': column, 'method':'delimiter', 'delimiter': '2', 'numchars':1}
        out = render(self.table, params)
        ref = self.table

        # Assert old column removed
        self.assertTrue((set(self.table.columns) - set(out.columns) == set([column])))

        ref = self.construct_expected(self.table, column,
                   [['1']*5,
                    ['3']*5])

        pd.testing.assert_frame_equal(out, ref)

    def test_split_one_col(self):
        # We had a crash here, so now we have a test
        table = pd.DataFrame({'A':['foo666']})
        params = {'column': 'A', 'method': 'right', 'delimiter': '2', 'numchars':3 }
        out = render(table, params)
        ref = pd.DataFrame({'A 1':['foo'], 'A 2':['666']})
        pd.testing.assert_frame_equal(out, ref)

    def test_split_float(self):
        column = 'floatcol'
        params = {'column': column, 'method':'delimiter', 'delimiter': '.', 'numchars':1}
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
        params = {'column': column, 'method':'delimiter', 'delimiter': '.', 'numchars':1}
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
        params = {'column': column, 'method':'delimiter', 'delimiter': '2', 'numchars':1}
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
        params = {'column': column, 'method':'delimiter', 'delimiter': '.', 'numchars':1}
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
        params = {'column': column, 'method':'delimiter', 'delimiter': '-', 'numchars':1}
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
        params = {'column': 'stringcol', 'method':'delimiter', 'delimiter': '!', 'numchars':1}
        out = render(self.table, params)
        ref = self.table
        self.assertTrue(out.equals(ref))

    def test_split_left(self):
        column = 'stringcol'
        params = {'column': column, 'method': 'left', 'delimiter':',', 'numchars':2}
        out = render(self.table, params)

        # Assert old column removed
        self.assertTrue((set(self.table.columns)-set(out.columns) == set([column])))

        ref = self.construct_expected(self.table, column,
                   [['a.']*5,
                    ['b']*5])

        pd.testing.assert_frame_equal(out, ref)

    def test_split_right(self):
        column = 'stringcol'
        params = {'column': column, 'method': 'right', 'delimiter':',', 'numchars':2}
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
        params = {'column': column, 'method': 'right', 'delimiter':',', 'numchars':-2 }
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table))

    def test_split_right_many_characters(self):
        # if we ask for more characters than there are, we should get all of them plus an empty column
        column = 'stringcol'
        params = {'column': column, 'method': 'right', 'delimiter':',', 'numchars':20}
        out = render(self.table, params)

        # Assert old column removed
        self.assertTrue((set(self.table.columns)-set(out.columns) == set([column])))

        ref = self.construct_expected(self.table, column,
                   [['']*5,
                    ['a.b']*5])

        pd.testing.assert_frame_equal(out, ref)

    def test_migrate_params_v0_to_v1(self):
        # force method=delimiter
        self.assertEqual(
            migrate_params({
                'column':'foo',
                'delimiter':',',
            }),
            {
                'column': 'foo',
                'method': 'delimiter',
                'delimiter': ',',
                'numchars': 1,
            }
        )
    def test_migrate_params_v1_to_v2(self):
        # convert method=int to method=str
        self.assertEqual(
            migrate_params({
                'column': 'foo',
                'method': 1,
                'delimiter': ',',
                'numchars': 3,
            }),
            {
                'column': 'foo',
                'method': 'left',
                'delimiter': ',',
                'numchars': 3,
            }
        )

    def test_migrate_params_v2(self):
        self.assertEqual(
            migrate_params({
                'column': 'foo',
                'method': 'right',
                'delimiter': ',',
                'numchars': 1,
            }),
            {
                'column':'foo',
                'method':'right',
                'delimiter':',',
                'numchars':1,
            }
        )


if __name__ == '__main__':
    unittest.main()
