from typing import Any, Dict
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from splitcolumn import render, migrate_params


def P(column: str = '', method: str = 'delimiter', delimiter: str = '-',
      numchars: str = 1) -> Dict[str, Any]:
    return {
        'column': column,
        'method': method,
        'delimiter': delimiter,
        'numchars': numchars,
    }


class TestSplitColumns(unittest.TestCase):
    def test_NOP(self):
        """Default parameters means no-op."""
        table = pd.DataFrame({'A': ['1-1', '2-2']})
        result = render(table, P())
        assert_frame_equal(result, pd.DataFrame({'A': ['1-1', '2-2']}))

    def test_split_dot(self):
        # Pandas default is to treat '..' as a regex.
        table = pd.DataFrame({'A': ['abcd..ef', 'bcde..fg']})
        result = render(table, P(column='A', delimiter='..'))
        assert_frame_equal(result, pd.DataFrame({
            'A 1': ['abcd', 'bcde'],
            'A 2': ['ef', 'fg'],
        }))

    def test_split_str(self):
        table = pd.DataFrame({'A': ['a.b', 'b.c'], 'B': [1, 2]})
        result = render(table, P(column='A', delimiter='.'))
        assert_frame_equal(result, pd.DataFrame({
            'A 1': ['a', 'b'],
            'A 2': ['b', 'c'],
            'B': [1, 2],
        }))

    def test_split_one_col(self):
        # We had a crash here, so now we have a test
        table = pd.DataFrame({'A': ['foo666']})
        result = render(table, P(column='A', method='right', numchars=3))
        assert_frame_equal(result,
                           pd.DataFrame({'A 1':['foo'], 'A 2':['666']}))

    def test_split_cat(self):
        table = pd.DataFrame({'A': ['b.c', 'b.c', 'a', '', np.nan]},
                             dtype='category')
        result = render(table, P(column='A', delimiter='.'))
        assert_frame_equal(result, pd.DataFrame({
            'A 1': ['b', 'b', 'a', '', np.nan],
            'A 2': ['c', 'c', np.nan, np.nan, np.nan],
        }, dtype='category'))

    def test_split_empty_cat_by_delimiter(self):
        table = pd.DataFrame({'A': [np.nan]}, dtype=str).astype('category')
        result = render(table, P(column='A', delimiter='.'))
        assert_frame_equal(
            result,
            pd.DataFrame({'A': [np.nan]}, dtype=str).astype('category')
        )

    def test_split_empty_cat_by_numchars(self):
        table = pd.DataFrame({'A': [np.nan]}, dtype=str).astype('category')
        result = render(table, P(column='A', method='left', numchars=2))
        assert_frame_equal(result, pd.DataFrame({
            'A 1': [np.nan],
            'A 2': [np.nan],
        }, dtype=str).astype('category'))

    def test_multiple_splits(self):
        table = pd.DataFrame({'A': ['2019-01-01', '2019-04', '2019', '']})
        result = render(table, P(column='A', delimiter='-'))
        assert_frame_equal(result, pd.DataFrame({
            'A 1': ['2019', '2019', '2019', ''],
            'A 2': ['01', '04', np.nan, np.nan],
            'A 3': ['01', np.nan, np.nan, np.nan],
        }))

    def test_no_delimiter_found(self):
        """Do not rename column when there are no matches."""
        table = pd.DataFrame({'A': ['x']})
        result = render(table, P(column='A', delimiter='!'))
        assert_frame_equal(result, pd.DataFrame({'A': ['x']}))

    def test_split_left(self):
        table = pd.DataFrame({'A': ['abc', 'de', 'f', '', np.nan]})
        result = render(table, P(column='A', method='left', numchars=2))
        assert_frame_equal(result, pd.DataFrame({
            'A 1': ['ab', 'de', 'f', '', np.nan],
            'A 2': ['c', '', '', '', np.nan],
        }))

    def test_split_right(self):
        table = pd.DataFrame({'A': ['abc', 'de', 'f', '', np.nan]})
        result = render(table, P(column='A', method='right', numchars=2))
        assert_frame_equal(result, pd.DataFrame({
            'A 1': ['a', '', '', '', np.nan],
            'A 2': ['bc', 'de', 'f', '', np.nan],
        }))

    def test_split_negative_numchars(self):
        result = render(pd.DataFrame({'A': ['a']}),
                        P(column='A', method='right', numchars=-2))
        self.assertEqual(
            result,
            'Please choose a positive number of characters.'
        )

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
