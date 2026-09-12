"""Unit tests for reference-report helpers; no models, network or inference."""
import unittest
import numpy as np
from whisper_pinned_oracle import difference, first_diff, timestamp_segments


class TokenizerStub:
    def token_to_id(self, text):
        return {'<|0.00|>':100, '<|endoftext|>':99}[text]

    def decode(self, ids, skip_special_tokens=False):
        return ''.join({1:' Hello',2:' world'}[x] for x in ids)


class OracleHelpers(unittest.TestCase):
    def test_first_difference(self):
        self.assertIsNone(first_diff([], []))
        self.assertIsNone(first_diff([1,2], [1,2]))
        self.assertEqual(first_diff([1,2], [1,3]), 1)
        self.assertEqual(first_diff([1], [1,2]), 1)
        self.assertEqual(first_diff([1,2], [1]), 1)

    def test_numerical_summary(self):
        m = difference(np.array([1,2],dtype=np.float32),np.array([1,3],dtype=np.float32))
        self.assertEqual(m['max_abs'],1)
        self.assertEqual(m['mean_abs'],.5)
        for a,b in [(np.zeros(2),np.zeros(3)),(np.array([np.nan]),np.zeros(1)),(np.zeros(1),np.array([np.inf]))]:
            with self.assertRaises(ValueError):difference(a,b)

    def test_timestamp_segments(self):
        t=TokenizerStub()
        self.assertEqual(timestamp_segments([100,1,2,125,99],t),[dict(Start=0,End=.5,Text='Hello world',Tokens=[1,2])])
        self.assertEqual(timestamp_segments([99],t),[])
        for tokens in [[1,125,99],[100,1,99],[100,1,100,99],[100,1,125],[]]:
            with self.assertRaises(ValueError):timestamp_segments(tokens,t)


if __name__ == '__main__':
    unittest.main()
