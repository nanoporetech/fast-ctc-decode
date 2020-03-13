#!/usr/env/bin python3

import numpy as np
from unittest import TestCase, main
from fast_ctc_decode import beam_search


class Tests(TestCase):

    def setUp(self):
        self.beam_size = 5
        self.alphabet = "NACGT"
        self.beam_cut_threshold = 0.1
        self.probs = self.get_random_data()

    def get_random_data(self, samples=100):
        return np.random.rand(samples, len(self.alphabet)).astype(np.float32)

    def test_beam_search(self):
        """ simple beam search test with the canonical alphabet"""
        seq, path = beam_search(self.probs, self.alphabet, self.beam_size, self.beam_cut_threshold)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_beam_search_alphabet(self):
        """ simple beam search test with different alphabet"""
        seq, path = beam_search(self.probs, "NRUST", self.beam_size, self.beam_cut_threshold)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_zero_beam_size(self):
        """ simple beam search test with zero beam size"""
        with self.assertRaises(ValueError):
            beam_search(self.probs, self.alphabet, 0, self.beam_cut_threshold)

    def test_zero_beam_cut_threshold(self):
        """ simple beam search test with beam cut threshold of 0.0"""
        seq, path = beam_search(self.probs, self.alphabet, self.beam_size, 0.0)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_negative_beam_cut_threshold(self):
        """ simple beam search test with beam cut threshold below 0.0"""
        seq, path = beam_search(self.probs, self.alphabet, self.beam_size, -0.1)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_max_beam_cut_threshold(self):
        """ simple beam search test with beam cut threshold of 1.0"""
        seq, path = beam_search(self.probs, self.alphabet, self.beam_size, 1.0)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(seq), 0) # with a threshold that high, we won't find anything

    def test_high_beam_cut_threshold(self):
        """ simple beam search test with beam cut threshold above 1.0"""
        seq, path = beam_search(self.probs, self.alphabet, self.beam_size, 1.1)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(seq), 0) # with a threshold that high, we won't find anything

    def test_beam_search_mismatched_alphabet_short(self):
        """ simple beam search test with too few alphabet chars"""
        alphabet = "NAGC"
        with self.assertRaises(ValueError):
            beam_search(self.probs, alphabet, self.beam_size, self.beam_cut_threshold)

    def test_beam_search_mismatched_alphabet_long(self):
        """ simple beam search test with too many alphabet chars"""
        alphabet = "NAGCTX"
        with self.assertRaises(ValueError):
            beam_search(self.probs, alphabet, self.beam_size, self.beam_cut_threshold)

    def test_beam_search_short_alphabet(self):
        """ simple beam search test with short alphabet"""
        self.alphabet = "NAG"
        self.probs = self.get_random_data()
        seq, path = beam_search(self.probs, self.alphabet, self.beam_size, self.beam_cut_threshold)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_beam_search_long_alphabet(self):
        """ simple beam search test with long alphabet"""
        self.alphabet = "NABCDEFGHIJK"
        self.probs = self.get_random_data(10000)
        seq, path = beam_search(self.probs, self.alphabet, self.beam_size, self.beam_cut_threshold)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_beam_search_path(self):
        """ simple beam search with path"""
        w = 5000
        x = np.zeros((w, len(self.alphabet)), np.float32)
        x[:, 0] = 0.5  # set stay prob

        # emit a base evenly spaced along w
        emit = np.arange(0, w, len(self.alphabet) - 1)
        for base, pos in enumerate(emit):
            x[pos, base % 4 + 1] = 1.0

        seq, path = beam_search(x, self.alphabet, self.beam_size, self.beam_cut_threshold)
        np.testing.assert_array_equal(emit, path)
        self.assertEqual(len(seq), len(path))

    def test_repeat_sequence_path(self):
        """ simple beam search path test with a repeated sequence """
        w = 20
        x = np.zeros((w, len(self.alphabet)), np.float32)
        x[:, 0] = 0.5  # set stay prob

        expected_path = [6, 13, 18]
        for idx in expected_path:
            x[idx, 0] = 0.0
            x[idx, 1] = 1.0

        seq, path = beam_search(x, self.alphabet, self.beam_size, self.beam_cut_threshold)

        self.assertEqual(seq, 'AAA')
        self.assertEqual(len(seq), len(path))
        self.assertEqual(path, expected_path)

    def test_repeat_sequence_path_with_spread(self):
        """ simple beam search path test with a repeated sequence with probabilities spread"""
        w = 20
        x = np.zeros((w, len(self.alphabet)), np.float32)
        x[:, 0] = 0.5  # set stay prob

        expected_path = [6, 13, 18]
        for idx in expected_path:
            x[idx-1:idx + 1, 0] = 0.0
            x[idx-1:idx + 1, 1] = 1.0

        seq, path = beam_search(x, self.alphabet, self.beam_size, self.beam_cut_threshold)

        self.assertEqual(seq, 'AAA')
        self.assertEqual(len(seq), len(path))
        self.assertEqual(path, expected_path)


if __name__ == '__main__':
    main()
