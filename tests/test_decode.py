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
        x = np.random.rand(samples, len(self.alphabet)).astype(np.float32)
        return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

    def test_beam_search(self):
        """ simple beam search test with the canonical alphabet"""
        seq, path, qstr = beam_search(self.probs, self.alphabet, self.beam_size, self.beam_cut_threshold)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_beam_search_list(self):
        """ simple beam search test with the canonical alphabet as a list"""
        seq, path, qstr = beam_search(self.probs, list(self.alphabet), self.beam_size, self.beam_cut_threshold)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_beam_search_tuple(self):
        """ simple beam search test with the canonical alphabet as a tuple"""
        seq, path, qstr = beam_search(self.probs, tuple(self.alphabet), self.beam_size, self.beam_cut_threshold)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_beam_search_named_args(self):
        """ simple beam search test with named arguments"""
        seq, path, qstr = beam_search(network_output=self.probs, alphabet=self.alphabet,
                                beam_size=self.beam_size,
                                beam_cut_threshold=self.beam_cut_threshold)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_beam_search_not_enough_args(self):
        """ simple beam search test with not enough arguments"""
        with self.assertRaises(TypeError):
            beam_search(self.probs)

    def test_beam_search_defaults(self):
        """ simple beam search test using argument defaults"""
        seq, path, qstr = beam_search(self.probs, self.alphabet)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_beam_search_alphabet(self):
        """ simple beam search test with different alphabet"""
        seq, path, qstr = beam_search(self.probs, "NRUST", self.beam_size, self.beam_cut_threshold)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_zero_beam_size(self):
        """ simple beam search test with zero beam size"""
        with self.assertRaises(ValueError):
            beam_search(self.probs, self.alphabet, 0, self.beam_cut_threshold)

    def test_zero_beam_cut_threshold(self):
        """ simple beam search test with beam cut threshold of 0.0"""
        seq, path, qstr = beam_search(self.probs, self.alphabet, self.beam_size, 0.0)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_negative_beam_cut_threshold(self):
        """ simple beam search test with beam cut threshold below 0.0"""
        with self.assertRaises(ValueError):
            beam_search(self.probs, self.alphabet, self.beam_size, -0.1)

    def test_beam_cut_threshold_boundary(self):
        """ simple beam search test with beam cut threshold of 1/len(alphabet)"""
        with self.assertRaises(ValueError):
            beam_search(self.probs, self.alphabet, self.beam_size, 1.0/len(self.alphabet))

    def test_high_beam_cut_threshold(self):
        """ simple beam search test with very high beam cut threshold"""
        with self.assertRaises(ValueError):
            beam_search(self.probs, self.alphabet, self.beam_size, 1.1)

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

    def test_nans(self):
        """beam_search is passed NaN values"""
        self.probs.fill(np.NaN)
        with self.assertRaisesRegex(RuntimeError, "Failed to compare values"):
            beam_search(self.probs, self.alphabet)

    def test_beam_search_short_alphabet(self):
        """ simple beam search test with short alphabet"""
        self.alphabet = "NAG"
        self.probs = self.get_random_data()
        seq, path, qstr = beam_search(self.probs, self.alphabet, self.beam_size, self.beam_cut_threshold)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_beam_search_long_alphabet(self):
        """ simple beam search test with long alphabet"""
        self.alphabet = "NABCDEFGHIJK"
        self.probs = self.get_random_data(10000)
        seq, path, qstr = beam_search(self.probs, self.alphabet, self.beam_size, beam_cut_threshold=0.0)
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

        seq, path, qstr = beam_search(x, self.alphabet, self.beam_size, self.beam_cut_threshold)
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

        seq, path, qstr = beam_search(x, self.alphabet, self.beam_size, self.beam_cut_threshold)

        self.assertEqual(seq, 'AAA')
        self.assertEqual(len(seq), len(path))
        self.assertEqual(path, expected_path)

    def test_repeat_sequence_path_with_multi_char_alpha(self):
        """ simple beam search path test with a repeated sequence and multi-char alphabet """
        w = 20
        self.alphabet = ["N", "AAA", "CCC", "GGG", "TTTT"]
        x = np.zeros((w, len(self.alphabet)), np.float32)
        x[:, 0] = 0.5  # set stay prob

        alphabet_idx = 1
        expected_path = [6, 13, 18]
        for idx in expected_path:
            x[idx, 0] = 0.0
            x[idx, alphabet_idx] = 1.0
            alphabet_idx += 1

        seq, path, qstr = beam_search(x, self.alphabet, self.beam_size, self.beam_cut_threshold)

        self.assertEqual(seq, 'AAACCCGGG')
        self.assertEqual(path, expected_path)

    def test_repeat_sequence_path_with_spread_probs(self):
        """ simple beam search path test with a repeated sequence with probabilities spread"""
        w = 20
        x = np.zeros((w, len(self.alphabet)), np.float32)
        x[:, 0] = 0.5  # set stay prob

        expected_path = [6, 13, 18]
        for idx in expected_path:
            x[idx-1:idx + 1, 0] = 0.0
            x[idx-1:idx + 1, 1] = 1.0

        seq, path, qstr = beam_search(x, self.alphabet, self.beam_size, self.beam_cut_threshold)

        self.assertEqual(seq, 'AAA')
        self.assertEqual(len(seq), len(path))
        self.assertEqual(path, expected_path)


if __name__ == '__main__':
    main()
