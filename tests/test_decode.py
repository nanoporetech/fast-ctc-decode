#!/usr/env/bin python3

import numpy as np
from unittest import TestCase, main
from fast_ctc_decode import *


class Test1DBeamSearch(TestCase):
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
        seq, path = beam_search(self.probs, self.alphabet, self.beam_size, self.beam_cut_threshold)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_beam_search_list(self):
        """ simple beam search test with the canonical alphabet as a list"""
        seq, path = beam_search(self.probs, list(self.alphabet), self.beam_size, self.beam_cut_threshold)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_beam_search_tuple(self):
        """ simple beam search test with the canonical alphabet as a tuple"""
        seq, path = beam_search(self.probs, tuple(self.alphabet), self.beam_size, self.beam_cut_threshold)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_beam_search_named_args(self):
        """ simple beam search test with named arguments"""
        seq, path = beam_search(network_output=self.probs, alphabet=self.alphabet,
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
        seq, path = beam_search(self.probs, self.alphabet)
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
        self.probs.fill(np.nan)
        with self.assertRaisesRegex(RuntimeError, "Failed to compare values"):
            beam_search(self.probs, self.alphabet)

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
        seq, path = beam_search(self.probs, self.alphabet, self.beam_size, beam_cut_threshold=0.0)
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

        seq, path = beam_search(x, self.alphabet, self.beam_size, self.beam_cut_threshold)

        self.assertEqual(seq, 'AAACCCGGG')
        self.assertEqual(path, expected_path)

    def test_repeat_sequence_path_with_spread_probs(self):
        """ simple beam search path test with a repeated sequence with probabilities spread"""
        w = 20
        x = np.zeros((w, len(self.alphabet)), np.float32)
        x[:, 0] = 0.5  # set stay prob

        spread = 3
        expected_path = [6, 13, 18]
        for idx in expected_path:
            x[idx:idx + spread, 0] = 0.0
            x[idx:idx + spread, 1] = 1.0

        seq, path = beam_search(x, self.alphabet, self.beam_size, self.beam_cut_threshold)

        self.assertEqual(seq, 'AAA')
        self.assertEqual(len(seq), len(path))
        self.assertEqual(path, expected_path)


class TestViterbiSearch(TestCase):
    def setUp(self):
        self.alphabet = "NACGT"
        self.probs = self.get_random_data()

    def get_random_data(self, samples=100):
        x = np.random.rand(samples, len(self.alphabet)).astype(np.float32)
        return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

    def test_random_data(self):
        """Test viterbi search on some random data"""
        seq, path = viterbi_search(self.probs, self.alphabet)
        self.assertEqual(len(seq), len(path))
        self.assertEqual(len(set(seq)), len(self.alphabet) - 1)

    def test_random_data(self):
        """Test viterbi search on some random data with qstring generation"""
        seq, path = viterbi_search(self.probs, self.alphabet, qstring=True)
        self.assertEqual(len(seq), len(path) * 2)

    def test_not_enough_args(self):
        """Not enough arguments provided"""
        with self.assertRaises(TypeError):
            viterbi_search(self.probs)

    def test_alphabet_too_small(self):
        """When the alphabet is too small, it should raise"""
        with self.assertRaises(ValueError):
            viterbi_search(self.probs, "NACG")

    def test_alphabet_too_large(self):
        """When the alphabet is too large, it should raise"""
        with self.assertRaises(ValueError):
            viterbi_search(self.probs, "NACGTR")

    def test_beam_search_path(self):
        """data with a predefined path"""
        w = 5000
        x = np.zeros((w, len(self.alphabet)), np.float32)
        x[:, 0] = 0.5  # set stay prob

        # emit a base evenly spaced along w
        emit = np.arange(0, w, len(self.alphabet) - 1)
        for base, pos in enumerate(emit):
            x[pos, base % 4 + 1] = 1.0

        seq, path = viterbi_search(x, self.alphabet)
        np.testing.assert_array_equal(emit, path)
        self.assertEqual(len(seq), len(path))

    def test_repeat_sequence_path(self):
        """test with a repeated sequence """
        w = 20
        x = np.zeros((w, len(self.alphabet)), np.float32)
        x[:, 0] = 0.5  # set stay prob

        expected_path = [6, 13, 18]
        for idx in expected_path:
            x[idx, 0] = 0.0
            x[idx, 1] = 1.0

        seq, path = viterbi_search(x, self.alphabet)

        self.assertEqual(seq, 'AAA')
        self.assertEqual(len(seq), len(path))
        self.assertEqual(path, expected_path)

    def test_repeat_sequence_path_with_qstring(self):
        """test with a repeated sequence with qstring generation """
        w = 20
        x = np.zeros((w, len(self.alphabet)), np.float32)
        x[:, 0] = 0.5  # set stay prob

        expected_path = [6, 13, 18]
        for idx in expected_path:
            x[idx, 0] = 0.0
            x[idx, 1] = 1.0

        seq, path = viterbi_search(x, self.alphabet, qstring=True)
        qual = seq[len(path):]
        seq = seq[:len(path)]

        self.assertEqual(seq, 'AAA')
        self.assertEqual(qual, 'III')
        self.assertEqual(len(seq), len(path))
        self.assertEqual(path, expected_path)

    def test_mean_qscores(self):
        """ test mean qscore generation """
        w = 20
        x = np.zeros((w, len(self.alphabet)), np.float32)
        x[:, 0] = 0.5  # set stay prob

        # Q10 = "5"
        x[3, 0] = 0.0
        x[3, 1] = 0.99
        x[4, 0] = 0.0
        x[4, 1] = 0.99

        # Q20 = "?"
        x[6, 0] = 0.0
        x[6, 2] = 0.999
        x[7, 0] = 0.0
        x[7, 2] = 0.999

        # Q5 = "&"
        x[9, 0] = 0.0
        x[9, 4] = 0.6
        x[10, 0] = 0.0
        x[10, 4] = 0.7
        x[11, 0] = 0.0
        x[11, 4] = 0.8

        # Q3 = "$"
        x[13, 0] = 0.0
        x[13, 4] = 0.4
        x[14, 0] = 0.0
        x[14, 4] = 0.5
        x[15, 0] = 0.0
        x[15, 4] = 0.6

        seq, path = viterbi_search(x, self.alphabet, qstring=True)
        qual = seq[len(path):]
        seq = seq[:len(path)]

        self.assertEqual(seq, 'ACTT')
        self.assertEqual(qual, '5?&$')
        self.assertEqual(len(seq), len(path))

    def test_repeat_sequence_path_with_multi_char_alpha(self):
        """Test that a multi-char alphabet works"""
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

        seq, path = viterbi_search(x, self.alphabet)

        self.assertEqual(seq, 'AAACCCGGG')
        self.assertEqual(path, expected_path)

    def test_beam_off_path(self):
        """a set a probabilities where a viterbi search would produce the wrong result"""
        x = np.array([
            [0.7, 0.1, 0.2],
            [0.7, 0.1, 0.2],
            [0.2, 0.3, 0.5],
            [0.2, 0.2, 0.6],
            [0.3, 0.3, 0.4],
            [0.2, 0.2, 0.6],
            [0.2, 0.3, 0.5],
            [0.7, 0.1, 0.2],
            [0.7, 0.1, 0.2],
        ], np.float32)

        seq, path = viterbi_search(x, "NAB")
        self.assertEqual(seq, "B")


class TestDuplexBeamSearch(TestCase):
    def setUp(self):
        self.beam_size = 5
        self.alphabet = "NACGT"
        self.beam_cut_threshold = 0.1
        self.probs_1 = self.get_random_data()
        self.probs_2 = self.get_random_data()

    def get_random_data(self, samples=100):
        x = np.random.rand(samples, len(self.alphabet)).astype(np.float32)
        return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

    def test_nans(self):
        """beam_search_duplex is passed NaN values"""
        self.probs_1.fill(np.nan)
        with self.assertRaisesRegex(RuntimeError, "Failed to compare values"):
            beam_search_duplex(self.probs_1, self.probs_2, self.alphabet)

    def test_identical_data(self):
        """Test duplex beam search on the same data twice"""
        x = np.array([
            [0.01, 0.98, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.98, 0.01],
            [0.9,  0.05, 0.05],
            [0.7,  0.05, 0.35],
            [0.9,  0.05, 0.05],
            [0.01, 0.98, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
        ], np.float32)
        seq = beam_search_duplex(x, x, "NAB")
        self.assertEqual("AAB", seq)

    def test_disagreeing_data(self):
        """Test duplex beam search on data that disagrees"""
        x = np.array([
            [0.01, 0.98, 0.01],
            [0.01, 0.34, 0.65],
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98],
        ], np.float32)
        self.assertEqual("ABAB", beam_search(x, "NAB")[0])
        y = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], np.float32)
        self.assertEqual("AB", beam_search_duplex(x, y, "NAB"))


if __name__ == '__main__':
    main()
