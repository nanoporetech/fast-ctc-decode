import sys
import time
import torch
import numpy as np
from itertools import groupby

# packaging is lacking for the following 2
# they need installing from GitHub manually
from ctcdecode import CTCBeamDecoder
from BeamSearch import ctcBeamSearch
from fast_ctc_decode import beam_search


def decode_ctc_greedy(predictions, labels, *args):
    """
    Argmax decoder with collapsing repeats
    """
    path = np.argmax(predictions, axis=1)
    return ''.join([labels[b] for b, g in groupby(path) if b])


def py_beam_search(predictions, labels, beam_width=5, beam_cut_threshold=0.1):
    """
    Python Beam search CTC decoder https://github.com/githubharald/CTCDecoder
      *modified to remove the language model functionality
    """
    return ctcBeamSearch(predictions, labels[1:], beamWidth=5)


def cpp_beam_search(predictions, labels, beam_width=5, beam_cut_threshold=0.1):
    """
    C++ Beam search CTC decoder https://github.com/parlance/ctcdecode
    """
    # add batch dimension expected by CTCBeamDecoder
    predictions = np.expand_dims(predictions, 0)
    predictions = torch.FloatTensor(predictions)
    decoder = CTCBeamDecoder(
        labels, beam_width=beam_width, cutoff_prob=beam_cut_threshold
    )
    beam_result, _, _, out_seq_len = decoder.decode(predictions)
    beam_result = beam_result[0][0][0:out_seq_len[0][0]]
    return ''.join(labels[x] for x in beam_result)


def benchmark(f, x, beam_size=5, beam_cut_threshold=0.1, labels='NACGT', limit=10):
    t0 = time.perf_counter()
    for y in x[:10]:
        sequence = f(y, labels, beam_size, beam_cut_threshold)
    return time.perf_counter() - t0


if __name__ == '__main__':

    limit = 10; n = 10; beam_size = 5; prune = 1e-1
    try:
        # better benchmark with real posteriors
        x = np.load(sys.argv[1])
    except IndexError:
        x = np.random.rand(10000, 5, 25)
        x = (x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)).astype(np.float32)

    print("beam_size={};prune_threshold={}".format(beam_size, prune))

    for f in [decode_ctc_greedy, beam_search, cpp_beam_search, py_beam_search]:
        timings = []
        for _ in range(n):
            timings.append(benchmark(f, x, beam_size, prune, limit=limit))
        print('{:18s}: mean(sd) of {} runs: {:2.6f}({:2.6f})'.format(
            f.__name__, n, np.mean(timings), np.std(timings))
        )
