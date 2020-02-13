# fast-ctc-decode

![test](https://github.com/nanoporetech/fast-ctc-decode/workflows/test/badge.svg)

Blazingly fast beam search.

```
$ pip install fast-ctc-decode
```

## Usage

```python
>>> from fast_ctc_decode import beam_search
>>>
>>> beam_size = 5
>>> alphabet = "NACGT"
>>> beam_prune_threshold = 0.1  # < 1 / len(alphabet)
>>> posteriors = np.random.rand(100, 5).astype(np.float32)
>>>
>>> beam_search(posteriors, alphabet, beam_size, beam_prune_threshold)
'ACACTCGCAGCGCGATACGACTGATCGAGATATACTCAGTGTACACAGT'
```

## Benchmark

| Implementation       | Time (s) | URL |
| -------------------- | -------- | --- |
| Greedy (Python)      |   0.0022 |     |
| Beam Search (Rust)   |   0.0033 | [nanoporetech/fast-ctc-decode](https://github.com/nanoporetech/fast-ctc-decode.git) |
| Beam Search (C++)    |   0.1034 | [parlance/ctcdecode](https://github.com/parlance/ctcdecode) |
| Beam Search (Python) |   3.3337 | [githubharald/CTCDecoder](https://github.com/githubharald/CTCDecoder) |


## Developer Quickstart

```
$ git clone https://github.com/nanoporetech/fast-ctc-decode.git
$ cd fast-ctc-decode
$ pip install --user maturin
$ make test
```

Note: You'll need a recent [rust](https://www.rust-lang.org/tools/install) compiler on your path to build the project.

## Credits

The original beam search implementation was developed by [@usamec](https://github.com/usamec) for [deepnano-blitz](https://github.com/fmfi-compbio/deepnano-blitz).
