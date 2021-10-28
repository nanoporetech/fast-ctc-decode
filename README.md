# fast-ctc-decode

![test-fast-ctc-decode](https://github.com/nanoporetech/fast-ctc-decode/workflows/test-fast-ctc-decode/badge.svg) [![PyPI version](https://badge.fury.io/py/fast-ctc-decode.svg)](https://badge.fury.io/py/fast-ctc-decode)

Blitzing fast CTC decoding library.

```
$ pip install fast-ctc-decode
$ npm i @nanopore/fast-ctc-decode
```

## Usage

### Python
```python
>>> from fast_ctc_decode import beam_search, viterbi_search
>>>
>>> alphabet = "NACGT"
>>> posteriors = np.random.rand(100, len(alphabet)).astype(np.float32)
>>>
>>> seq, path = viterbi_search(posteriors, alphabet)
>>> seq
'ACACTCGCAGCGCGATACGACTGATCGAGATATACTCAGTGTACACAGT'
>>>
>>> seq, path = beam_search(posteriors, alphabet, beam_size=5, beam_cut_threshold=0.1)
>>> seq
'ACACTCGCAGCGCGATACGACTGATCGAGATATACTCAGTGTACACAGT'
```

### Node / Web
```js
import init, { beam_search, viterbi_search } from 'fast-ctc';

const floatArr = [0.0, 0.4, 0.6, 0.0, 0.3, 0.7, 0.3, 0.3];
const alphabet = ["N","A","G"];
const beamSize = 5;
const beamCutThreshold = Number(0.0).toPrecision(2);
const collapseRepeats = true;
const shape = [10, 3];
const string = false;
const qBias = Number(0.0).toPrecision(2);
const qScale = Number(1.0).toPrecision(2);

// On web, note the base path will be your public folder
init('fast_ctc_decode_wasm_bg.wasm');

const viterbisearch = await beam_search(floatArr, alphabet, string, qScale, qBias, collapseRepeats, shape);

const beamsearch = await beam_search(floatArr, alphabet, beamSize, beamCutThreshold, collapseRepeats, shape);

console.log(viterbisearch); // ACCCAE
console.log(beamsearch); // ACCCAE
```

## Benchmark

| Implementation       | Time (s) | URL |
| -------------------- | -------- | --- |
| Viterbi (Rust)       |   0.0003 | [nanoporetech/fast-ctc-decode](https://github.com/nanoporetech/fast-ctc-decode.git) |
| Viterbi (Python)     |   0.0022 |     |
| Beam Search (Rust)   |   0.0033 | [nanoporetech/fast-ctc-decode](https://github.com/nanoporetech/fast-ctc-decode.git) |
| Beam Search (C++)    |   0.1034 | [parlance/ctcdecode](https://github.com/parlance/ctcdecode) |
| Beam Search (Python) |   3.3337 | [githubharald/CTCDecoder](https://github.com/githubharald/CTCDecoder) |


## Developer Quickstart

### Python

```
$ git clone https://github.com/nanoporetech/fast-ctc-decode.git
$ cd fast-ctc-decode
$ pip install --user maturin
$ make test
```

### JavaScript / Node

```
npm i
npm test
```

Note: You'll need a recent [rust](https://www.rust-lang.org/tools/install) compiler on your path to build the project.

By default, a fast (and less accurate) version of exponentiation is used for the 2D search. This can
be disabled by passing `--cargo-extra-args="--no-default-features"` to maturin, which provides more
accurate calculations but makes the 2D search take about twice as long.

## Credits

The original 1D beam search implementation was developed by [@usamec](https://github.com/usamec) for [deepnano-blitz](https://github.com/fmfi-compbio/deepnano-blitz).

The 2D beam search is based on [@jordisr](https://github.com/jordisr) and [@ihh](https://github.com/ihh) work in their [pair consensus decoding](https://doi.org/10.1101/2020.02.25.956771) paper.

### Licence and Copyright
(c) 2019 Oxford Nanopore Technologies Ltd.

fast-ctc-decode is distributed under the terms of the MIT License.  If a copy of the License
was not distributed with this file, You can obtain one at https://github.com/nanoporetech/fast-ctc-decode/

### Research Release

Research releases are provided as technology demonstrators to provide early access to features or stimulate Community development of tools. Support for this software will be minimal and is only provided directly by the developers. Feature requests, improvements, and discussions are welcome and can be implemented by forking and pull requests. However much as we would like to rectify every issue and piece of feedback users may have, the developers may have limited resource for support of this software. Research releases may be unstable and subject to rapid iteration by Oxford Nanopore Technologies.
