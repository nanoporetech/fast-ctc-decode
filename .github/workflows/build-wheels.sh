#!/bin/bash
set -e -x

curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env
rustup default 1.68.2

cargo build

for PYBIN in /opt/python/cp3[7891]*/bin; do
    "${PYBIN}/pip" install maturin
    "${PYBIN}/maturin" build -F python -i "${PYBIN}/python" --release
done

for wheel in target/wheels/*.whl; do
    auditwheel repair "${wheel}"
done
