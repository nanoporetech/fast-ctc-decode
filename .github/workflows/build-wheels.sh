#!/bin/bash
set -e -x

for PYBIN in /opt/python/cp3[5678]*/bin; do
    "${PYBIN}/pip" install maturin
    "${PYBIN}/maturin" build -i "${PYBIN}/python"
done

for wheel in target/wheels/*.whl; do
    auditwheel repair "${wheel}"
done

/opt/python/cp35-cp35m/bin/python -m pip install twine
/opt/python/cp35-cp35m/bin/python -m twine upload wheelhouse
