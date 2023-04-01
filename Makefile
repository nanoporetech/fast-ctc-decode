clean:
	cargo clean
	rm -rf *~ dist *.egg-info build target

build:
	maturin build -F python  -i python3 --release

develop:
	maturin develop -F python --release

test:
	cargo test --features python
	python3 tests/test_decode.py
