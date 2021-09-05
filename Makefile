clean:
	cargo clean
	rm -rf *~ dist *.egg-info build target

build:
	maturin build -i python3 --release

develop:
	maturin develop --release

test:
	cargo test
	python3 tests/test_decode.py
