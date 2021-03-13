clean:
	rm -rf *~ dist *.egg-info build target

build:
	maturin build --release

develop:
	maturin develop

test:
	cargo test
	python3 tests/test_decode.py
