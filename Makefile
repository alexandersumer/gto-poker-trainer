fmt:
	cargo fmt --all

clippy:
	cargo clippy --all-targets -- -D warnings

test:
	cargo test --all --all-features

release:
	cargo build --release

python-qa: release
	python -m pip install -r qa/requirements.txt
	GTO_TRAINER_BIN=./target/release/gto-trainer python -m pytest qa/tests

browser-qa: release
	npm ci
	npx playwright install --with-deps chromium
	GTO_TRAINER_HOST=127.0.0.1 GTO_TRAINER_PORT=8082 npm test

qa: python-qa browser-qa

ci: fmt clippy test qa

.PHONY: fmt clippy test release python-qa browser-qa qa ci
