fmt:
	cargo fmt --all

clippy:
	cargo clippy --all-targets -- -D warnings

test:
	cargo test --all --all-features

ci: fmt clippy test

.PHONY: fmt clippy test ci
