#!/usr/bin/env sh
# Build the browser playground into ../veer-site/public/playground.
# Needs: rustup target add wasm32-unknown-unknown; cargo install wasm-bindgen-cli --version 0.2.130
set -e
cd "$(dirname "$0")/.."
SITE="${VEER_SITE:-../veer-site}"
cargo build --target wasm32-unknown-unknown --release -p veer-wasm
wasm-bindgen --target web --no-typescript --out-dir "$SITE/public/playground" target/wasm32-unknown-unknown/release/veer_wasm.wasm
mkdir -p "$SITE/src/examples"
cp examples/*.veer "$SITE/src/examples/"
ls -la "$SITE/public/playground"
