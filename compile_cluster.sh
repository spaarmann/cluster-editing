#!/usr/bin/env bash
RUSTFLAGS='-C target-cpu=sandybridge' cargo build --release --target x86_64-unknown-linux-musl
