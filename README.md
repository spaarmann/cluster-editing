# Pandora

Pandora is a solver for the `Cluster Editing` problem, written for
[my bachelor thesis](https://github.com/spaarmann/bachelor-thesis) and as a submission for the
[PACE Challenge 2021](https://pacechallenge.org/2021/).

The solver follows a fixed-parameter approach, combining various techniques introduced in existing
publications on the topic. It solves the unweighted problem as required by the PACE challenge, but
internally works largely based on an equivalent weighted instance, and would be easy to adjust to
also solve weighted input instances. A more detailed description of the techniques used will be
added soon.

## Installation

The solver is written in Rust, and compiled using the standard `cargo` tool. It currently uses two
unstable standard library features and thus requires a *nightly* Rust version.

If `rustup` (the Rust toolchain manager) is installed but no nightly Rust is installed yet,
installations might look as follows:

```
$ rustup install nightly
$ git clone https://github.com/spaarmann/cluster-editing
$ cd cluster-editing
$ rustup override set nightly
$ cargo build --release
```

An executable will then be placed into `target/release/cluster-editing`. Adjust as appropriate if a
toolchain is already installed etc.

The resulting binary will be optimized for the system it was compiled on, possibly using instruction
set extensions that are supported there but not necessarily widely on other systems, and dynamically
linked to the system `libc`. One can also compile using `./compile_optilio.sh` to target a CPU model
compatible with the optil.io system used by the PACE challenge and with a statically linked MUSL
libc, for greater portability. (This is how the submission for the challenge was compiled.)

## External Libraries

The solver uses some open-source libraries from `crates.io`, which are automatically downloaded
during the build process by `cargo`:

- [`rustc-hash`](https://crates.io/crates/rustc-hash) provides `HashMap` and `HashSet`
  implementations that are deterministic, unlike those in the Rust standard library. This is useful
  for reproducibility, but otherwise not critical for correctness of the solver.

- [`petgraph`](https://crates.io/crates/petgraph) is used in a few places to represent a graph. The
  solver mainly operates on its own internal `Graph` structure, and the `petgraph` dependency could
  easily be taken out. It is present mainly for useful debugging features such as outputting a graph
  in a format suitable for visualizing using `graphviz`.

- `log`, `env_logger`, and `structopt` are used for logging and argument parsing respectively, but
  have no other impact on the solver itself.

- Additionally, `lazy_static`, `regex`, `rayon`, and `wait-timeout` are used in some tools also
  contained in this repository, like the `runner` used to run the solver on multiple instances in
  parallel or a tool used to test the effectiveness of reduction techniques, but not in the solver
  itself.

These dependencies also have a few transitive dependencies themselves, largely libraries relatively
fundamental to the Rust ecosystem. Overall, none of the dependencies are necessary for the main
solver algorithm itself, they only provide adjacent functionality.

## License

Licensed under either the [Apache License, Version 2.0](./LICENSE-APACHE.md) or the [MIT
License](./LICENSE-MIT.md) at your option.
