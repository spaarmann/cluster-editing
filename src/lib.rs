#![feature(map_first_last)]
#![feature(binary_heap_retain)]
// Writing graph theory code tends to do this :)
#![allow(clippy::many_single_char_names)]
// `if let None = ` is good actually if there's a very symmetric `if let Some` two lines down
#![allow(clippy::redundant_pattern_matching)]
// Also for symmetry
#![allow(clippy::iter_nth_zero)]

#[macro_use]
pub mod util;

pub mod algo;
pub mod conflicts;
pub mod critical_cliques;
pub mod graph;
pub mod graph_writer;
pub mod graphviz;
pub mod induced_costs;
pub mod parser;
pub mod reduction;
pub mod upper_bound;

pub type PetGraph = petgraph::Graph<usize, u8, petgraph::Undirected, u32>;
pub type Weight = f32;
pub use graph::Graph;
