pub mod algo;
pub mod graphviz;
pub mod parser;

pub type Graph = petgraph::stable_graph::StableGraph<u32, u8, petgraph::Undirected, u32>;
