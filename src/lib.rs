pub mod algo;
pub mod critical_cliques;
pub mod graphviz;
pub mod parser;

pub type Graph = petgraph::Graph<u32, u8, petgraph::Undirected, u32>;
