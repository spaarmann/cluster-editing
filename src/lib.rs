pub mod algo;
pub mod critical_cliques;
pub mod graph;
pub mod graphviz;
pub mod parser;

pub type PetGraph = petgraph::Graph<usize, u8, petgraph::Undirected, u32>;
pub type Weight = f32;
pub use graph::Graph;
