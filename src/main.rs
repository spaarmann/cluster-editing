use cluster_editing::{algo, graphviz, parser, Graph};

use std::error::Error;
use std::path::PathBuf;

use petgraph::prelude::*;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "cluster-editing",
    about = "Solves the Cluster Editing problem for graphs."
)]
struct Opt {
    /// Input file, using the graph format of the PACE 2021 challenge.
    /// `stdin` if not specified.
    #[structopt(parse(from_os_str))]
    input: Option<PathBuf>,

    /// Output file. `stdout` if not specified.
    #[structopt(parse(from_os_str))]
    output: Option<PathBuf>,

    /// Print the input graph to the given path, as a PNG file.
    /// Requires a working graphviz installation accessible in the path.
    #[structopt(short = "i", long = "print-input", parse(from_os_str))]
    print_input: Option<PathBuf>,

    /// Print the output graph to the given path, as a PNG file.
    /// Requires a working graphviz installation accessible in the path.
    #[structopt(short = "o", long = "print-output", parse(from_os_str))]
    print_output: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();

    let graph: Graph = match opt.input {
        Some(path) => parser::parse_file(path),
        None => parser::parse(std::io::stdin().lock()),
    }?;

    if let Some(path) = opt.print_input {
        graphviz::print_graph(path, &graph);
    }

    let components = algo::split_into_connected_components(&graph);
    for (i, c) in components.into_iter().enumerate() {
        graphviz::print_graph(format!("component{}.png", i), &c);
    }

    // TODO: Do stuff with the graph
    let output = graph;

    if let Some(path) = opt.print_output {
        graphviz::print_graph(path, &output);
    }

    Ok(())
}
