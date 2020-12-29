use cluster_editing::{algo, graphviz, parser, Graph, PetGraph};

use std::error::Error;
use std::path::PathBuf;

use log::info;
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
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("Starting solver...");

    let opt = Opt::from_args();

    let graph: PetGraph = match opt.input {
        Some(path) => parser::parse_file(path),
        None => parser::parse(std::io::stdin().lock()),
    }?;

    if let Some(path) = opt.print_input {
        graphviz::print_graph(path, &graph);
    }

    let components = algo::split_into_connected_components(&graph);

    info!(
        "Decomposed input graph into {} components",
        components.len()
    );

    let mut result = graph.clone();

    for (i, c) in components.into_iter().enumerate() {
        info!("Solving component {}...", i);
        let (cg, imap) = Graph::new_from_petgraph(&c);
        let (k, edits) = algo::find_optimal_cluster_editing(&cg);
        info!(
            "Found a cluster editing of {} edits for component {}: {:?}",
            k, i, edits
        );

        for edit in edits {
            match edit {
                algo::Edit::Insert(u, v) => {
                    result.add_edge(NodeIndex::new(imap[u]), NodeIndex::new(imap[v]), 0);
                }
                algo::Edit::Delete(u, v) => {
                    result.remove_edge(
                        result
                            .find_edge(NodeIndex::new(imap[u]), NodeIndex::new(imap[v]))
                            .unwrap(),
                    );
                }
            };
        }
    }

    if let Some(path) = opt.print_output {
        graphviz::print_graph(path, &result);
    }

    Ok(())
}
