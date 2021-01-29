use cluster_editing::{algo, graphviz, parser, Graph, PetGraph};

use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;

use log::info;
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

    /// Print the critical clique graph associated with the input graph, as a PNG file.
    /// Requires a working graphviz installation accessible in the path.
    #[structopt(long = "print-cliques", parse(from_os_str))]
    print_cliques: Option<PathBuf>,

    /// Which command is used to print the graph images. Can generally be any Graphviz tool,
    /// default is `sfdp`. `fdp` can be used for somewhat better images that take a longer time to
    /// create.
    #[structopt(long = "print-command", default_value = "sfdp")]
    print_command: String,

    #[structopt(long = "full-reduction-interval", default_value = "6")]
    full_reduction_interval: i32,

    #[structopt(short = "d", long = "debug", parse(try_from_str = parse_key_val), number_of_values = 1)]
    debug_options: Option<Vec<(String, String)>>,
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
        graphviz::print_graph(&opt.print_command, path, &graph);
    }

    if let Some(path) = opt.print_cliques {
        let (graph, _) = Graph::new_from_petgraph(&graph);
        let crit_graph = cluster_editing::critical_cliques::build_crit_clique_graph(&graph);
        graphviz::print_graph(&opt.print_command, path, &crit_graph.into_petgraph());
    }

    let debug_opts = opt
        .debug_options
        .map(|o| o.into_iter().collect())
        .unwrap_or_else(|| HashMap::new());

    let params = algo::Parameters {
        full_reduction_interval: opt.full_reduction_interval,
        debug_opts,
    };

    info!("Running with debug options: {:?}", params.debug_opts);

    let result = algo::execute_algorithm(&graph, params);

    info!(
        "Output graph has {} nodes and {} edges.",
        result.node_count(),
        result.edge_count()
    );

    if let Some(path) = opt.print_output {
        graphviz::print_graph(&opt.print_command, path, &result);
    }

    Ok(())
}

use std::str::FromStr;
// Based on structopt's `keyvalue` example.
fn parse_key_val<K, V>(s: &str) -> Result<(K, V), Box<dyn Error>>
where
    K: FromStr,
    K::Err: Error + 'static,
    V: FromStr,
    V::Err: Error + 'static,
{
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid key=value pair: no `=` found in `{}`", s))?;
    Ok((s[..pos].parse()?, s[pos + 1..].parse()?))
}
