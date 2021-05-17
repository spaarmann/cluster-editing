use cluster_editing::{
    algo, algo::Edit, algo::ProblemInstance, graph_writer, graphviz, parser, Graph, PetGraph,
    Weight,
};

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use log::{error, info};
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

    /// Writes the input graph file out in another format.
    /// Syntax: `--write-input <format>:<path>.
    /// Supported formats are: `tgf`, `peace`, `gr`.
    #[structopt(long = "write-input")]
    write_input: Option<String>,

    /// Print the output graph to the given path, as a PNG file.
    /// Requires a working graphviz installation accessible in the path.
    #[structopt(short = "o", long = "print-output", parse(from_os_str))]
    print_output: Option<PathBuf>,

    /// Writes the output graph file out in another format.
    /// Syntax: `--write-output <format>:<path>.
    /// Supported formats are: `tgf`, `peace`, `gr`.
    #[structopt(long = "write-output")]
    write_output: Option<String>,

    /// Print the critical clique graph associated with the input graph, as a PNG file.
    /// Requires a working graphviz installation accessible in the path.
    #[structopt(long = "print-cliques", parse(from_os_str))]
    print_cliques: Option<PathBuf>,

    /// Print the graph after initial parameter-independent reduction, as a PNG file.
    /// Requires a working graphviz installation accessible in the path.
    #[structopt(long = "print-reduced", parse(from_os_str))]
    print_reduced: Option<PathBuf>,

    /// Which command is used to print the graph images. Can generally be any Graphviz tool,
    /// default is `sfdp`. `fdp` can be used for somewhat better images that take a longer time to
    /// create.
    #[structopt(long = "print-command", default_value = "sfdp")]
    print_command: String,

    #[structopt(long = "full-reduction-interval", default_value = "50")]
    full_reduction_interval: i32,

    #[structopt(long = "fast-reduction-interval", default_value = "200")]
    fast_reduction_interval: i32,

    /// If specified, various statistics will be written into files in the given directory.
    /// WARNING: This will write *many* files and incur significant overhead. It should probably
    /// get a much better output format before being used more seriously.
    #[structopt(long = "write-stats", parse(from_os_str))]
    stats_dir: Option<PathBuf>,

    /// If specified, information about what techniques led to how much reduction in parameter k
    /// over the whole execution will be written to this file.
    #[structopt(long = "k-red-stats", parse(from_os_str))]
    k_red_stats: Option<PathBuf>,

    #[structopt(short = "d", long = "debug", parse(try_from_str = parse_key_val), number_of_values = 1)]
    debug_options: Option<Vec<(String, String)>>,

    /// Suppresses the output required by the challenge since it can get quite long and isn't
    /// really necessary most of the time.
    #[structopt(long = "noout", short = "n")]
    no_out: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    info!("Starting solver...");

    let opt = Opt::from_args();

    let input_name;

    let graph: PetGraph = match opt.input {
        // The optil.io website used for the PACE challenge seems to pass an empty string arg
        // for some reason, this if guard ignores that.
        Some(path) if path.to_str().map(|s| !s.trim().is_empty()).unwrap_or(false) => {
            input_name = path.display().to_string();
            parser::parse_file(path)
        }
        Some(_) | None => {
            input_name = "stdin".to_string();
            parser::parse(std::io::stdin().lock())
        }
    }?;

    info!("Taking input graph from {}", input_name);

    if let Some(path) = opt.print_input {
        graphviz::print_graph(&opt.print_command, path, &graph);
    }

    if let Some(spec) = opt.write_input {
        write_graph(&Graph::<Weight>::new_from_petgraph(&graph).0, &spec);
    }

    if let Some(path) = opt.print_cliques {
        let (graph, _) = Graph::new_from_petgraph(&graph);
        let crit_graph = cluster_editing::critical_cliques::build_crit_clique_graph(&graph);
        graphviz::print_graph(&opt.print_command, path, &crit_graph.to_petgraph());
    }

    let debug_opts = opt
        .debug_options
        .map(|o| o.into_iter().collect())
        .unwrap_or_else(HashMap::new);

    let mut params = algo::Parameters::new(
        opt.full_reduction_interval,
        opt.fast_reduction_interval,
        debug_opts,
        opt.stats_dir,
    );

    if let Some(path) = opt.print_reduced {
        let (graph, imap) = Graph::new_from_petgraph(&graph);
        let mut instance = ProblemInstance::new(&params, graph.clone(), imap.clone());
        instance.k = f32::MAX;
        instance.k_max = f32::MAX;
        cluster_editing::reduction::initial_param_independent_reduction(&mut instance);
        graphviz::print_graph_vecs(
            &opt.print_command,
            path,
            &instance.g.to_petgraph(Some(&instance.imap), false),
        );
    }

    info!("Running with parameters: {:?}", params);

    let start = Instant::now();

    let (result_graph, result_edits) = algo::execute_algorithm(&graph, &mut params);

    let time = start.elapsed().as_secs_f32();

    info!("Finished in {} seconds!", time);

    info!(
        "Output graph has {} nodes and {} edges.",
        result_graph.node_count(),
        result_graph.edge_count()
    );

    if let Some(path) = opt.print_output {
        graphviz::print_graph(&opt.print_command, path, &result_graph);
    }

    if let Some(spec) = opt.write_output {
        write_graph(&Graph::<Weight>::new_from_petgraph(&result_graph).0, &spec);
    }

    if let Some(red_stats_file) = opt.k_red_stats {
        write_k_red_stats(&red_stats_file, &params);
    }

    if !opt.no_out {
        for edit in result_edits {
            match edit {
                Edit::Insert(u, v) => println!("{} {}", u + 1, v + 1),
                Edit::Delete(u, v) => println!("{} {}", u + 1, v + 1),
            }
        }
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

fn write_graph(graph: &Graph<Weight>, spec: &str) {
    match spec.split_once(':') {
        None => error!("Format for --write-input or --write-output not valid!"),
        Some((format, path)) => match format {
            "tgf" => graph_writer::write_graph_tgf(graph, None, path),
            "peace" => graph_writer::write_graph_peace(graph, None, path),
            "gr" => graph_writer::write_graph_gr(graph, path),
            _ => error!("Unknown format for --write-input!"),
        },
    }
}

fn write_k_red_stats(path: &Path, params: &algo::Parameters) {
    info!("Printing k reduction stats to {}", path.display());

    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "file | branching | early exit | rules123 | rule4 | rule5 | induced cost | zeroes"
    )
    .unwrap();

    let stats = params.stats.borrow();
    write!(writer, "{} | ", path.file_name().unwrap().to_string_lossy()).unwrap();
    write!(writer, "{} | ", stats.k_red_from_branching).unwrap();
    write!(writer, "{} | ", stats.k_red_from_early_exit).unwrap();
    write!(writer, "{} | ", stats.k_red_from_rules123).unwrap();
    write!(writer, "{} | ", stats.k_red_from_rule4).unwrap();
    write!(writer, "{} | ", stats.k_red_from_rule5).unwrap();
    write!(writer, "{} | ", stats.k_red_from_ind_cost).unwrap();
    write!(writer, "{}", stats.k_red_from_zeroes).unwrap();

    writer.flush().unwrap();
}
