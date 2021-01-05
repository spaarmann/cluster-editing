use cluster_editing::{algo, parser, Graph, PetGraph};

use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;

use log::info;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "reduction",
    about = "Performs parameter-independent reduction on multiple graphs and outputs statistics."
)]
struct Opt {
    /// Input files, using the graph format of the PACE 2021 challenge.
    #[structopt(parse(from_os_str))]
    input: Vec<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let opt = Opt::from_args();

    println!(
        "Starting reduction tester on {} input files.",
        opt.input.len()
    );

    let mut reduction_amounts = HashMap::<usize, usize>::new();

    for input in opt.input {
        let filename = input
            .file_name()
            .expect("every input is a file")
            .to_str()
            .expect("input paths are valid UTF-8");

        let graph: PetGraph = parser::parse_file(&input)?;
        let (graph, imap) = Graph::new_from_petgraph(&graph);
        let (components, _) = graph.split_into_components(&imap);

        let before = graph.size();
        let mut after = 0;

        for c in components.into_iter() {
            let (cg, imap) = c;
            let (reduced, _, _, _) = algo::param_independent_reduction(&cg, &imap);

            after += reduced.present_node_count();
        }

        *reduction_amounts.entry(before - after).or_default() += 1;

        info!("Reduced {}: {} nodes --> {} nodes", filename, before, after);
    }

    println!("Done, printing statistics...");

    let mut reduction_amounts = reduction_amounts.into_iter().collect::<Vec<_>>();
    reduction_amounts.sort_unstable_by(|&(_, c1), &(_, c2)| c1.cmp(&c2).reverse());

    println!("reduction | count");
    for (r, c) in reduction_amounts {
        println!("{:9} | {}", r, c);
    }

    Ok(())
}
