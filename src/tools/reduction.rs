use cluster_editing::{
    algo::{Parameters, ProblemInstance},
    critical_cliques, parser, reduction, Graph, PetGraph,
};

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

    /// Which reductions to apply. Default: all.
    /// Possible values: crit_cliques, rules123, rule4, rule5
    /// To specify more than one rule, give the argument mutliple times.
    #[structopt(long = "reduction", number_of_values = 1)]
    reductions: Option<Vec<String>>,
}

fn do_reduction(opt: &Opt, name: &str) -> bool {
    opt.reductions
        .as_ref()
        .map(|r| r.iter().any(|n| n == name))
        .unwrap_or(true)
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let opt = Opt::from_args();

    println!(
        "Starting reduction tester on {} input files.",
        opt.input.len()
    );

    let crit_cliques = do_reduction(&opt, "crit_cliques");
    let rules123 = do_reduction(&opt, "rules123");
    let rule4 = do_reduction(&opt, "rule4");
    let rule5 = do_reduction(&opt, "rule5");

    println!(
        "Reductions: crit_cliques {}, rules123 {}, rule4 {}, rule5 {}",
        crit_cliques, rules123, rule4, rule5
    );

    let mut reduction_amounts = HashMap::<String, (usize, usize, usize)>::new();

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
        let mut edits = 0;

        info!("Starting reduction on {}", filename);

        for c in components.into_iter() {
            let (cg, imap) = c;
            let params = Parameters::new(6, 2, HashMap::new(), None);
            let mut instance = ProblemInstance::new(&params, cg.clone(), imap.clone());
            instance.k = f32::MAX;
            instance.k_max = f32::MAX;

            //reduction::initial_param_independent_reduction(&mut instance);

            if crit_cliques {
                crit_clique_reduction(&mut instance);
            }

            let mut applied_any_rule = true;
            while applied_any_rule && instance.k > 0.0 && instance.g.present_node_count() > 1 {
                applied_any_rule = false;

                // Don't repeatedly apply rules123 by themselves, one call will do all possible
                // applications.
                if rules123 {
                    reduction::rules123(&mut instance);
                }

                // Rule 4
                if rule4 {
                    while reduction::rule4(&mut instance) {
                        applied_any_rule = true;
                    }
                }

                if applied_any_rule {
                    continue;
                }

                // Rule 5
                if rule5 {
                    applied_any_rule = reduction::rule5(&mut instance);
                }
            }

            after += instance.g.present_node_count();
            edits +=
                cluster_editing::util::diff_graphs(&cg, &imap, &instance.g, &instance.imap).len();
        }

        reduction_amounts.insert(filename.to_string(), (before, after, edits));

        info!(
            "Reduced {}: {} nodes --> {} nodes, {} edits",
            filename, before, after, edits
        );
    }

    println!("Done, printing statistics...");

    let mut reduction_amounts = reduction_amounts.into_iter().collect::<Vec<_>>();
    reduction_amounts.sort_unstable_by(|(f1, _), (f2, _)| f1.cmp(f2));

    println!("file | before | after | edits");
    for (f, (before, after, edits)) in reduction_amounts {
        println!("{} | {:3} | {:3} | {}", f, before, after, edits);
    }

    Ok(())
}

pub fn crit_clique_reduction(p: &mut ProblemInstance) {
    let (g, imap) = critical_cliques::merge_cliques(&p.g, &p.imap, &mut p.path_log);
    p.g = g;
    p.imap = imap;
}
