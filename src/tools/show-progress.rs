use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use regex::Regex;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "show-progress",
    about = "Analyzes a cluster-editing log file of a single (potentially cancelled/interrupted) run and prints how much progress on solving was made."
)]
struct Opt {
    /// Input file, should consist of output of the main cluster-editing tool.
    #[structopt(parse(from_os_str))]
    input: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let opt = Opt::from_args();

    let filename = opt.input.file_name().unwrap().to_str().unwrap();
    let file = File::open(opt.input.clone())?;
    let reader = BufReader::new(file);

    let mut max_component = -1;
    let mut max_k = -1;
    let mut last_comp_finished = -1;

    let comp_start_re = Regex::new(r"Solving component (\d+)\.\.\.").unwrap();
    let new_k_re = Regex::new(r"\[driver\] Starting search with k=(\d+)\.\.\.").unwrap();
    let comp_done_re =
        Regex::new(r"Found a cluster editing with k=(\d+) and (\d+) edits for component (\d+):")
            .unwrap();
    let graph_done_re = Regex::new(r"Final set of \d+ de-duplicated edits").unwrap();

    for line in reader.lines() {
        let line = line.unwrap();
        if let Some(captures) = comp_start_re.captures(&line) {
            let comp = captures.get(1).unwrap().as_str().parse().unwrap();
            if comp != max_component + 1 {
                panic!("Jumped from comp {} to {}!", max_component, comp);
            }

            max_component = comp;
            max_k = -1;
        }

        if let Some(captures) = new_k_re.captures(&line) {
            let k = captures.get(1).unwrap().as_str().parse().unwrap();
            if max_k != -1 && k != max_k + 1 {
                panic!("Jumped from k {} to {}", max_k, k);
            }

            max_k = k;
        }

        if let Some(captures) = comp_done_re.captures(&line) {
            let k: i32 = captures.get(1).unwrap().as_str().parse().unwrap();
            let comp = captures.get(3).unwrap().as_str().parse().unwrap();

            if comp != last_comp_finished + 1 {
                panic!(
                    "Jumped from finished comp {} to {}",
                    last_comp_finished, comp
                );
            }
            if k != 0 {
                log::warn!("Comp {} final k {} is not 0!", comp, k);
            }

            last_comp_finished = comp;
        }

        if graph_done_re.is_match(&line) {
            println!("{} completed entirely", filename);
            return Ok(());
        }
    }

    println!(
        "{} got to component {}, k = {}",
        filename, max_component, max_k
    );

    Ok(())
}
