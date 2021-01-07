use std::error::Error;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use log::info;
use rayon::{prelude::*, ThreadPoolBuilder};
use structopt::StructOpt;
use wait_timeout::ChildExt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "runner",
    about = "Runs the solver on a given set of graphs, with a given amount of parallelism and a timeout per graph."
)]
struct Opt {
    //TODO: It would be nice to just call the algorithm without starting another process maybe.
    /// Solver program.
    #[structopt(
        default_value = "target/release/cluster-editing",
        parse(from_os_str),
        long = "solver"
    )]
    solver: PathBuf,
    /// Timeout per graph, in minutes.
    #[structopt(default_value = "30", long = "timeout")]
    timeout: u64,
    /// Amount of parallel workers.
    #[structopt(default_value = "10", long = "num-workers")]
    num_workers: usize,
    /// Output directory. A result file for each input file will be created in this directory.
    #[structopt(parse(from_os_str))]
    output_dir: PathBuf,
    /// Input files, using the graph format of the PACE 2021 challenge.
    #[structopt(parse(from_os_str))]
    input: Vec<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let opt = Opt::from_args();

    info!("Starting runner on {} input files.", opt.input.len());

    ThreadPoolBuilder::new()
        .num_threads(opt.num_workers)
        .build_global()
        .unwrap();

    let completed = opt
        .input
        .par_iter()
        .map(|in_path| do_file(&opt.solver, in_path, &opt.output_dir, opt.timeout))
        .filter(|&x| x)
        .count();

    info!("Done. {} of {} completed.", completed, opt.input.len());

    Ok(())
}

fn do_file(command: &PathBuf, in_path: &PathBuf, out_dir: &PathBuf, timeout: u64) -> bool {
    let filename = in_path
        .file_name()
        .expect("every input is a file")
        .to_str()
        .expect("input paths are valid UTF-8");

    info!("Starting worker for {}...", filename);

    let mut child = Command::new(command)
        .arg(in_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let result = match child
        .wait_timeout(Duration::from_secs(timeout * 60))
        .unwrap()
    {
        Some(status) => {
            info!("Completed {} with status {}", filename, status);
            true
        }
        None => {
            info!("{} timed out!", filename);
            child.kill().unwrap();
            child.wait().unwrap();
            false
        }
    };

    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();

    let mut output_path = out_dir.clone();
    output_path.push(format!("{}.out", filename));
    let mut output_file = File::create(output_path).unwrap();

    output_file.write_all("== stdout ==\n".as_bytes()).unwrap();
    io::copy(&mut stdout, &mut output_file).unwrap();

    output_file
        .write_all("\n== stderr ==\n".as_bytes())
        .unwrap();
    io::copy(&mut stderr, &mut output_file).unwrap();

    output_file.flush().unwrap();

    result
}
