use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

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
    /// Optional arguments to pass along to the solver. Any `$i` will be replaced by the current
    /// instance.
    #[structopt(long = "solver-args", default_value = "")]
    solver_args: String,
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

    let (completed_count, total_time) = opt
        .input
        .par_iter()
        .map(|in_path| {
            do_file(
                &opt.solver,
                &opt.solver_args,
                in_path,
                &opt.output_dir,
                opt.timeout,
            )
        })
        .filter_map(|(success, time)| if success { Some(time) } else { None })
        .fold(|| (0, 0.0), |(c, total), t| (c + 1, total + t))
        .reduce(|| (0, 0.0), |(c1, t1), (c2, t2)| (c1 + c2, t1 + t2));

    info!(
        "Done. {} of {} completed in {} seconds.",
        completed_count,
        opt.input.len(),
        total_time
    );

    Ok(())
}

fn do_file(
    command: &Path,
    args: &str,
    in_path: &Path,
    out_dir: &Path,
    timeout: u64,
) -> (bool, f32) {
    let filename = in_path
        .file_name()
        .expect("every input is a file")
        .to_str()
        .expect("input paths are valid UTF-8");

    info!("Starting worker for {}...", filename);

    let now = Instant::now();
    let mut command = Command::new(command);
    let args = args.replace("$i", filename);
    if !args.is_empty() {
        command.args(args.split(' '));
    }
    let mut child = command
        .arg(in_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    // So, apparently, I/O is still quite annoying, even in 2021.
    // We need to keep reading from stdout & stderr of the child process
    // because otherwise that may eventually block on writing to them, but
    // we obviously want it to continue calculating stuff.
    // But we can't *read* from them without blocking ourselves in the process,
    // which would prevent us from handling the timeout properly.
    // So we spawn two threads that read stdout and stderr as available, blocking
    // while doing so, and send the data into a channel.
    // To make things a bit easier, we can treat the channel as having an infinite
    // buffer, so we just wait for the child to either exit or time out and then
    // read the channels in their entirety at once.

    let stdout_rx = {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                tx.send(line.unwrap()).unwrap();
            }
        });
        rx
    };
    let stderr_rx = {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                tx.send(line.unwrap()).unwrap();
            }
        });
        rx
    };

    let result = match child
        .wait_timeout(Duration::from_secs(timeout * 60))
        .unwrap()
    {
        Some(status) => {
            let elapsed = now.elapsed().as_secs_f32();
            info!(
                "Completed {} with status {} in {}",
                filename, status, elapsed
            );
            (true, elapsed)
        }
        None => {
            info!("{} timed out!", filename);
            child.kill().unwrap();
            child.wait().unwrap();
            (false, 0.0)
        }
    };

    let mut output_path = out_dir.to_path_buf();
    output_path.push(format!("{}.out", filename));
    let mut output_file = File::create(output_path).unwrap();

    output_file.write_all("== stdout ==\n".as_bytes()).unwrap();

    while let Ok(line) = stdout_rx.recv() {
        writeln!(output_file, "{}", line).unwrap();
    }
    //io::copy(&mut stdout, &mut output_file).unwrap();

    output_file
        .write_all("\n== stderr ==\n".as_bytes())
        .unwrap();

    while let Ok(line) = stderr_rx.recv() {
        writeln!(output_file, "{}", line).unwrap();
    }
    //io::copy(&mut stderr, &mut output_file).unwrap();

    output_file
        .write_all(format!("\n== time ==\nElapsed time: {}", result.1).as_bytes())
        .unwrap();

    output_file.flush().unwrap();

    result
}
