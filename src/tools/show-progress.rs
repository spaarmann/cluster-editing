#![allow(clippy::redundant_pattern_matching)]

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use lazy_static::lazy_static;
use regex::Regex;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "show-progress",
    about = "Analyzes cluster-editing log files of (potentially cancelled/interrupted) runs and prints how much progress on solving was made."
)]
struct Opt {
    /// Input file or folder. If a file, the log file will be analyzed and maximum progress
    /// printed. If a folder, the same will happen for every file directly (non-recursively) inside the folder.
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    /// If specified, will compare the results of the given run to the first one. If the first is a
    /// file, this must be too, and the same for directories.
    #[structopt(long = "diff", parse(from_os_str))]
    diff: Option<PathBuf>,
}

enum RunProgress {
    Finished {
        filename: String,
        time: f32,
    },
    Cancelled {
        filename: String,
        comp: i32,
        max_k: i32,
    },
}

impl RunProgress {
    fn filename(&self) -> &str {
        match self {
            Self::Finished { filename, time: _ } => filename,
            Self::Cancelled {
                filename,
                comp: _,
                max_k: _,
            } => filename,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let opt = Opt::from_args();

    let base_path = opt.input;
    if base_path.is_file() {
        let result = analyze_file(base_path)?;

        if let Some(diff_path) = opt.diff {
            let diff_result = analyze_file(diff_path)?;
            print_diff(&result, &diff_result);
        } else {
            print_result(&result, "");
        }

        return Ok(());
    }

    let mut results = HashMap::new();

    // It's a directory!
    for entry in base_path.read_dir()? {
        let file = entry?.path();
        if file.metadata()?.is_file() {
            let result = analyze_file(file)?;
            results.insert(result.filename().to_string(), result);
        }
    }

    if let Some(diff_path) = opt.diff {
        let mut diff_results = HashMap::new();
        for entry in diff_path.read_dir()? {
            let file = entry?.path();
            if file.metadata()?.is_file() {
                let result = analyze_file(file)?;
                diff_results.insert(result.filename().to_string(), result);
            }
        }

        for (name, first) in &results {
            if let Some(second) = diff_results.get(name) {
                print_diff(first, second);
            } else {
                print_result(first, " in first, not present in second.");
            }
        }
        for (name, second) in &diff_results {
            if let None = results.get(name) {
                print_result(second, " in second, not present in first.");
            }
        }
    } else {
        for result in &results {
            print_result(result.1, "");
        }
    }

    Ok(())
}

fn analyze_file<P: AsRef<Path>>(file: P) -> Result<RunProgress, Box<dyn Error>> {
    let file = file.as_ref();
    let filename = file.file_name().unwrap().to_str().unwrap();
    let file = File::open(file)?;
    let reader = BufReader::new(file);

    let mut max_component = -1;
    let mut max_k = -1;
    let mut last_comp_finished = -1;

    lazy_static! {
        static ref COMP_START_RE: Regex = Regex::new(r"Solving component (\d+)\.\.\.").unwrap();
        static ref NEW_K_RE: Regex =
            Regex::new(r"\[driver\] Starting search with k=(\d+)\.\.\.").unwrap();
        static ref COMP_DONE_RE: Regex = Regex::new(
            r"Found a cluster editing with k=(\d+) and (\d+) edits for component (\d+):"
        )
        .unwrap();
        static ref GRAPH_DONE_RE: Regex = Regex::new(r"Finished in ([\d\.]+) seconds!").unwrap();
    }

    for line in reader.lines() {
        let line = line.unwrap();
        if let Some(captures) = COMP_START_RE.captures(&line) {
            let comp = captures.get(1).unwrap().as_str().parse().unwrap();
            if comp != max_component + 1 {
                panic!("Jumped from comp {} to {}!", max_component, comp);
            }

            max_component = comp;
            max_k = -1;
            continue;
        }

        if let Some(captures) = NEW_K_RE.captures(&line) {
            let k = captures.get(1).unwrap().as_str().parse().unwrap();
            if max_k != -1 && k != max_k + 1 {
                panic!("Jumped from k {} to {}", max_k, k);
            }

            max_k = k;
            continue;
        }

        if let Some(captures) = COMP_DONE_RE.captures(&line) {
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
            continue;
        }

        if let Some(captures) = GRAPH_DONE_RE.captures(&line) {
            let time = captures.get(1).unwrap().as_str().parse().unwrap();
            return Ok(RunProgress::Finished {
                filename: filename.to_string(),
                time,
            });
        }
    }

    Ok(RunProgress::Cancelled {
        filename: filename.to_string(),
        comp: max_component,
        max_k,
    })
}

fn print_result(result: &RunProgress, suffix: &str) {
    match result {
        RunProgress::Cancelled {
            filename,
            comp,
            max_k,
        } => println!(
            "{} got to component {}, k = {}{}",
            filename, comp, max_k, suffix
        ),
        RunProgress::Finished { filename, time } => {
            println!("{} finished in {} seconds{}", filename, time, suffix)
        }
    }
}

fn print_diff(first: &RunProgress, second: &RunProgress) {
    use RunProgress::*;

    match (first, second) {
        (
            Finished {
                filename: name1,
                time: time1,
            },
            Finished {
                filename: name2,
                time: time2,
            },
        ) => {
            assert_eq!(name1, name2);
            println!(
                "{} finished went from {}s to {}s: {:+}s",
                name1,
                time1,
                time2,
                time2 - time1
            );
        }
        (
            Finished {
                filename: name1,
                time: time1,
            },
            Cancelled {
                filename: name2,
                comp: comp2,
                max_k: k2,
            },
        ) => {
            assert_eq!(name1, name2);
            println!(
                "{} first finished in {}s, but was now cancelled at comp {}, k {}",
                name1, time1, comp2, k2
            );
        }
        (
            Cancelled {
                filename: name1,
                comp: comp1,
                max_k: k1,
            },
            Finished {
                filename: name2,
                time: time2,
            },
        ) => {
            assert_eq!(name1, name2);
            println!(
                "{} was first cancelled at comp {}, k {}, but now finished in {}s",
                name1, comp1, k1, time2
            );
        }
        (
            Cancelled {
                filename: name1,
                comp: comp1,
                max_k: k1,
            },
            Cancelled {
                filename: name2,
                comp: comp2,
                max_k: k2,
            },
        ) => {
            assert_eq!(name1, name2);
            if comp1 == comp2 {
                println!(
                    "{} cancelled at same comp {}, k from {} to {}: {:+}k",
                    name1,
                    comp1,
                    k1,
                    k2,
                    k2 - k1
                );
            } else {
                println!(
                    "{} cancelled first at comp {}, now in comp {}: {:+}comp",
                    name1,
                    comp1,
                    comp2,
                    comp2 - comp1
                );
            }
        }
    }
}
