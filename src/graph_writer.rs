use crate::graph::{Graph, GraphWeight, IndexMap};

use std::fmt::Display;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub fn write_graph_tgf<T: GraphWeight + Display, P: AsRef<Path>>(
    g: &Graph<T>,
    imap: Option<&IndexMap>,
    path: P,
) {
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);

    for v in g.nodes() {
        if let Some(map) = imap {
            writeln!(writer, "{} {:?}", v, map[v]).unwrap();
        } else {
            writeln!(writer, "{} {}", v, v).unwrap();
        }
    }

    writeln!(writer, "#").unwrap();

    for u in g.nodes() {
        for v in (u + 1)..g.size() {
            continue_if_not_present!(g, v);

            if g.get(u, v) > T::ZERO {
                writeln!(writer, "{} {}", u, v).unwrap();
            }
        }
    }

    writer.flush().unwrap();
}

pub fn write_graph_peace<T: GraphWeight + Display, P: AsRef<Path>>(
    g: &Graph<T>,
    imap: Option<&IndexMap>,
    path: P,
) {
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);

    for u in g.nodes() {
        for v in (u + 1)..g.size() {
            continue_if_not_present!(g, v);

            let uv = g.get(u, v);

            if let Some(imap) = imap {
                let u_name = if imap[u].len() == 1 {
                    imap[u][0].to_string()
                } else {
                    format!("{:?}", imap[u])
                };
                let v_name = if imap[v].len() == 1 {
                    imap[v][0].to_string()
                } else {
                    format!("{:?}", imap[v])
                };

                writeln!(writer, "{}\t{}\t{}", u_name, v_name, uv).unwrap();
            } else {
                writeln!(writer, "{}\t{}\t{}", u, v, uv).unwrap();
            }
        }
    }

    writer.flush().unwrap();
}
