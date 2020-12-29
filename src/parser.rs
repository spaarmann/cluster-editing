use petgraph::data::{Build, Create};
use std::{
    fs::File,
    io::{self, prelude::*, BufReader},
};

fn make_error(text: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, text)
}

// Both parse_file and parse produce a graph by parsing their input according
// to the input format of the PACE challenge.
// The node weights are set to the index/ID of the node.
// CAUTION: The weights here are 0-indexed, because petgraph is. The input
// files are 1-indexed; so the output should be too.

pub fn parse_file<G, P: AsRef<std::path::Path>>(path: P) -> io::Result<G>
where
    G: Create + Build<NodeWeight = usize>,
    G::EdgeWeight: Default,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    parse(reader)
}

pub fn parse<G, R: BufRead>(reader: R) -> io::Result<G>
where
    G: Create + Build<NodeWeight = usize>,
    G::EdgeWeight: Default,
{
    let mut lines = reader.lines();

    let mut n: Option<usize> = None;
    let mut m: Option<usize> = None;
    while let Some(line) = lines.next() {
        let line = line?;
        match line.bytes().next().unwrap() {
            b'c' => continue,
            b'p' => {
                let mut split = line.split(' ');
                n = split.nth(2).and_then(|s| s.parse().ok());
                m = split.nth(0).and_then(|s| s.parse().ok());
                break;
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Did not find problem descriptor line",
                ))
            }
        }
    }

    let n = n.ok_or(make_error("could not read vertex count"))?;
    let m = m.ok_or(make_error("could not read edge count"))?;

    let mut graph = G::with_capacity(n, m);

    let mut node_ids = Vec::with_capacity(n);
    for i in 0..n {
        node_ids.push(graph.add_node(i));
    }

    for line in lines {
        let line = line?;
        if line.starts_with('c') {
            continue;
        }

        let mut split = line.split(' ');
        let u = split
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or(make_error("invalid edge format"))?;
        let v = split
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or(make_error("invalid edge format"))?;

        graph.add_edge(node_ids[u - 1], node_ids[v - 1], Default::default());
    }

    Ok(graph)
}
