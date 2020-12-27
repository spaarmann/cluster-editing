use log::info;
use petgraph::dot::{Config, Dot};
use petgraph::visit::{GraphProp, GraphRef, IntoEdgeReferences, IntoNodeReferences, NodeIndexable};

use std::fmt::Display;
use std::io::Write;
use std::process::{Command, Stdio};

pub fn print_graph<'a, G, P: AsRef<std::path::Path>>(path: P, graph: G)
where
    G: GraphRef + IntoNodeReferences + IntoEdgeReferences + NodeIndexable + GraphProp,
    G::EdgeWeight: Display,
    G::NodeWeight: Display,
{
    info!("Writing graph image to {}", path.as_ref().display());

    let dot = Dot::with_config(graph, &[Config::EdgeNoLabel]);

    let mut graphviz = Command::new("fdp")
        .arg("-Tpng")
        .arg(format!("-o{}", path.as_ref().display()))
        .stdin(Stdio::piped())
        .spawn()
        .expect("Failed to spawn graphviz process");

    {
        let stdin = graphviz
            .stdin
            .as_mut()
            .expect("Failed to open graphviz stdin pipe.");
        stdin
            .write_all(dot.to_string().as_bytes())
            .expect("Failed to write to graphviz stdin pipe.");
    }

    graphviz.wait().expect("Executing graphviz failed");
}
