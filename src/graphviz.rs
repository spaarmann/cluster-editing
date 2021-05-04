use log::info;
use petgraph::dot::{Config, Dot};
use petgraph::visit::{
    GraphProp, GraphRef, IntoEdgeReferences, IntoNodeReferences, NodeCount, NodeIndexable,
};

use std::fmt::{Debug, Display};
use std::io::Write;
use std::process::{Command, Stdio};

pub fn print_graph<G, P: AsRef<std::path::Path>>(command: &str, path: P, graph: G)
where
    G: GraphRef + IntoNodeReferences + IntoEdgeReferences + NodeIndexable + GraphProp + NodeCount,
    G::EdgeWeight: Display,
    G::NodeWeight: Display,
{
    info!(
        "Writing graph image to {}, graph has {} nodes",
        path.as_ref().display(),
        graph.node_count()
    );

    let dot = Dot::with_config(graph, &[Config::EdgeNoLabel]);
    print(command, path, &dot.to_string());
}

pub fn print_graph_vecs<G, P: AsRef<std::path::Path>>(command: &str, path: P, graph: G)
where
    G: GraphRef + IntoNodeReferences + IntoEdgeReferences + NodeIndexable + GraphProp + NodeCount,
    G::EdgeWeight: Display + Debug,
    G::NodeWeight: Debug,
{
    info!(
        "Writing graph image to {}, graph has {} nodes",
        path.as_ref().display(),
        graph.node_count()
    );

    let dot = Dot::with_config(graph, &[Config::EdgeNoLabel]);
    print(command, path, &format!("{:?}", dot));
}

pub fn print_debug_graph<G, P: AsRef<std::path::Path>>(command: &str, path: P, graph: G)
where
    G: GraphRef + IntoNodeReferences + IntoEdgeReferences + NodeIndexable + GraphProp + NodeCount,
    G::EdgeWeight: Debug,
    G::NodeWeight: Debug,
{
    info!(
        "Writing debug graph image to {}, graph has {} nodes",
        path.as_ref().display(),
        graph.node_count()
    );

    let dot = Dot::with_config(graph, &[]);
    print(command, path, &format!("{:?}", dot));
}

fn print<P: AsRef<std::path::Path>>(command: &str, path: P, dot_str: &str) {
    let mut graphviz = Command::new(command)
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
            .write_all(dot_str.as_bytes())
            .expect("Failed to write to graphviz stdin pipe.");
    }

    graphviz.wait().expect("Executing graphviz failed");
}
