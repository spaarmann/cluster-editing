use log::info;
use petgraph::dot::{Config, Dot};
use petgraph::visit::{
    GraphProp, GraphRef, IntoEdgeReferences, IntoNodeReferences, NodeCount, NodeIndexable,
};
use petgraph::{EdgeType, Graph, Undirected};

use std::fmt::{Debug, Display};
use std::io::Write;
use std::process::{Command, Stdio};

pub fn print_graph<'a, G, P: AsRef<std::path::Path>>(command: &str, path: P, graph: G)
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

    let dot = Dot::with_config(graph, &[]);

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
            .write_all(dot.to_string().as_bytes())
            .expect("Failed to write to graphviz stdin pipe.");
    }

    graphviz.wait().expect("Executing graphviz failed");
}

pub fn make_display_graph<N: Debug, E: Debug>(
    g: Graph<N, E, Undirected>,
) -> Graph<String, String, Undirected> {
    let mut out = Graph::with_capacity(g.node_count(), g.edge_count());

    let mut map = std::collections::HashMap::new();
    for v in g.node_indices() {
        map.insert(v, out.add_node(format!("{:?}", g.node_weight(v).unwrap())));
    }

    for e in g.edge_indices() {
        let (e1, e2) = g.edge_endpoints(e).unwrap();
        let u = map[&e1];
        let v = map[&e2];
        out.add_edge(u, v, format!("{:?}", g.edge_weight(e).unwrap()));
    }

    out
}
