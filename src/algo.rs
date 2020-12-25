use crate::Graph;

use std::collections::HashMap;

use petgraph::visit::EdgeRef;

pub fn split_into_connected_components(g: &Graph) -> Vec<Graph> {
    let ccs = petgraph::algo::tarjan_scc(g);

    ccs.into_iter()
        .map(|cc| {
            let mut component = Graph::with_capacity(cc.len(), cc.len());
            let mut index_map = HashMap::with_capacity(cc.len());
            for v in cc {
                let new_idx = component.add_node(g[v]);
                index_map.insert(v, new_idx);

                for e in g.edges(v) {
                    let src = e.source();
                    let tgt = e.target();

                    if index_map.contains_key(&src) && index_map.contains_key(&tgt) {
                        component.add_edge(index_map[&src], index_map[&tgt], *e.weight());
                    }
                }
            }

            component
        })
        .collect()
}
