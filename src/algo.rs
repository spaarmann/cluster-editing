use crate::{Graph, PetGraph};

use std::collections::HashMap;

use log::{info, trace};
use petgraph::visit::EdgeRef;

pub fn split_into_connected_components(g: &PetGraph) -> Vec<PetGraph> {
    let ccs = petgraph::algo::tarjan_scc(g);

    ccs.into_iter()
        .map(|cc| {
            let mut component = PetGraph::with_capacity(cc.len(), cc.len());
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

//#[derive(Copy, Clone, Debug)]
//pub enum Edge {
//    Original(u32, u32),
//    Merged(u32),
//}

#[derive(Copy, Clone, Debug)]
pub enum Edit {
    Insert((usize, usize)),
    Delete((usize, usize)),
}

pub fn find_optimal_cluster_editing(g: &Graph) -> (i32, Vec<Edit>) {
    let mut k = 0.0;

    // TODO: Not sure if executing the algo once with k = 0 is the best
    // way of handling already-disjoint-clique-components.

    info!(
        "Computing optimal solution for graph with {} nodes.",
        g.node_count()
    );

    loop {
        let g = g.clone();

        trace!("[driver] Starting search with k={}", k);

        unsafe {
            K_MAX = k;
        }

        if let Some((g, edits)) = find_cluster_editing(g, Vec::new(), k) {
            // TODO: Depending on what exactly we need to output, this might
            // need to turn edits for merged vertices etc. back into the expected
            // things.
            return (k as i32, edits);
        }

        k += 1.0;
    }
}

static mut K_MAX: f32 = 0.0;

/// Tries to find a solution of size <= k for the problem instance in `rg`.
/// Returns `None` if none can be found, or the set of edits made if a solution
/// was found.
// `k` is stored as float because it needs to be compared with and changed by values from
// the WeightMap a lot, which are floats.
fn find_cluster_editing(mut g: Graph, mut edits: Vec<Edit>, k: f32) -> Option<(Graph, Vec<Edit>)> {
    // TODO: Finish up the reduction (mainly merging vertices) and enable it.
    //reduce(rg, &mut k);

    // TODO: "If a connected component decomposes into two components, we calculate
    // the optimum solution for these components separately." Figure out the details
    // of what this means and how to do it efficiently.

    trace!(
        "{} [k={}] Searching triple",
        "\t".repeat((unsafe { K_MAX - k.max(0.0) }) as usize),
        k
    );

    // Search for a conflict triple
    // TODO: Surely this can be done a little smarter?
    let mut triple = None;
    'outer: for u in g.nodes() {
        for v in g.nodes() {
            if u == v {
                continue;
            }

            if !g.has_edge(u, v) {
                continue;
            }

            for w in g.nodes() {
                if v == w || u == w {
                    continue;
                }

                if g.has_edge(u, w) && !g.has_edge(v, w) {
                    // `vuw` is a conflict triple!
                    triple = Some((v, u, w));
                    break 'outer;
                }
            }
        }
    }

    let (v, u, w) = match triple {
        None => {
            // No more conflict triples, this graph is solved!
            return Some((g, edits));
        }
        Some(t) => t,
    };

    trace!(
        "{} [k={}] Found triple, branching",
        "\t".repeat((unsafe { K_MAX - k.max(0.0) }) as usize),
        k
    );

    // Found a conflict triple, now branch into 3 cases:

    // 1. Insert vw, set uv, uw, vw to permanent
    {
        let mut g = g.clone();
        let mut edits = edits.clone();
        let vw = g.get_mut(v, w);
        // TODO: Might not need this check after edge merging is in? Maybe?
        if vw.is_finite() {
            let k = k + *vw;
            let res = if k >= 0.0 {
                *vw = f32::INFINITY;
                edits.push(Edit::Insert((v, w)));
                g.set(u, w, f32::INFINITY);
                g.set(u, v, f32::INFINITY);
                find_cluster_editing(g, edits, k)
            } else {
                None
            };

            if res.is_some() {
                return res;
            }
        }
    }

    // 2. Delete uv, set uw to permanent, and set uv and vw to forbidden
    {
        let mut g = g.clone();
        let mut edits = edits.clone();
        let uv = g.get_mut(u, v);
        // TODO: Might not need this check after edge merging is in? Maybe?
        if uv.is_finite() {
            let k = k - *uv;
            let res = if k >= 0.0 {
                *uv = f32::NEG_INFINITY;
                edits.push(Edit::Delete((u, v)));
                g.set(u, w, f32::INFINITY);
                g.set(v, w, f32::NEG_INFINITY);
                find_cluster_editing(g, edits, k)
            } else {
                None
            };

            if res.is_some() {
                return res;
            }
        }
    }

    // 3. Delete uw, set uw to forbidden
    {
        let uw = g.get_mut(u, w);
        // TODO: Might not need this check after edge merging is in? Maybe?
        if uw.is_finite() {
            let k = k - *uw;
            let res = if k >= 0.0 {
                *uw = f32::NEG_INFINITY;
                edits.push(Edit::Delete((u, w)));
                find_cluster_editing(g, edits, k)
            } else {
                None
            };

            if res.is_some() {
                return res;
            }
        }
    }

    None
}

/// Reduces the problem instance. Modifies the mutable arguments directly to be a smaller
/// instance. If this discovers the instance is not solvable at all, returns `None`. Otherwise
/// returns the list of edits performed (which may be empty).
fn reduce(g: &mut Graph, k: &mut f32) -> Option<Vec<Edit>> {
    todo!();
}
