use crate::Graph;

use std::collections::{BinaryHeap, HashMap};

use log::trace;
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

type WeightMap = HashMap<(u32, u32), i32>;

/// Creates a map storing the edge weights used internally by the algorithm from an input graph.
/// Map will contain all pairs (u, v), u < v as keys where u and v are the indices of the vertices in the
/// input graph; these are stored as weights in the petgraph structures.
/// Edges existing in the input graph are assigned weight 1, edges not existing are assigned -1.
fn create_weight_map(g: &Graph) -> WeightMap {
    let mut map = HashMap::new();

    for u in g.node_indices() {
        for v in g.node_indices() {
            let u_w = *g.node_weight(u).unwrap();
            let v_w = *g.node_weight(v).unwrap();
            if u_w < v_w {
                map.insert((u_w, v_w), g.find_edge(u, v).map(|_| 1).unwrap_or(-1));
            }
        }
    }

    map
}

//#[derive(Copy, Clone, Debug)]
//pub enum Edge {
//    Original(u32, u32),
//    Merged(u32),
//}

#[derive(Copy, Clone, Debug)]
pub enum Edit {
    Insert((u32, u32)),
    Delete((u32, u32)),
}

/// Graph structure used internally by the algorithm to store
/// the petgraph Graph, the weight map, and a mapping of merged
/// vertices to what they were originally.
#[derive(Clone)]
struct RGraph {
    g: Graph,
    w: WeightMap,
    merged: HashMap<u32, (u32, u32)>,
}

pub fn find_optimal_cluster_editing(g: &Graph) -> (i32, Vec<Edit>) {
    let mut k = 0;
    let weights = create_weight_map(&g);

    // TODO: Not sure if executing the algo once with k = 0 is the best
    // way of handling already-disjoint-clique-components.

    loop {
        let rg = RGraph {
            g: g.clone(),
            w: weights.clone(),
            merged: HashMap::new(),
        };

        trace!("[driver] Starting search with k={}", k);

        unsafe {
            K_MAX = k;
        }

        if let Some((rg, edits)) = find_cluster_editing(rg, Vec::new(), k) {
            // TODO: Depending on what exactly we need to output, this might
            // need to turn edits for merged vertices etc. back into the expected
            // things.
            return (k, edits);
        }

        k += 1;
    }
}

const POS_INF: i32 = i32::MAX;
const NEG_INF: i32 = i32::MIN;

static mut K_MAX: i32 = 0;

/// Tries to find a solution of size <= k for the problem instance in `rg`.
/// Returns `None` if none can be found, or the set of edits made if a solution
/// was found.
// `k` is always non-negative, but we use an `i32` to store it because we need to
// compare and add/subtract it with other signed values almost exclusively, and would
// thus need to sprinkle `as` casts everywhere otherwise.
fn find_cluster_editing(
    mut rg: RGraph,
    mut edits: Vec<Edit>,
    k: i32,
) -> Option<(RGraph, Vec<Edit>)> {
    // TODO: Finish up the reduction (mainly merging vertices) and enable it.
    //reduce(rg, &mut k);

    // TODO: "If a connected component decomposes into two components, we calculate
    // the optimum solution for these components separately." Figure out the details
    // of what this means and how to do it efficiently.

    trace!(
        "{} [k={}] Searching triple",
        "\t".repeat((unsafe { K_MAX - k.max(0) }) as usize),
        k
    );

    // Search for a conflict triple
    // TODO: Surely this can be done a little smarter?
    let mut triple = None;
    'outer: for u in rg.g.node_indices() {
        for v in rg.g.node_indices() {
            let uv = rg.g.find_edge(u, v);

            if uv.is_none() {
                continue;
            }

            for w in rg.g.node_indices() {
                if v == w || u == w {
                    continue;
                }

                let uw = rg.g.find_edge(u, w);
                let vw = rg.g.find_edge(v, w);

                if uw.is_some() && vw.is_none() {
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
            return Some((rg, edits));
        }
        Some(t) => t,
    };

    trace!(
        "{} [k={}] Found triple, branching",
        "\t".repeat((unsafe { K_MAX - k.max(0) }) as usize),
        k
    );

    let v_i = *rg.g.node_weight(v).unwrap();
    let u_i = *rg.g.node_weight(u).unwrap();
    let w_i = *rg.g.node_weight(w).unwrap();

    // Found a conflict triple, now branch into 3 cases:

    // 1. Insert vw, set uv, uw, vw to permanent
    {
        let mut rg = rg.clone();
        let mut edits = edits.clone();
        let k = k + rg.get_weight(v_i, w_i);
        let res = if k >= 0 {
            rg.g.add_edge(v, w, 0);
            edits.push(Edit::Insert((v_i, w_i)));
            rg.set_weight(v_i, w_i, POS_INF);
            rg.set_weight(u_i, w_i, POS_INF);
            rg.set_weight(u_i, v_i, POS_INF);
            find_cluster_editing(rg, edits, k)
        } else {
            None
        };

        if res.is_some() {
            return res;
        }
    }

    // 2. Delete uv, set uw to permanent, and set uv and vw to forbidden
    {
        let mut rg = rg.clone();
        let mut edits = edits.clone();
        let k = k - rg.get_weight(u_i, v_i);
        let res = if k >= 0 {
            rg.g.remove_edge(rg.g.find_edge(u, v).unwrap());
            edits.push(Edit::Delete((u_i, v_i)));
            rg.set_weight(u_i, w_i, POS_INF);
            rg.set_weight(u_i, v_i, NEG_INF);
            rg.set_weight(v_i, w_i, NEG_INF);
            find_cluster_editing(rg, edits, k)
        } else {
            None
        };

        if res.is_some() {
            return res;
        }
    }

    // 3. Delete uw, set uw to forbidden
    {
        let k = k - rg.get_weight(u_i, w_i);
        let res = if k >= 0 {
            rg.g.remove_edge(rg.g.find_edge(u, w).unwrap());
            edits.push(Edit::Delete((u_i, w_i)));
            rg.set_weight(u_i, w_i, NEG_INF);
            find_cluster_editing(rg, edits, k)
        } else {
            None
        };

        if res.is_some() {
            return res;
        }
    }

    None
}

/// Reduces the problem instance. Modifies the mutable arguments directly to be a smaller
/// instance. If this discovers the instance is not solvable at all, returns `None`. Otherwise
/// returns the list of edits performed (which may be empty).
fn reduce(rg: &mut RGraph, k: &mut i32) -> Option<Vec<Edit>> {
    // TODO: Böcker et al.: A fixed-parameter approach for weighted cluster editing includes
    // a "remove cliques" step; not entirely clear what they mean exactly or how to implement
    // it efficiently yet.
    // Dehne et al. describe a kernelization rule as "delete the connected components that are
    // cliques", which is the interpretation of the above that at least makes immediate sense and
    // is obviously correct.
    // This seems a little weird as a reduction step though; we should be handling each component
    // separately anyway (even if we decompose into new components during the algorithm, see TODO
    // above in `find_cluster_editing`). So this would seem to end up meaning "if the current
    // component is a clique, just stop"; which, if implemented as an addition check at the start,
    // would save us the effort of trying to reduce the clique I suppose. Might be worth doing, but
    // that should probably be measured.

    // Böcker et al.: A fixed-parameter approach for weighted cluster editing, section 3 rule 2
    // Check for unaffordable edge modifications
    for u in rg.g.node_indices() {
        for v in rg.g.node_indices() {
            let u_i = *rg.g.node_weight(u).unwrap();
            let v_i = *rg.g.node_weight(v).unwrap();
            if u_i < v_i {
                let mut icf = 0;
                let mut icp = 0;

                let u_neighbors = rg.g.neighbors(u).collect::<Vec<_>>();
                let v_neighbors = rg.g.neighbors(v).collect::<Vec<_>>();

                for w in &u_neighbors {
                    let w_i = *rg.g.node_weight(*w).unwrap();
                    if v_neighbors.contains(w) {
                        // w in intersection of neighborhoods
                        icf += i32::min(rg.get_weight(u_i, w_i), rg.get_weight(v_i, w_i));
                    } else {
                        // w in symmetric difference of neighborhoods
                        icp +=
                            i32::max(rg.get_weight(u_i, w_i).abs(), rg.get_weight(v_i, w_i).abs());
                    }
                }

                for w in v_neighbors {
                    // intersection is completely handled above,
                    // just need the second part of the symmetric difference here
                    if !u_neighbors.contains(&w) {
                        let w_i = *rg.g.node_weight(w).unwrap();
                        icp +=
                            i32::max(rg.get_weight(u_i, w_i).abs(), rg.get_weight(v_i, w_i).abs());
                    }
                }

                let weight_uv = rg.w[&(u_i, v_i)];
                let should_be_permanent = weight_uv.max(0) + icf > *k;
                let should_be_forbidden = (-weight_uv).max(0) + icp > *k;

                if should_be_permanent && should_be_forbidden {
                    // Instance not solvable!
                    return None;
                }
                if should_be_permanent {
                    if weight_uv < 0 {
                        rg.g.add_edge(u, v, 0);
                        *k += weight_uv;
                    }
                    rg.w.insert((u_i, v_i), POS_INF);
                }
                if should_be_forbidden {
                    if weight_uv > 0 {
                        rg.g.remove_edge(rg.g.find_edge(u, v).unwrap());
                        *k -= weight_uv;
                    }
                    rg.w.insert((u_i, v_i), NEG_INF);
                }
            }
        }
    }

    todo!();
}

impl RGraph {
    fn get_weight(&self, u: u32, v: u32) -> i32 {
        if u < v {
            self.w[&(u, v)]
        } else if u > v {
            self.w[&(v, u)]
        } else {
            panic!("Tried to get weight of a self-loop!");
        }
    }

    fn set_weight(&mut self, u: u32, v: u32, weight: i32) {
        if u < v {
            self.w.insert((u, v), weight);
        } else if u > v {
            self.w.insert((v, u), weight);
        } else {
            panic!("Tried to set weight of a self-loop!");
        }
    }
}
