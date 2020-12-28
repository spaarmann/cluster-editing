use crate::Graph;

use std::collections::{BTreeSet, HashMap};

use log::{info, trace};
use petgraph::{graph::NodeIndex, visit::EdgeRef};

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

type WeightMap = HashMap<(u32, u32), f32>;

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
                map.insert((u_w, v_w), g.find_edge(u, v).map(|_| 1.0).unwrap_or(-1.0));
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
    let mut k = 0.0;
    let weights = create_weight_map(&g);

    // TODO: Not sure if executing the algo once with k = 0 is the best
    // way of handling already-disjoint-clique-components.

    info!(
        "Computing optimal solution for graph with {} nodes and {} edges.",
        g.node_count(),
        g.edge_count()
    );

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
fn find_cluster_editing(
    mut rg: RGraph,
    mut edits: Vec<Edit>,
    k: f32,
) -> Option<(RGraph, Vec<Edit>)> {
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
        "\t".repeat((unsafe { K_MAX - k.max(0.0) }) as usize),
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
        let res = if k >= 0.0 {
            rg.g.add_edge(v, w, 0);
            edits.push(Edit::Insert((v_i, w_i)));
            rg.set_weight(v_i, w_i, f32::INFINITY);
            rg.set_weight(u_i, w_i, f32::INFINITY);
            rg.set_weight(u_i, v_i, f32::INFINITY);
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
        let res = if k >= 0.0 {
            rg.g.remove_edge(rg.g.find_edge(u, v).unwrap());
            edits.push(Edit::Delete((u_i, v_i)));
            rg.set_weight(u_i, w_i, f32::INFINITY);
            rg.set_weight(u_i, v_i, f32::NEG_INFINITY);
            rg.set_weight(v_i, w_i, f32::NEG_INFINITY);
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
        let res = if k >= 0.0 {
            rg.g.remove_edge(rg.g.find_edge(u, w).unwrap());
            edits.push(Edit::Delete((u_i, w_i)));
            rg.set_weight(u_i, w_i, f32::NEG_INFINITY);
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
fn reduce(rg: &mut RGraph, k: &mut f32) -> Option<Vec<Edit>> {
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

    // induced costs are stored in a `BTreeSet` per vertex, which is in turn stored
    // in this `HashMap`, indexed the same way `RGraph.w` is.
    let mut icps = HashMap::<u32, BTreeSet<InducedCostPair>>::new(); // costs induced by setting an edge permanent
    let mut icfs = HashMap::<u32, BTreeSet<InducedCostPair>>::new(); // costs induced by setting an edge forbidden

    // TODO: This relies on our graph being a StableGraph! Which is not ideal.
    // We store `NodeIndex`s in the heaps which would otherwise get invalidated when we merge
    // vertices.

    // Calculate icf(uv) and icp(uv) for each pair uv in the graph, storing them in the hashmaps +
    // heaps from above.
    for u in rg.g.node_indices() {
        for v in rg.g.node_indices() {
            let u_i = *rg.g.node_weight(u).unwrap();
            let v_i = *rg.g.node_weight(v).unwrap();

            // Only need to compute each edge once.
            if u_i < v_i {
                let mut icf = 0.0;
                let mut icp = 0.0;

                let u_neighbors = rg.g.neighbors(u).collect::<Vec<_>>();
                let v_neighbors = rg.g.neighbors(v).collect::<Vec<_>>();

                for w in &u_neighbors {
                    let w_i = *rg.g.node_weight(*w).unwrap();
                    if v_neighbors.contains(w) {
                        // w in intersection of neighborhoods
                        icf += f32::min(rg.get_weight(u_i, w_i), rg.get_weight(v_i, w_i));
                    } else {
                        // w in symmetric difference of neighborhoods
                        icp +=
                            f32::max(rg.get_weight(u_i, w_i).abs(), rg.get_weight(v_i, w_i).abs());
                    }
                }

                for w in v_neighbors {
                    // intersection is completely handled above,
                    // just need the second part of the symmetric difference here
                    if !u_neighbors.contains(&w) {
                        let w_i = *rg.g.node_weight(w).unwrap();
                        icp +=
                            f32::max(rg.get_weight(u_i, w_i).abs(), rg.get_weight(v_i, w_i).abs());
                    }
                }

                // Insert compute cost into the binary heaps.
                let u_heap_p = icps.entry(u_i).or_default();
                u_heap_p.insert(InducedCostPair {
                    other: v,
                    other_i: v_i,
                    cost: icp,
                });
                let v_heap_p = icps.entry(v_i).or_default();
                v_heap_p.insert(InducedCostPair {
                    other: u,
                    other_i: u_i,
                    cost: icp,
                });

                let u_heap_f = icfs.entry(u_i).or_default();
                u_heap_f.insert(InducedCostPair {
                    other: v,
                    other_i: v_i,
                    cost: icf,
                });
                let v_heap_f = icfs.entry(v_i).or_default();
                v_heap_f.insert(InducedCostPair {
                    other: u,
                    other_i: u_i,
                    cost: icf,
                });
                /*let weight_uv = rg.w[&(u_i, v_i)];
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
                }*/
            }
        }
    }

    loop {
        // Find max icf and icp values.

        // TODO: We currently use BTreeSets to store the icf and icp values.
        // THe paper suggests binary heaps, but the std BinaryHeap does not support
        // removing objects, which we need below. If this is still in the final impl,
        // revisit what, if anything, to do about this. 

        // TODO: Avoid the double lookup here
        let max_p_node =
            rg.g.node_indices()
                .max_by_key(|u| icps[rg.g.node_weight(*u).unwrap()].peek())
                .unwrap();
        let max_f_node =
            rg.g.node_indices()
                .max_by_key(|u| icfs[rg.g.node_weight(*u).unwrap()].peek())
                .unwrap();

        let max_p_i = *rg.g.node_weight(max_p_node).unwrap();
        let max_f_i = *rg.g.node_weight(max_f_node).unwrap();

        let max_p = *icps[&max_p_i].peek().unwrap();
        let max_f = *icfs[&max_f_i].peek().unwrap();

        if max_p.cost < *k && max_f.cost < *k {
            break;
        }

        if max_p > max_f {
            // Set {max_p_i, max_p.other} to permanent
            // Update k
            // Update heaps
        } else {
            let u = max_f_node;
            let u_i = max_f_i;
            let v = max_f.other;
            let v_i = max_f.other_i;

            let weight_uv = rg.get_weight(u_i, v_i);
            // Update k
            *k = *k - weight_uv.abs();
            // Set uv to forbidden
            rg.set_weight(u_i, v_i, f32::NEG_INFINITY);
            if weight_uv <= 0.0 {
                // Edge already didn't exist, nothing further to update.
                continue;
            }

            rg.g.remove_edge(rg.g.find_edge(u, v).unwrap());

            // Update heaps
            for x in rg.g.node_indices() {
                if x == v || x == u {
                    continue;
                }

                let x_i = *rg.g.node_weight(x).unwrap();

                let weight_xv = rg.get_weight(x_i, v_i);

                // Update icf(ux)

                // icf depends on intersection of N(u) and N(x), if v is not in N(x) that can't
                // have changed by removing the edge.
                if weight_xv >= 0.0 {
                    let u_heap = &icfs[&u_i];
                    let x_heap = &icfs[&x_i];

                    u_heap.

                    // Get the old icf(ux) value
                    // Compute the new one
                    // Replace the old with the new in the heap
                }
            }
        }
    }

}

impl RGraph {
    fn get_weight(&self, u: u32, v: u32) -> f32 {
        if u < v {
            self.w[&(u, v)]
        } else if u > v {
            self.w[&(v, u)]
        } else {
            panic!("Tried to get weight of a self-loop!");
        }
    }

    fn set_weight(&mut self, u: u32, v: u32, weight: f32) {
        if u < v {
            self.w.insert((u, v), weight);
        } else if u > v {
            self.w.insert((v, u), weight);
        } else {
            panic!("Tried to set weight of a self-loop!");
        }
    }
}

/// This is used to store the induced costs in a BinaryHeap.
/// It keeps around access to the other end of the edge and the induced
/// cost computed, and provies a `PartialOrd` and `Ord` impl that compares
/// the `cost` only; thus enabling us to sort the heaps correctly.
#[derive(Copy, Clone, PartialEq)]
struct InducedCostPair {
    other: NodeIndex,
    other_i: u32,
    cost: f32,
}

impl PartialOrd for InducedCostPair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InducedCostPair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost
            .partial_cmp(&other.cost)
            .unwrap() // PANIC if cost is NaN!
            .then_with(|| self.other.cmp(&other.other))
            .then_with(|| self.other_i.cmp(&other.other_i))
    }
}

// NEVER produce NaN values for the cost!
impl Eq for InducedCostPair {}
