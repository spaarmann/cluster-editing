use crate::{
    graph::{GraphWeight, IndexMap},
    reduction,
    util::InfiniteNum,
    Graph, PetGraph, Weight,
};

use log::info;
use petgraph::graph::NodeIndex;

#[derive(Copy, Clone, Debug)]
pub enum Edit {
    Insert(usize, usize),
    Delete(usize, usize),
}

impl Edit {
    pub fn insert(edits: &mut Vec<Edit>, imap: &IndexMap, u: usize, v: usize) {
        for &orig_u in &imap[u] {
            for &orig_v in &imap[v] {
                edits.push(Self::Insert(orig_u, orig_v));
            }
        }
    }

    pub fn delete(edits: &mut Vec<Edit>, imap: &IndexMap, u: usize, v: usize) {
        for &orig_u in &imap[u] {
            for &orig_v in &imap[v] {
                edits.push(Self::Delete(orig_u, orig_v));
            }
        }
    }
}

pub fn execute_algorithm(graph: &PetGraph) -> PetGraph {
    let mut result = graph.clone();
    let (g, imap) = Graph::<Weight>::new_from_petgraph(&graph);
    let (components, _) = g.split_into_components(&imap);

    info!(
        "Decomposed input graph into {} components",
        components.len()
    );

    for (i, c) in components.into_iter().enumerate() {
        info!("Solving component {}...", i);
        let (cg, imap) = c;
        let (k, edits) = find_optimal_cluster_editing(&cg);

        // TODO: The algorithm can produce "overlapping" edits. It might e.g. have a "delete(uv)"
        // edit followed later by an "insert(uv)" edit. This is handled correctly below when
        // computing the output graph, but ignored when outputting the edit set.

        info!(
            "Found a cluster editing with k={} and {} edits for component {}: {:?}",
            k,
            edits.len(),
            i,
            edits
                .iter()
                .map(|e| match e {
                    Edit::Insert(u, v) => Edit::Insert(imap[*u][0], imap[*v][0]),
                    Edit::Delete(u, v) => Edit::Delete(imap[*u][0], imap[*v][0]),
                })
                .collect::<Vec<_>>()
        );

        for edit in edits {
            match edit {
                Edit::Insert(u, v) => {
                    // This imap is only for mapping from components to the full graph, so each
                    // entry only contains a single vertex.
                    if let None =
                        result.find_edge(NodeIndex::new(imap[u][0]), NodeIndex::new(imap[v][0]))
                    {
                        result.add_edge(NodeIndex::new(imap[u][0]), NodeIndex::new(imap[v][0]), 0);
                    }
                }
                Edit::Delete(u, v) => {
                    if let Some(e) =
                        result.find_edge(NodeIndex::new(imap[u][0]), NodeIndex::new(imap[v][0]))
                    {
                        result.remove_edge(e);
                    }
                }
            };
        }
    }

    // Find and print the actual diff of the graphs, in terms of the vertex indices of the original input
    let mut edits = Vec::new();
    for u in graph.node_indices() {
        for v in graph.node_indices() {
            if u == v {
                continue;
            }
            if v > u {
                continue;
            }

            let original = graph.find_edge(u, v);
            let new = result.find_edge(u, v);

            match (original, new) {
                (None, Some(_)) => edits.push(Edit::Insert(
                    *graph.node_weight(u).unwrap(),
                    *graph.node_weight(v).unwrap(),
                )),
                (Some(_), None) => edits.push(Edit::Delete(
                    *graph.node_weight(u).unwrap(),
                    *graph.node_weight(v).unwrap(),
                )),
                _ => { /* no edit */ }
            }
        }
    }

    info!(
        "Final set of {} de-duplicated edits: {:?}",
        edits.len(),
        edits
    );
    result
}

pub fn find_optimal_cluster_editing(g: &Graph<Weight>) -> (i32, Vec<Edit>) {
    // TODO: Not sure if executing the algo once with k = 0 is the best
    // way of handling already-disjoint-clique-components.

    let original_node_count = g.size();
    info!(
        "Computing optimal solution for graph with {} nodes.",
        original_node_count
    );

    let (reduced_g, imap, edits, k_start) =
        reduction::initial_param_independent_reduction(g, &IndexMap::identity(g.size()));

    info!(
        "Reduced graph from {} nodes to {} nodes using parameter-independent reduction.",
        original_node_count,
        reduced_g.size()
    );

    let mut k = k_start;
    loop {
        let reduced_g = reduced_g.clone();
        // The imap is used to always have a mapping from the current indices used by the graph to
        // what indices those vertices have in the original graph.
        // The algorithm works on reduced/modified graphs in parts, but when editing those we want
        // to create `Edit` values that are usable on the original graph; we can create those by
        // using the imap.
        let imap = imap.clone();

        info!("[driver] Starting search with k={}...", k);
        unsafe {
            K_MAX = k;
        }

        let edits = edits.clone();

        if let Some((_, edits)) = find_cluster_editing(reduced_g, imap, edits, k) {
            return (k as i32, edits);
        }

        k += 1.0;
    }
}

static mut K_MAX: f32 = 0.0;

/// Tries to find a solution of size <= k for the problem instance in `rg`.
/// Returns `None` if none can be found, or the set of edits made and the available cost remaining
/// if a solution was found.
// `k` is stored as float because it needs to be compared with and changed by values from
// the WeightMap a lot, which are floats.
fn find_cluster_editing(
    mut g: Graph<Weight>,
    mut imap: IndexMap,
    mut edits: Vec<Edit>,
    mut k: f32,
) -> Option<(f32, Vec<Edit>)> {
    // If k is already 0, we can only if we currently have a solution; there is no point in trying
    // to do further reductions or splitting as we can't afford any edits anyway.

    if k > 0.0 {
        let (components, component_map) = g.split_into_components(&imap);
        if components.len() > 1 {
            // If a connected component decomposes into two components, we calculate
            // the optimum solution for these components separately.
            // TODO: Still not entirely convinced why this is actually *correct*.

            let _k_start = k;
            for (_i, (comp, comp_imap)) in components.into_iter().enumerate() {
                dbg_trace_indent!(_k_start, "Starting component {}, remaining k is {}", _i, k);

                // returns early if we can't even find a solution for the component,
                // otherwise take the remaining k and proceed to the next component.
                let comp_res = match find_cluster_editing(comp, comp_imap, edits, k) {
                    None => {
                        dbg_trace_indent!(
                            _k_start,
                            "Finished component {} with 'no solution found', returning.",
                            _i
                        );
                        return None;
                    }
                    Some(x) => x,
                };
                k = comp_res.0;
                edits = comp_res.1;

                dbg_trace_indent!(_k_start, "Finished component {}, remaining k is {}", _i, k);
            }

            // We still need to "cash in" any zero-edges that connect the different components.
            let mut zero_count = 0.0;
            for u in 0..g.size() {
                continue_if_not_present!(g, u);
                for v in (u + 1)..g.size() {
                    continue_if_not_present!(g, v);
                    if component_map[u] != component_map[v] {
                        if g.get_direct(u, v) == Weight::ZERO {
                            zero_count += 1.0;
                        }
                    }
                }
            }

            k -= zero_count / 2.0;

            dbg_trace_indent!(
                _k_start,
                "After component split, cashed zero-edges, k now {}",
                k
            );

            if k >= 0.0 {
                return Some((k, edits));
            } else {
                return None;
            }
        }
    }

    dbg_trace_indent!(k, "Performing reduction");
    let _k_start = k;
    reduction::general_param_independent_reduction(&mut g, &mut imap, &mut edits, &mut k);
    dbg_trace_indent!(_k_start, "Reduced from k={} to k={}", _k_start, k);

    if k < 0.0 {
        return None;
    }

    dbg_trace_indent!(_k_start, "Searching triple");

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

    let (v, u, _w) = match triple {
        None => {
            // No more conflict triples.
            // If there are still zero-edges, we need to "cash in" the 0.5 edit cost we deferred
            // when merging.
            let mut zero_count = 0.0;
            for u in 0..g.size() {
                continue_if_not_present!(g, u);
                for v in (u + 1)..g.size() {
                    continue_if_not_present!(g, v);
                    if g.get_direct(u, v).is_zero() {
                        zero_count += 1.0;
                    }
                }
            }

            k -= zero_count / 2.0;

            dbg_trace_indent!(
                _k_start,
                "Found no triple, realized {} zero-edges.",
                zero_count
            );

            if k < 0.0 {
                // not enough cost left over to actually realize those zero-edges.
                return None;
            }

            return Some((k, edits));
        }
        Some(t) => t,
    };

    dbg_trace_indent!(_k_start, "Found triple ({}-{}-{}), branching", v, u, _w);

    // Found a conflict triple, now branch into 2 cases:
    // 1. Set uv to forbidden
    let best = {
        let mut g = g.clone();
        let mut edits = edits.clone();
        let imap = imap.clone();
        let uv = g.get_mut(u, v);
        // TODO: Might not need this check after edge merging is in? Maybe?
        if uv.is_finite() {
            let k = k - *uv as f32;
            if k >= 0.0 {
                dbg_trace_indent!(
                    _k_start,
                    "Branch: Set {}-{} forbidden, k after edit: {} !",
                    u,
                    v,
                    k
                );

                Edit::delete(&mut edits, &imap, u, v);
                *uv = InfiniteNum::NEG_INFINITY;

                find_cluster_editing(g, imap, edits, k)
            } else {
                dbg_trace_indent!(
                    _k_start,
                    "Skipping Branch: Setting {}-{} to forbidden reduces k past 0!",
                    u,
                    v
                );
                None
            }
        } else {
            None
        }
    };

    // 2. Merge uv
    let res2 = {
        let mut g = g.clone();
        let mut edits = edits.clone();
        let mut imap = imap.clone();
        let uv = g.get_mut(u, v);
        // TODO: Might not need this check after edge merging is in? Maybe?
        if uv.is_finite() {
            let mut k = k;
            merge(&mut g, &mut imap, &mut k, &mut edits, u, v);

            if k >= 0.0 {
                dbg_trace_indent!(
                    _k_start,
                    "Branch: Merge {}-{}, k after merging: {} !",
                    u,
                    v,
                    k
                );

                find_cluster_editing(g, imap, edits, k)
            } else {
                dbg_trace_indent!(
                    _k_start,
                    "Skipping Branch: Merging {}-{} reduces k past 0!",
                    u,
                    v
                );
                None
            }
        } else {
            None
        }
    };

    match (best, res2) {
        (None, r) => r,
        (r, None) => r,
        (Some((k1, e1)), Some((k2, e2))) => {
            dbg_trace_indent!(k, "Both branches succeeded, with k1={} and k2={}.", k1, k2);

            Some(if k1 > k2 { (k1, e1) } else { (k2, e2) })
        }
    }
}

/// Merge u and v. The merged vertex becomes the new vertex at index u in the graph, while v is
/// marked as not present anymore.
pub fn merge(
    g: &mut Graph<Weight>,
    imap: &mut IndexMap,
    k: &mut f32,
    edits: &mut Vec<Edit>,
    u: usize,
    v: usize,
) {
    assert!(g.is_present(u));
    assert!(g.is_present(v));

    let _start_k = *k;
    let _start_edit_len = edits.len();

    for w in 0..g.size() {
        if w == u || w == v || !g.is_present(w) {
            continue;
        }

        let uw = g.get(u, w);
        let vw = g.get(v, w);

        if uw > Weight::ZERO {
            if vw < Weight::ZERO {
                // (+, -)
                let new_weight = merge_nonmatching_nonzero(g, imap, k, edits, u, v, w);
                g.set(u, w, new_weight);
            } else if vw > Weight::ZERO {
                // (+, +)
                g.set(u, w, uw + vw);
            } else {
                // (+, 0)
                *k -= 0.5;
                Edit::insert(edits, imap, v, w);
            }
        } else if uw < Weight::ZERO {
            if vw < Weight::ZERO {
                // (-, -)
                g.set(u, w, uw + vw);
            } else if vw > Weight::ZERO {
                // (-, +)
                let new_weight = merge_nonmatching_nonzero(g, imap, k, edits, v, u, w);
                g.set(u, w, new_weight);
            } else {
                // (-, 0)
                *k -= 0.5;
            }
        } else {
            if vw < Weight::ZERO {
                // (0, -)
                *k -= 0.5;
                g.set(u, w, vw);
            } else if vw > Weight::ZERO {
                // (0, +)
                *k -= 0.5;
                g.set(u, w, vw);
                Edit::insert(edits, imap, u, w);
            } else {
                // (0, 0)
                *k -= 0.5;
            }
        }
    }

    g.set_present(v, false);
    let mut imap_v = imap.take(v);
    imap[u].append(&mut imap_v);

    dbg_trace_indent!(
        _start_k,
        "Merged {} and {}. k was {}, is now {}. New edits: {:?}",
        u,
        v,
        _start_k,
        k,
        &edits[_start_edit_len..]
    );
}

// `merge` helper. Merge uw and vw, under the assumption that weight(uw) > 0 and weight(vw) < 0.
// Adds appropriate edits to `edits` and modifies `k` and returns the correct new weight for the
// merged edge.
fn merge_nonmatching_nonzero(
    g: &Graph<Weight>,
    imap: &IndexMap,
    k: &mut f32,
    edits: &mut Vec<Edit>,
    u: usize,
    v: usize,
    w: usize,
) -> Weight {
    let uw = g.get(u, w);
    let vw = g.get(v, w);

    if (uw + vw).is_zero() {
        *k -= uw as f32 - 0.5;
        Edit::delete(edits, imap, u, w);
        return Weight::ZERO;
    } else {
        if uw > -vw {
            *k -= -vw as f32;
            Edit::insert(edits, imap, v, w);
        } else {
            *k -= uw as f32;
            Edit::delete(edits, imap, u, w);
        }
        return uw + vw;
    }
}
