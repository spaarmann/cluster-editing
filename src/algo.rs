use crate::{critical_cliques, graph::IndexMap, Graph};

use log::{info, trace};

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

macro_rules! dbg_trace_indent {
    ($k:expr, $s:expr) => (
        #[cfg(debug)]
        {
            log!($crate::Level::Trace, concat!("{}[k={}]", $s),
                "\t".repeat((unsafe { K_MAX - $k.max(0.0) }) as usize), $k);
        }
    );
    ($k:expr, $s:expr, $($arg:tt)+) => (
        #[cfg(debug)]
        {
            log!($crate::Level::Trace, concat!("{}[k={}]", $s),
                "\t".repeat((unsafe { K_MAX - $k.max(0.0) }) as usize),
                $k, $($arg)+);
        }
    )
}

pub fn find_optimal_cluster_editing(g: &Graph) -> (i32, Vec<Edit>) {
    let mut k = 0.0;

    // TODO: Not sure if executing the algo once with k = 0 is the best
    // way of handling already-disjoint-clique-components.

    let original_node_count = g.size();
    info!(
        "Computing optimal solution for graph with {} nodes.",
        original_node_count
    );

    loop {
        let mut g = g.clone();
        // The imap is used to always have a mapping from the current indices used by the graph to
        // what indices those vertices have in the original graph.
        // The algorithm works on reduced/modified graphs in parts, but when editing those we want
        // to create `Edit` values that are usable on the original graph; we can create those by
        // using the imap.
        let mut imap = IndexMap::identity(g.size());

        info!("[driver] Starting search with k={}, reducing now...", k);
        unsafe {
            K_MAX = k;
        }

        let mut reduced_k = k;
        let edits = Vec::new();
        /*match reduce(&mut g, &mut imap, &mut reduced_k) {
            None => {
                k += 1.0;
                continue;
            }
            Some(reduce_edits) => edits = reduce_edits,
        }*/

        /*info!(
            "[driver] Reduced problem with k={} to k={}, from n={} to n={}",
            k,
            reduced_k,
            original_node_count,
            g.size()
        );*/

        if let Some((_, edits)) = find_cluster_editing(g, imap, edits, reduced_k) {
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
    mut g: Graph,
    imap: IndexMap,
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

            let k_start = k;
            for (i, (comp, comp_imap)) in components.into_iter().enumerate() {
                dbg_trace_indent!(k_start, "Starting component {}, remaining k is {}", i, k);

                // returns early if we can't even find a solution for the component,
                // otherwise take the remaining k and proceed to the next component.
                let comp_res = match find_cluster_editing(comp, comp_imap, edits, k) {
                    None => {
                        dbg_trace_indent!(
                            k_start,
                            "Finished component {} with 'no solution found', returning.",
                            i
                        );
                        return None;
                    }
                    Some(x) => x,
                };
                k = comp_res.0;
                edits = comp_res.1;

                dbg_trace_indent!(k_start, "Finished component {}, remaining k is {}", i, k);
            }

            // We still need to "cash in" any zero-edges that connect the different components.
            let mut zero_count = 0.0;
            for u in 0..g.size() {
                if !g.is_present(u) {
                    continue;
                }
                for v in (u + 1)..g.size() {
                    if !g.is_present(v) {
                        continue;
                    }
                    if component_map[u] != component_map[v] {
                        if g.get_direct(u, v).abs() < 0.001 {
                            zero_count += 1.0;
                        }
                    }
                }
            }

            k -= zero_count / 2.0;

            dbg_trace_indent!(
                k_start,
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

    dbg_trace_indent!(k, "Searching triple");

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
            // No more conflict triples.
            // If there are still zero-edges, we need to "cash in" the 0.5 edit cost we deferred
            // when merging.
            let mut zero_count = 0.0;
            for u in 0..g.size() {
                if !g.is_present(u) {
                    continue;
                }
                for v in (u + 1)..g.size() {
                    if !g.is_present(v) {
                        continue;
                    }
                    if g.get_direct(u, v).abs() < 0.001 {
                        zero_count += 1.0;
                    }
                }
            }

            k -= zero_count / 2.0;

            dbg_trace_indent!(k, "Found no triple, realized {} zero-edges.", zero_count);

            if k < 0.0 {
                // not enough cost left over to actually realize those zero-edges.
                return None;
            }

            return Some((k, edits));
        }
        Some(t) => t,
    };

    dbg_trace_indent!(k, "Found triple ({}-{}-{}), branching", v, u, w);

    // Found a conflict triple, now branch into 2 cases:
    // 1. Set uv to forbidden
    let best = {
        let mut g = g.clone();
        let mut edits = edits.clone();
        let mut imap = imap.clone();
        let uv = g.get_mut(u, v);
        // TODO: Might not need this check after edge merging is in? Maybe?
        if uv.is_finite() {
            let k_before = k;
            let mut k = k - *uv;
            if k >= 0.0 {
                dbg_trace_indent!(
                    k_before,
                    "Branch: Set {}-{} forbidden, k after edit: {} !",
                    u,
                    v,
                    k
                );

                Edit::delete(&mut edits, &imap, u, v);
                *uv = f32::NEG_INFINITY;

                find_cluster_editing(g, imap, edits, k)
            } else {
                dbg_trace_indent!(
                    k_before,
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
            let k_before_merge = k;
            merge(&mut g, &mut imap, &mut k, &mut edits, u, v);

            if k >= 0.0 {
                dbg_trace_indent!(
                    k_before_merge,
                    "Branch: Merge {}-{}, k after merging: {} !",
                    u,
                    v,
                    k
                );

                find_cluster_editing(g, imap, edits, k)
            } else {
                dbg_trace_indent!(
                    k_before_merge,
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

// Merge u and v. The merged vertex becomes the new vertex at index u in the graph, while v is
// marked as not present anymore.
fn merge(
    g: &mut Graph,
    imap: &mut IndexMap,
    k: &mut f32,
    edits: &mut Vec<Edit>,
    u: usize,
    v: usize,
) {
    assert!(g.is_present(u));
    assert!(g.is_present(v));

    let start_k = *k;
    let start_edit_len = edits.len();

    for w in 0..g.size() {
        if w == u || w == v || !g.is_present(w) {
            continue;
        }

        let uw = g.get(u, w);
        let vw = g.get(v, w);

        if uw > 0.0 {
            if vw < 0.0 {
                // (+, -)
                let new_weight = merge_nonmatching_nonzero(g, imap, k, edits, u, v, w);
                g.set(u, w, new_weight);
            } else if vw > 0.0 {
                // (+, +)
                g.set(u, w, uw + vw);
            } else {
                // (+, 0)
                *k -= 0.5;
                Edit::insert(edits, imap, v, w);
            }
        } else if uw < 0.0 {
            if vw < 0.0 {
                // (-, -)
                g.set(u, w, uw + vw);
            } else if vw > 0.0 {
                // (-, +)
                let new_weight = merge_nonmatching_nonzero(g, imap, k, edits, v, u, w);
                g.set(u, w, new_weight);
            } else {
                // (-, 0)
                *k -= 0.5;
            }
        } else {
            if vw < 0.0 {
                // (0, -)
                *k -= 0.5;
                g.set(u, w, vw);
            } else if vw > 0.0 {
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
        start_k,
        "Merged {} and {}. k was {}, is now {}. New edits: {:?}",
        u,
        v,
        start_k,
        k,
        &edits[start_edit_len..]
    );
}

// `merge` helper. Merge uw and vw, under the assumption that weight(uw) > 0 and weight(vw) < 0.
// Adds appropriate edits to `edits` and modifies `k` and returns the correct new weight for the
// merged edge.
fn merge_nonmatching_nonzero(
    g: &Graph,
    imap: &IndexMap,
    k: &mut f32,
    edits: &mut Vec<Edit>,
    u: usize,
    v: usize,
    w: usize,
) -> f32 {
    let uw = g.get(u, w);
    let vw = g.get(v, w);

    if (uw + vw).abs() < 0.001 {
        *k -= uw - 0.5;
        Edit::delete(edits, imap, u, w);
        return 0.0;
    } else {
        if uw > -vw {
            *k -= -vw;
            Edit::insert(edits, imap, v, w);
        } else {
            *k -= uw;
            Edit::delete(edits, imap, u, w);
        }
        return uw + vw;
    }
}

/// Reduces the problem instance. Modifies the mutable arguments directly to be a smaller
/// instance. If this discovers the instance is not solvable at all, returns `None`. Otherwise
/// returns the list of edits performed (which may be empty).
fn reduce(g: &mut Graph, imap: &mut IndexMap, k: &mut f32) -> Option<Vec<Edit>> {
    let old_k = *k;
    let edits = critical_cliques::apply_reductions(g, imap, k);

    if *k < 0.0 {
        dbg_trace_indent!(
            old_k,
            "Found 'no solution' from applying reductions, k now {}",
            k
        );
    } else if old_k > *k {
        dbg_trace_indent!(
            old_k,
            "{} [k={}] Reduced instance from k={} to k={}",
            old_k,
            k
        );
    }

    // TODO: This is just for debugging, should take out at some point
    if *k > 0.0 {
        for u in 0..g.size() {
            if !g.is_present(u) {
                continue;
            }
            for v in (u + 1)..g.size() {
                if !g.is_present(v) {
                    continue;
                }
                assert!(g.get_direct(u, v).is_finite());
            }
        }
    }

    edits
}
