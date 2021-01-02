use crate::{critical_cliques, graph::IndexMap, Graph};

use log::{info, trace};

#[derive(Copy, Clone, Debug)]
pub enum Edit {
    Insert(usize, usize),
    Delete(usize, usize),
}

impl Edit {
    pub fn insert(imap: &IndexMap, u: usize, v: usize) -> Self {
        Self::Insert(imap[u], imap[v])
    }

    pub fn delete(imap: &IndexMap, u: usize, v: usize) -> Self {
        Self::Delete(imap[u], imap[v])
    }
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
        let edits;
        match reduce(&mut g, &mut imap, &mut reduced_k) {
            None => {
                k += 1.0;
                continue;
            }
            Some(reduce_edits) => edits = reduce_edits,
        }

        info!(
            "[driver] Reduced problem with k={} to k={}, from n={} to n={}",
            k,
            reduced_k,
            original_node_count,
            g.size()
        );

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
        let components = g.split_into_components(&imap);
        if components.len() > 1 {
            // If a connected component decomposes into two components, we calculate
            // the optimum solution for these components separately.
            // TODO: Still not entirely convinced why this is actually *correct*.

            for (comp, comp_imap) in components {
                // returns early if we can't even find a solution for the component,
                // otherwise take the remaining k and proceed to the next component.
                let comp_res = find_cluster_editing(comp, comp_imap, edits, k)?;
                k = comp_res.0;
                edits = comp_res.1;
            }

            return Some((k, edits));
        }
    }

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
            return Some((k, edits));
        }
        Some(t) => t,
    };

    trace!(
        "{} [k={}] Found triple, branching",
        "\t".repeat((unsafe { K_MAX - k.max(0.0) }) as usize),
        k
    );

    // Found a conflict triple, now branch into 3 cases:

    // TODO: We now check all 3 branches, even if an earlier one has found a solution and take the
    // most optimal solution, because otherwise it seems like the component handling above is
    // incorrect?

    // 1. Insert vw, set uv, uw, vw to permanent
    let best = {
        let mut g = g.clone();
        let mut edits = edits.clone();
        let imap = imap.clone();
        let vw = g.get_mut(v, w);
        // TODO: Might not need this check after edge merging is in? Maybe?
        if vw.is_finite() {
            let k = k + *vw;
            if k >= 0.0 {
                *vw = f32::INFINITY;
                edits.push(Edit::insert(&imap, v, w));
                g.set(u, w, f32::INFINITY);
                g.set(u, v, f32::INFINITY);
                find_cluster_editing(g, imap, edits, k)
            } else {
                None
            }
        } else {
            None
        }
    };

    // 2. Delete uv, set uw to permanent, and set uv and vw to forbidden
    let res2 = {
        let mut g = g.clone();
        let mut edits = edits.clone();
        let imap = imap.clone();
        let uv = g.get_mut(u, v);
        // TODO: Might not need this check after edge merging is in? Maybe?
        if uv.is_finite() {
            let k = k - *uv;
            if k >= 0.0 {
                *uv = f32::NEG_INFINITY;
                edits.push(Edit::delete(&imap, u, v));
                g.set(u, w, f32::INFINITY);
                g.set(v, w, f32::NEG_INFINITY);
                find_cluster_editing(g, imap, edits, k)
            } else {
                None
            }
        } else {
            None
        }
    };

    let best = match (best, res2) {
        (None, r) => r,
        (r, None) => r,
        (Some((k1, e1)), Some((k2, e2))) => Some(if k1 > k2 { (k1, e1) } else { (k2, e2) }),
    };

    // 3. Delete uw, set uw to forbidden
    let res3 = {
        let uw = g.get_mut(u, w);
        // TODO: Might not need this check after edge merging is in? Maybe?
        if uw.is_finite() {
            let k = k - *uw;
            if k >= 0.0 {
                *uw = f32::NEG_INFINITY;
                edits.push(Edit::delete(&imap, u, w));
                find_cluster_editing(g, imap, edits, k)
            } else {
                None
            }
        } else {
            None
        }
    };

    let best = match (best, res3) {
        (None, r) => r,
        (r, None) => r,
        (Some((k1, e1)), Some((k2, e2))) => Some(if k1 > k2 { (k1, e1) } else { (k2, e2) }),
    };

    best
}

fn merge(
    g: &mut Graph,
    imap: &mut IndexMap,
    k: &mut f32,
    edits: &mut Vec<Edit>,
    u: usize,
    v: usize,
) {
}

/// Reduces the problem instance. Modifies the mutable arguments directly to be a smaller
/// instance. If this discovers the instance is not solvable at all, returns `None`. Otherwise
/// returns the list of edits performed (which may be empty).
fn reduce(g: &mut Graph, imap: &mut IndexMap, k: &mut f32) -> Option<Vec<Edit>> {
    let old_k = *k;

    /*if unsafe { K_MAX == 3.0 } && old_k == 2.0 {
        trace!(
            "{} [k={}] Printing debug graph!",
            "\t".repeat((unsafe { K_MAX - old_k.max(0.0) }) as usize),
            old_k,
        );
        crate::graphviz::print_graph("debug-pre-reduce.png", &g.into_petgraph(None));
    }*/

    let edits = critical_cliques::apply_reductions(g, imap, k);

    /*if unsafe { K_MAX == 3.0 } && old_k == 2.0 {
        crate::graphviz::print_graph("debug-post-reduce.png", &g.into_petgraph(None));
    }*/

    if *k < 0.0 {
        log::trace!(
            "{} [k={}] Found 'no solution' from applying reductions, k now {}",
            "\t".repeat((unsafe { K_MAX - old_k.max(0.0) }) as usize),
            old_k,
            k
        );
    } else if old_k > *k {
        log::trace!(
            "{} [k={}] Reduced instance from k={} to k={}",
            "\t".repeat((unsafe { K_MAX - old_k.max(0.0) }) as usize),
            old_k,
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
