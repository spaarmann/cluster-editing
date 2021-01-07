use crate::{critical_cliques, graph::IndexMap, Graph};

use std::collections::HashSet;

use log::info;

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
        #[cfg(not(release))]
        {
            log::log!(log::Level::Trace, concat!("{}[k={}]", $s),
                "\t".repeat((unsafe { K_MAX - $k.max(0.0) }) as usize), $k);
        }
    );
    ($k:expr, $s:expr, $($arg:tt)+) => (
        #[cfg(not(release))]
        {
            log::log!(log::Level::Trace, concat!("{}[k={}]", $s),
                "\t".repeat((unsafe { K_MAX - $k.max(0.0) }) as usize),
                $k, $($arg)+);
        }
    )
}

/// `continue_if_not_present(g, u)` executes a `continue` statement if vertex `u` is not present in
/// Graph `g`.
macro_rules! continue_if_not_present {
    ($g:expr, $u:expr) => {
        if !$g.is_present($u) {
            continue;
        }
    };
}

pub fn find_optimal_cluster_editing(g: &Graph) -> (i32, Vec<Edit>) {
    // TODO: Not sure if executing the algo once with k = 0 is the best
    // way of handling already-disjoint-clique-components.

    let original_node_count = g.size();
    info!(
        "Computing optimal solution for graph with {} nodes.",
        original_node_count
    );

    let (reduced_g, imap, edits, k_start) =
        param_independent_reduction(g, &IndexMap::identity(g.size()));

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
    g: Graph,
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
                        if g.get_direct(u, v).abs() < 0.001 {
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

    dbg_trace_indent!(k, "Found triple ({}-{}-{}), branching", v, u, _w);

    // Found a conflict triple, now branch into 2 cases:
    // 1. Set uv to forbidden
    let best = {
        let mut g = g.clone();
        let mut edits = edits.clone();
        let imap = imap.clone();
        let uv = g.get_mut(u, v);
        // TODO: Might not need this check after edge merging is in? Maybe?
        if uv.is_finite() {
            let _k_before = k;
            let k = k - *uv;
            if k >= 0.0 {
                dbg_trace_indent!(
                    _k_before,
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
                    _k_before,
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
            let _k_before_merge = k;
            merge(&mut g, &mut imap, &mut k, &mut edits, u, v);

            if k >= 0.0 {
                dbg_trace_indent!(
                    _k_before_merge,
                    "Branch: Merge {}-{}, k after merging: {} !",
                    u,
                    v,
                    k
                );

                find_cluster_editing(g, imap, edits, k)
            } else {
                dbg_trace_indent!(
                    _k_before_merge,
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

    let _start_k = *k;
    let _start_edit_len = edits.len();

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

/// Performs parameter-independent reduction on the graph. A new graph and corresponding IndexMap
/// are returned, along with a list of edits performed.
/// The reduction may also allow placing a lower bounds on the optimal `k` parameter, which is also
/// returned.
pub fn param_independent_reduction(
    g: &Graph,
    imap: &IndexMap,
) -> (Graph, IndexMap, Vec<Edit>, f32) {
    // Simply merging all critical cliques leads to a graph with at most 4 * k_opt vertices.
    let (g, imap) = critical_cliques::merge_cliques(g, imap);
    let k_start = (g.size() / 4) as f32;

    // Rules from "Böcker et al., Exact Algorithms for Cluster Editing: Evaluation and
    // Experiments", 2011
    let (g, imap, edits) = apply_reduction_rules(g.clone(), imap.clone());

    // TODO: If we never end up getting a parameter-independent reduction that produces edits,
    // remove that return value.
    (g, imap, edits, k_start)
}

fn apply_reduction_rules(mut g: Graph, mut imap: IndexMap) -> (Graph, IndexMap, Vec<Edit>) {
    let mut edits = Vec::new();

    // TODO: Optimize these! This is a super naive implementation, even the paper directly
    // describes a better strategy for doing it.

    let mut applied_any_rule = true;
    while applied_any_rule {
        applied_any_rule = false;

        // Rule 1 (heavy non-edge rule)
        for u in 0..g.size() {
            continue_if_not_present!(g, u);
            for v in 0..g.size() {
                if u == v {
                    continue;
                }
                continue_if_not_present!(g, v);

                let uv = g.get(u, v);
                if uv >= 0.0 || !uv.is_finite() {
                    continue;
                }

                let sum = g.neighbors(u).map(|w| g.get(u, w)).sum();
                if -uv >= sum {
                    g.set(u, v, f32::NEG_INFINITY);
                    applied_any_rule = true;
                }
            }
        }

        // Rule 2 (heavy edge rule, single end)
        for u in 0..g.size() {
            continue_if_not_present!(g, u);
            for v in 0..g.size() {
                if u == v {
                    continue;
                }
                continue_if_not_present!(g, v);

                let uv = g.get(u, v);
                if uv <= 0.0 {
                    continue;
                }

                let sum = g
                    .nodes()
                    .filter_map(|w| {
                        if u == w || v == w {
                            None
                        } else {
                            Some(g.get(u, w).abs())
                        }
                    })
                    .sum();

                if uv >= sum {
                    merge(&mut g, &mut imap, &mut 0.0, &mut edits, u, v);
                    applied_any_rule = true;
                }
            }
        }

        // Rule 3 (heavy edge rule, both ends)
        for u in 0..g.size() {
            continue_if_not_present!(g, u);
            // This rule is already "symmetric" so we only go through pairs in one order, not
            // either order.
            for v in (u + 1)..g.size() {
                continue_if_not_present!(g, v);

                let uv = g.get_direct(u, v);
                if uv <= 0.0 {
                    continue;
                }

                let sum = g
                    .neighbors(u)
                    .map(|w| if w == v { 0.0 } else { g.get(u, w) })
                    .sum::<f32>()
                    + g.neighbors(v)
                        .map(|w| if w == u { 0.0 } else { g.get(v, w) })
                        .sum::<f32>();

                if uv >= sum {
                    merge(&mut g, &mut imap, &mut 0.0, &mut edits, u, v);
                    applied_any_rule = true;
                }
            }
        }

        // TODO: Think through if this doesn't do weird things if we already set some non-edges to
        // NEG_INFINITY
        // TODO: Comment this stuff (including MinCut) and probably clean it up a bit ^^'

        // Rule 4
        if g.present_node_count() <= 1 {
            break;
        }

        let mut c = HashSet::new();
        // Choose initial u
        let first = g
            .nodes()
            .fold((usize::MAX, f32::NEG_INFINITY), |(max_u, max), u| {
                let val = g
                    .nodes()
                    .map(|v| if u == v { 0.0 } else { g.get(u, v).abs() })
                    .sum::<f32>();
                if val > max {
                    (u, val)
                } else {
                    (max_u, max)
                }
            })
            .0;
        c.insert(first);

        loop {
            let mut max = (usize::MAX, f32::NEG_INFINITY, 0); // (vertex, connectivity, count of connected vertices in c)
            let mut second = (usize::MAX, f32::NEG_INFINITY);

            let mut iter_count = 0;
            for w in g.nodes() {
                if c.contains(&w) {
                    continue;
                }
                iter_count += 1;
                let (sum, connected_count) = c.iter().fold((0.0, 0), |(sum, mut count), &v| {
                    let vw = g.get(v, w);
                    if vw > 0.0 {
                        count += 1;
                    }
                    (sum + vw, count)
                });
                if sum > max.1 {
                    second = (max.0, max.1);
                    max = (w, sum, connected_count);
                } else if sum > second.1 {
                    second = (w, sum);
                }
            }

            if max.1 < 0.0 && max.1.is_infinite() {
                // Didn't find anything to add.
                break;
            }

            let w = max.0;
            c.insert(w);

            if max.1 > second.1 * 2.0 {
                let k_c = min_cut(&g, &c, first);

                let sum_neg_internal = c
                    .iter()
                    .map(|&u| {
                        c.iter()
                            .map(|&v| {
                                if u != v {
                                    g.get(u, v).max(0.0).abs()
                                } else {
                                    0.0
                                }
                            })
                            .sum::<f32>()
                    })
                    .sum::<f32>();

                let sum_pos_crossing = c
                    .iter()
                    .map(|&u| {
                        g.nodes()
                            .filter(|v| !c.contains(v))
                            .map(|v| g.get(u, v).min(0.0))
                            .sum::<f32>()
                    })
                    .sum::<f32>();

                if k_c > sum_neg_internal + sum_pos_crossing {
                    let mut nodes = c.into_iter();
                    let first = nodes.next().unwrap();
                    for v in nodes {
                        merge(&mut g, &mut imap, &mut 0.0, &mut edits, first, v);
                    }
                    applied_any_rule = true;
                    break;
                }
            }

            let connected_count_outside = g
                .nodes()
                .filter(|v| !c.contains(v))
                .filter(|&v| g.get(v, w) > 0.0)
                .count();

            if connected_count_outside > max.2 {
                break;
            }
        }
    }

    // TODO: It would seem that merging steps above could potentially result in zero-edges in the
    // graph. The algorithm is generally described as requiring that *no* zero-edges are in the
    // input, so doesn't this pose a problem?
    // For now, this checks if we do ever produce any zero-edges and logs an error if so.
    for u in 0..g.size() {
        continue_if_not_present!(g, u);
        for v in g.neighbors(u) {
            if g.get(u, v).abs() <= 0.001 {
                log::error!("Produced a zero-edge during parameter-independent reduction!");
            }
        }
    }

    // TODO: Consider condensing the Graph after this (i.e. produce a smaller-sized graph that only
    // contains the vertices that are still "present" in g)
    (g, imap, edits)
}

fn min_cut(g: &Graph, c: &HashSet<usize>, a: usize) -> f32 {
    let mut g = g.clone();
    let mut c = c.clone();

    fn merge_mc(g: &mut Graph, c: &mut HashSet<usize>, u: usize, v: usize) {
        for &w in c.iter() {
            if w == u || w == v {
                continue;
            }
            let uw = g.get(u, w);
            let vw = g.get(v, w);
            if uw + vw > 0.0 {
                g.set(u, w, uw + vw);
            }
        }
        g.set_present(v, false);
        c.remove(&v);
    }

    fn min_cut_phase(g: &mut Graph, c: &mut HashSet<usize>, a: usize) -> f32 {
        let mut set = HashSet::new();
        set.insert(a);
        let mut last_two = (a, 0);
        while &set != c {
            let mut best = (0, f32::NEG_INFINITY);
            for &y in c.iter() {
                if set.contains(&y) {
                    continue;
                }

                let sum = set.iter().map(|&x| g.get(x, y).abs()).sum();
                if sum > best.1 {
                    best = (y, sum);
                }
            }

            set.insert(best.0);
            last_two = (best.0, last_two.0);
        }

        let (t, s) = last_two;
        let cut_weight = c.iter().filter(|&&v| v != t).map(|&v| g.get(t, v)).sum();

        merge_mc(g, c, s, t);
        cut_weight
    }

    let mut best = f32::INFINITY;
    while c.len() > 1 {
        let cut_weight = min_cut_phase(&mut g, &mut c, a);
        if cut_weight < best {
            best = cut_weight;
        }
    }

    best
}

/*
/// Performs parameter-depdenent reduction on the graph, modifying it directly. May also modify the
/// parameter.
/// The reduction find that no solution exists, in that case is returns `false`. Otherwise it
/// returns `true`.
fn param_dependent_reduction(
    g: &mut Graph,
    imap: &mut IndexMap,
    k: &mut f32,
    edits: &mut Vec<Edit>,
) -> bool {
    todo!()
}
*/
