use crate::{
    algo::{merge, Edit},
    critical_cliques,
    graph::{GraphWeight, IndexMap},
    util::InfiniteNum,
    Graph, Weight,
};

use crate::algo::K_MAX;

use std::collections::HashSet;

/// Performs initial parameter-independent reduction on the graph. A new graph and corresponding IndexMap
/// are returned, along with a list of edits performed. The reduction assumes an unweighted graph
/// as input (i.e. one with only weights 1 and -1).
/// The reduction may also allow placing a lower bounds on the optimal `k` parameter, which is also
/// returned.
pub fn initial_param_independent_reduction(
    g: &Graph<Weight>,
    imap: &IndexMap,
    _final_path_debugs: &mut String,
) -> (Graph<Weight>, IndexMap, Vec<Edit>, f32) {
    // Simply merging all critical cliques leads to a graph with at most 4 * k_opt vertices.
    let (mut g, mut imap) = critical_cliques::merge_cliques(g, imap, _final_path_debugs);
    let k_start = (g.size() / 4) as f32;

    let mut edits = Vec::new();
    general_param_independent_reduction(
        &mut g,
        &mut imap,
        &mut edits,
        &mut 0.0,
        _final_path_debugs,
    );

    // TODO: It would seem that merging steps above could potentially result in zero-edges in the
    // graph. The algorithm is generally described as requiring that *no* zero-edges are in the
    // input, so doesn't this pose a problem?
    // For now, this checks if we do ever produce any zero-edges and logs an error if so.
    for u in 0..g.size() {
        continue_if_not_present!(g, u);
        for v in g.neighbors(u) {
            if g.get(u, v).is_zero() {
                log::error!("Produced a zero-edge during parameter-independent reduction!");
            }
        }
    }

    // TODO: If we never end up getting a parameter-independent reduction that produces edits,
    // remove that return value.
    (g, imap, edits, k_start)
}

/// Rules from "Böcker et al., Exact Algorithms for Cluster Editing: Evaluation and
/// Experiments", 2011
/// Can be applied during the search as well, can modify `k` appropriately and don't require any
/// specific form of the input.
pub fn general_param_independent_reduction(
    g: &mut Graph<Weight>,
    imap: &mut IndexMap,
    edits: &mut Vec<Edit>,
    k: &mut f32,
    _final_path_debugs: &mut String,
) {
    // TODO: Optimize some of these! This is a super naive implementation, even the paper directly
    // describes a better strategy for doing it.

    let _k_start = *k;
    dbg_trace_indent!(
        _k_start,
        "Starting gen-param-indep-reduction, k {}, edits {:?}.",
        k,
        edits
    );

    let mut applied_any_rule = true;
    while applied_any_rule && *k > 0.0 {
        applied_any_rule = false;

        let r5 = true;
        let r4 = true;
        let r3 = true;
        let r2 = true;
        let r1 = true;

        // Rule 1 (heavy non-edge rule)
        if r1 {
            applied_any_rule |= rule1(g, imap, k, _final_path_debugs);
        }

        // Rule 2 (heavy edge rule, single end)
        if r2 {
            applied_any_rule |= rule2(g, imap, edits, k, _final_path_debugs);
        }

        // Rule 3 (heavy edge rule, both ends)
        if r3 {
            applied_any_rule |= rule3(g, imap, edits, k, _final_path_debugs);
        }

        if g.present_node_count() <= 1 {
            break;
        }

        // Rule 4
        if r4 {
            applied_any_rule |= rule4(g, imap, edits, k, _final_path_debugs);
        }

        // Try Rule 5 only if no other rules could be applied
        if applied_any_rule {
            continue;
        }

        // Rule 5
        if r5 {
            applied_any_rule = rule5(g, imap, edits, k, _final_path_debugs);
        }
    }

    dbg_trace_indent!(
        _k_start,
        "Finished gen-param-indep-reduction, {}, edits {:?}.",
        k,
        edits
    );
}

pub fn rule1(
    g: &mut Graph<Weight>,
    imap: &IndexMap,
    k: &f32,
    _final_path_debugs: &mut String,
) -> bool {
    let mut applied = false;
    for u in 0..g.size() {
        continue_if_not_present!(g, u);
        for v in 0..g.size() {
            if u == v {
                continue;
            }
            continue_if_not_present!(g, v);

            let uv = g.get(u, v);
            if uv >= Weight::ONE || !uv.is_finite() {
                continue;
            }

            let sum = g.neighbors(u).map(|w| g.get(u, w)).sum();
            if -uv >= sum {
                _final_path_debugs
                    .push_str(&format!("rule1, forbidding {:?}-{:?}\n", imap[u], imap[v]));
                dbg_trace_indent!(*k, "rule1, forbidding {:?}-{:?}", imap[u], imap[v]);

                g.set(u, v, InfiniteNum::NEG_INFINITY);
                applied = true;
            }
        }
    }
    applied
}

pub fn rule2(
    g: &mut Graph<Weight>,
    imap: &mut IndexMap,
    edits: &mut Vec<Edit>,
    k: &mut f32,
    _final_path_debugs: &mut String,
) -> bool {
    let mut applied = false;
    for u in 0..g.size() {
        continue_if_not_present!(g, u);
        for v in 0..g.size() {
            if u == v {
                continue;
            }
            continue_if_not_present!(g, v);

            let uv = g.get(u, v);
            if uv <= Weight::ZERO {
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
                dbg_trace_indent!(*k, "rule2 merge {:?}-{:?}", imap[u], imap[v]);
                _final_path_debugs.push_str(&format!("rule2, merge {:?}-{:?}\n", imap[u], imap[v]));

                merge(g, imap, k, edits, u, v);
                applied = true;
            }
        }
    }
    applied
}

pub fn rule3(
    g: &mut Graph<Weight>,
    imap: &mut IndexMap,
    edits: &mut Vec<Edit>,
    k: &mut f32,
    _final_path_debugs: &mut String,
) -> bool {
    let mut applied = false;
    for u in 0..g.size() {
        continue_if_not_present!(g, u);
        // This rule is already "symmetric" so we only go through pairs in one order, not
        // either order.
        for v in (u + 1)..g.size() {
            continue_if_not_present!(g, v);

            let uv = g.get_direct(u, v);
            if uv <= Weight::ZERO {
                continue;
            }

            let sum = g
                .neighbors(u)
                .map(|w| if w == v { Weight::ZERO } else { g.get(u, w) })
                .sum::<Weight>()
                + g.neighbors(v)
                    .map(|w| if w == u { Weight::ZERO } else { g.get(v, w) })
                    .sum::<Weight>();

            if uv >= sum {
                dbg_trace_indent!(*k, "rule3 merge {:?}-{:?}", imap[u], imap[v]);
                _final_path_debugs.push_str(&format!("rule3, merge {:?}-{:?}\n", imap[u], imap[v]));

                merge(g, imap, k, edits, u, v);
                applied = true;
            }
        }
    }
    applied
}

pub fn rule4(
    g: &mut Graph<Weight>,
    imap: &mut IndexMap,
    edits: &mut Vec<Edit>,
    k: &mut f32,
    _final_path_debugs: &mut String,
) -> bool {
    let mut applied = false;
    // TODO: Think through if this doesn't do weird things if we already set some non-edges to
    // NEG_INFINITY
    // TODO: Comment this stuff (including MinCut) and probably clean it up a bit ^^'
    let mut c = HashSet::new();
    // Choose initial u
    let first = g
        .nodes()
        .fold(
            (usize::MAX, InfiniteNum::NEG_INFINITY),
            |(max_u, max), u| {
                let val = g
                    .nodes()
                    .map(|v| {
                        if u == v {
                            Weight::ZERO
                        } else {
                            g.get(u, v).abs()
                        }
                    })
                    .sum::<Weight>();
                if val > max {
                    (u, val)
                } else {
                    (max_u, max)
                }
            },
        )
        .0;
    c.insert(first);

    dbg_trace_indent!(*k, "Chose {:?} as first for rule4.", imap[first]);

    loop {
        let mut max = (usize::MAX, InfiniteNum::NEG_INFINITY, 0); // (vertex, connectivity, count of connected vertices in c)
        let mut second = (usize::MAX, InfiniteNum::NEG_INFINITY);

        for w in g.nodes() {
            if c.contains(&w) {
                continue;
            }
            let (sum, connected_count) =
                c.iter().fold((Weight::ZERO, 0), |(sum, mut count), &v| {
                    let vw = g.get(v, w);
                    if vw > Weight::ZERO {
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

        if max.1 < Weight::ZERO && max.1.is_infinite() {
            // Didn't find anything to add.
            break;
        }

        let w = max.0;
        c.insert(w);

        if max.1 > second.1 * 2.0 {
            let k_c = min_cut(&g, &c, first);

            // TODO: This probably double-counts every edge?
            let sum_neg_internal = c
                .iter()
                .map(|&u| {
                    c.iter()
                        .map(|&v| {
                            if u != v {
                                g.get(u, v).min(Weight::ZERO).abs()
                            } else {
                                Weight::ZERO
                            }
                        })
                        .sum::<Weight>()
                })
                .sum::<Weight>();

            let sum_pos_crossing = c
                .iter()
                .map(|&u| {
                    g.nodes()
                        .filter(|v| !c.contains(v))
                        .map(|v| g.get(u, v).max(Weight::ZERO))
                        .sum::<Weight>()
                })
                .sum::<Weight>();

            if k_c > sum_neg_internal + sum_pos_crossing {
                for &v in &c {
                    if v != first {
                        dbg_trace_indent!(
                            *k,
                            "rule4 merge. first {:?} - v {:?}, edits so far {:?}",
                            imap[first],
                            imap[v],
                            edits
                        );

                        _final_path_debugs
                            .push_str(&format!("rule4, merge {:?}-{:?}\n", imap[first], imap[v]));

                        merge(g, imap, k, edits, first, v);
                    }
                }
                applied = true;
                break;
            }
        }

        let connected_count_outside = g
            .nodes()
            .filter(|v| !c.contains(v))
            .filter(|&v| g.get(v, w) > Weight::ZERO)
            .count();

        if connected_count_outside > max.2 {
            break;
        }
    }
    applied
}

pub fn rule5(
    g: &mut Graph<Weight>,
    imap: &mut IndexMap,
    edits: &mut Vec<Edit>,
    k: &mut f32,
    _final_path_debugs: &mut String,
) -> bool {
    #[derive(Clone)]
    struct R5Data {
        delta_u: Weight,
        delta_v: Weight,
        max_x: Weight,
        relative_difference: Weight,
        relevant_pairs: Vec<(Weight, Weight)>,
    }

    fn compute_initial_data(g: &Graph<Weight>, imap: &IndexMap, u: usize, v: usize) -> R5Data {
        let mut delta_u = Weight::ZERO;
        let mut delta_v = Weight::ZERO;
        let mut max_x = Weight::ZERO;
        let mut relative_difference = Weight::ZERO;
        let mut relevant_pairs = Vec::new();

        for w in g.nodes() {
            if w == u || w == v {
                continue;
            }

            let x = g.get(u, w);
            let y = g.get(v, w);

            // Both edges or both non-edges => add tuple, adjust relative_difference
            // One edge, one non-edge => adjust deltas (but differently!)
            let x_edge = x > Weight::ZERO;
            let y_edge = y > Weight::ZERO;

            //log::warn!("w: {:?}, x {}, y {}", imap[w], x, y);

            if x_edge == y_edge {
                relevant_pairs.push((x, y));
                relative_difference += (x - y).abs();
                if x.is_finite() {
                    max_x += x.abs();
                }
            } else {
                if x_edge {
                    //log::warn!("du += {}; dv -= {} (because of {:?})", x, y, imap[w]);
                    delta_u += x;
                    delta_v -= y;
                } else {
                    //log::warn!("du -= {}; dv += {} (because of {:?})", x, y, imap[w]);
                    delta_u -= x;
                    delta_v += y;
                }
            }
        }

        R5Data {
            delta_u,
            delta_v,
            max_x,
            relative_difference,
            relevant_pairs,
        }
    };

    fn compute_max(d: R5Data, upper_bound: Weight) -> Weight {
        let m_size = (2.0 * d.max_x + 1.0).ceil() as usize;
        let mut m: Vec<Option<isize>> = vec![None; m_size];
        let mut m_inf: Option<isize> = None; // Extra entry "in" m for infinity values

        // [-max_x, ..., -1, 0, 1, ..., max_x]
        let max_x = d.max_x as isize;
        let get_idx = |x: isize| (x + max_x) as usize;
        let get_val = |i: usize| (i as isize - max_x);

        // Initialize array
        m[get_idx(0)] = Some(0);

        for (x_j, y_j) in d.relevant_pairs {
            let mut max_value = 0;

            let x = x_j as isize;
            let y = y_j as isize;

            let m_prev = m.clone();

            // TODO: The handling for when y_j is -Inf here seems.. iffy. I've just used
            // saturating_add to handle it somehow, but, uh, not sure how correct that is.
            // Pretty sure the PEACE code just silently under/overflows at some points if
            // that happens?

            if x_j == Weight::NEG_INFINITY {
                // Update only inf field if x_j is -inf.
                for i in 0..m.len() {
                    if let Some(old_val) = m_prev[i] {
                        let val = old_val.saturating_add(y);
                        if m_inf.is_none() || val > m_inf.unwrap() {
                            m_inf = Some(val);
                            max_value = max_value.max(val);
                        }
                    }
                }
            } else {
                m_inf = Some(m_inf.unwrap_or(0).saturating_add(y));
                max_value = max_value.max(m_inf.unwrap());

                for i in 0..m.len() as isize {
                    if let Some(old_val) = m_prev[i as usize] {
                        let idx_down = (i - x) as usize;
                        let val_plus_y = old_val.saturating_add(y);
                        if m_prev[idx_down].map(|p| val_plus_y > p).unwrap_or(true) {
                            m[idx_down] = Some(val_plus_y);
                            max_value = max_value.max(isize::min(get_val(idx_down), val_plus_y));
                        }

                        let idx_up = (i + x) as usize;
                        let val_minus_y = old_val.saturating_sub(y);
                        if m_prev[idx_up].map(|p| val_minus_y > p).unwrap_or(true) {
                            m[idx_up] = Some(val_minus_y);
                            max_value = max_value.max(isize::min(get_val(idx_up), val_minus_y));
                        }

                        if m_inf.map(|p| val_plus_y > p).unwrap_or(true) {
                            m_inf = Some(val_plus_y);
                            max_value = max_value.max(val_plus_y);
                        }
                    }
                }
            }

            let current = max_value as Weight + Weight::min(d.delta_u, d.delta_v);
            if current > upper_bound {
                // Early exit if we already exceed the bound.
                // Current is only going to grow, and at this point we already can't satisfy the
                // final condition anymore.
                return current;
            }
        }

        let mut max_value = Weight::NEG_INFINITY;
        for i in 0..m.len() {
            if let Some(val) = m[i] {
                max_value = max_value.max(Weight::min(
                    get_val(i) as Weight + d.delta_u,
                    val as Weight + d.delta_v,
                ));
            }
        }
        max_value = max_value.max(m_inf.unwrap_or(0) as Weight + d.delta_v);

        max_value
    }

    let mut applied = false;

    for u in 0..g.size() {
        continue_if_not_present!(g, u);

        for v in (u + 1)..g.size() {
            continue_if_not_present!(g, v);

            let uv = g.get_direct(u, v);
            if uv <= Weight::ZERO {
                continue;
            }

            let d = compute_initial_data(g, imap, u, v);

            if uv <= Weight::min(d.delta_u, d.delta_v) {
                // No chance of satisfying condition
                continue;
            }

            if uv >= 0.5 * (d.relative_difference + d.delta_u + d.delta_v) {
                // Can always merge, no need to compute the more expensive stuff
                dbg_trace_indent!(
                    *k,
                    "merge before dynprog: {:?}-{:?}, uv {}, rel_diff {}, du {}, dv {}, edits so far: {:?}",
                    imap[u],
                    imap[v],
                    uv,
                    d.relative_difference,
                    d.delta_u,
                    d.delta_v,
                    edits
                );
                _final_path_debugs
                    .push_str(&format!("rule5, early merge {:?}-{:?}\n", imap[u], imap[v]));
                merge(g, imap, k, edits, u, v);
                applied = true;
                continue;
            }

            let _d = d.clone();
            let max = compute_max(d, uv);

            if uv > max {
                dbg_trace_indent!(
                    *k,
                    "merge after dynprog: {:?}-{:?}, uv {}, rel_diff {}, du {}, dv {}, max {}, edits so far: {:?}",
                    imap[u],
                    imap[v],
                    uv,
                    _d.relative_difference,
                    _d.delta_u,
                    _d.delta_v,
                    max,
                    edits
                );
                _final_path_debugs.push_str(&format!("rule5, merge {:?}-{:?}\n", imap[u], imap[v]));
                merge(g, imap, k, edits, u, v);
                applied = true;
                continue;
            }

            // TODO: Paper claims it might be good to use the two bounds above to decide in which
            // order the below computation should be executed for the edges, but the PEACE impl
            // doesn't seem to do anything of the sort either, afaict.

            /*
            let neighbors_u = g.neighbors(u).collect::<HashSet<_>>();
            let neighbors_v = g.neighbors(v).collect::<HashSet<_>>();

            let n_u = {
                let mut n = neighbors_u
                    .difference(&neighbors_v)
                    .copied()
                    .collect::<HashSet<_>>();
                n.remove(&v);
                n
            };
            let n_v = {
                let mut n = neighbors_v
                    .difference(&neighbors_u)
                    .copied()
                    .collect::<HashSet<_>>();
                n.remove(&u);
                n
            };

            let w = {
                let mut w = g.nodes().collect::<HashSet<_>>();
                w = w.difference(&n_u).copied().collect();
                w = w.difference(&n_v).copied().collect();
                w.remove(&u);
                w.remove(&v);
                w
            };

            let delta_u = weight_to_set(g, u, &n_u) - weight_to_set(g, u, &n_v);
            let delta_v = weight_to_set(g, v, &n_v) - weight_to_set(g, v, &n_u);

            if uv <= Weight::min(delta_u, delta_v) {
                // No chance of satisfying condition
                continue;
            }

            let sum = w
                .iter()
                .map(|&w| (g.get(u, w) - g.get(v, w)).abs())
                .sum::<Weight>();
            if uv >= 0.5 * (sum + delta_u + delta_v) {
                // Can always merge, no need to compute the more expensive stuff
                merge(g, imap, k, edits, u, v);
                applied = true;
                continue;
            }

            // TODO: Apparently it might be good to use the two bounds above to decide in which
            // order the below computation should be executed for the edges.

            let b = w
                .iter()
                .map(|&w| (g.get(u, w), g.get(v, w)))
                .collect::<Vec<_>>();
            let max_x = b
                .iter()
                .filter(|&(x, _)| x.is_finite())
                .map(|&(x, _)| x.abs())
                .sum::<Weight>()
                .ceil() as isize;

            // The indexing here relies on weights actually being integers! We may be storing them
            // as floats for convenient infinity-handling, but any finite weights should always
            // have integer values.

            // This implementation is pretty directly based on that in the PEACE source code, the
            // description of the actual implementation of rule 5 in the paper itself is... kind of
            // bad. It ignores the possible presence of -Infinity values entirely, which means
            // parts of it are actually incorrect/not implementable as described, and it's also
            // rather unclear in various other aspects (use of `x` to mean rather different things
            // all the time, including inside a single formula, a formulation in terms of sets of
            // pairs of edge weights that would seem to be incorrect if there are duplicates of
            // those (sets are not lists, the implementation uses a list), ...).

            // [-max_x, ..., -1, 0, 1, ..., max_x, max_x + 1 => infinity slot]
            let idx_offset = max_x;

            let inf_idx = (max_x * 2 + 1) as usize;

            let mut m_prev = vec![Weight::NEG_INFINITY; (max_x * 2 + 2) as usize];
            m_prev[0 + idx_offset as usize] = Weight::ZERO;

            // Get an index into an m_j array based the `x` value
            let get_idx = |i: Weight| {
                if i.is_finite() {
                    (i.round() as isize + idx_offset) as usize
                } else {
                    inf_idx
                }
            };

            let mut m_next = m_prev.clone();

            for (x_j, y_j) in b {
                if x_j.is_infinite() {}

                for x in -max_x..=max_x {
                    let x = x as Weight;

                    log::warn!(
                        "j {}, x {}, x_j {}, get_idx(x) {}, get_idx(x + x_j) {}, get_idx(x - x_j) {}",
                        j,
                        x,
                        x_j,
                        get_idx(x),
                        get_idx(x + x_j),
                        get_idx(x - x_j)
                    );
                    let first = ms[j - 1][get_idx(x)];
                    let second = ms[j - 1][get_idx(x + x_j)] - y_j;
                    let third = ms[j - 1][get_idx(x - x_j)] + y_j;

                    // This is then m_j[get_idx(x)]
                    m_j.push(first.max(second).max(third));
                }

                // TODO: Hm, is this really always correct?
                // This is then m_j[get_idx(Weight::NEG_INFINITY)]
                m_j.push(Weight::NEG_INFINITY);

                ms.push(m_j);
            }

            let last_m = ms.last().unwrap();

            let max = (-max_x..=max_x)
                .map(|x| x as Weight)
                .fold(Weight::NEG_INFINITY, |max, x| {
                    Weight::max(max, Weight::min(x + delta_u, last_m[get_idx(x)] + delta_v))
                });

            if uv >= max {
                merge(g, imap, k, edits, u, v);
                applied = true;
            }*/
        }
    }

    fn weight_to_set<'a>(
        g: &Graph<Weight>,
        u: usize,
        set: impl IntoIterator<Item = &'a usize>,
    ) -> Weight {
        set.into_iter().map(|&v| g.get(u, v)).sum()
    }

    applied
}

fn min_cut(g: &Graph<Weight>, c: &HashSet<usize>, a: usize) -> Weight {
    let mut g = g.clone();
    let mut c = c.clone();

    fn merge_mc(g: &mut Graph<Weight>, c: &mut HashSet<usize>, u: usize, v: usize) {
        for &w in c.iter() {
            if w == u || w == v {
                continue;
            }
            let uw = g.get(u, w);
            let vw = g.get(v, w);
            if uw + vw > Weight::ZERO {
                g.set(u, w, uw + vw);
            }
        }
        g.set_present(v, false);
        c.remove(&v);
    }

    fn min_cut_phase(g: &mut Graph<Weight>, c: &mut HashSet<usize>, a: usize) -> Weight {
        let mut set = HashSet::new();
        set.insert(a);
        let mut last_two = (a, 0);
        while &set != c {
            let mut best = (0, InfiniteNum::NEG_INFINITY);
            for &y in c.iter() {
                if set.contains(&y) {
                    continue;
                }

                let sum = set.iter().map(|&x| g.get(x, y).abs()).sum::<Weight>();
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

    let mut best = InfiniteNum::INFINITY;
    while c.len() > 1 {
        let cut_weight = min_cut_phase(&mut g, &mut c, a);
        if cut_weight < best {
            best = cut_weight;
        }
    }

    best
}
