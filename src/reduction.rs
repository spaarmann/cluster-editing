use crate::{
    algo::ProblemInstance, critical_cliques, graph::GraphWeight, util::InfiniteNum, Graph, Weight,
};

use std::collections::HashSet;

/// The reduction assumes an unweighted graph as input (i.e. one with only weights 1 and -1).
pub fn initial_param_independent_reduction(p: &mut ProblemInstance) -> f32 {
    // Simply merging all critical cliques leads to a graph with at most 4 * k_opt vertices.
    let (g, imap) = critical_cliques::merge_cliques(&p.g, &p.imap, &mut p.path_log);
    p.g = g;
    p.imap = imap;

    let k_start = (p.g.size() / 4) as f32;

    full_param_independent_reduction(p);

    // TODO: It would seem that merging steps above could potentially result in zero-edges in the
    // graph. The algorithm is generally described as requiring that *no* zero-edges are in the
    // input, so doesn't this pose a problem?
    // For now, this checks if we do ever produce any zero-edges and logs an error if so.
    for u in p.g.nodes() {
        for v in p.g.nodes() {
            if u == v {
                continue;
            }
            if p.g.get(u, v).is_zero() {
                log::error!("Produced a zero-edge during parameter-independent reduction!");
            }
        }
    }

    k_start
}

/// Rules from "Böcker et al., Exact Algorithms for Cluster Editing: Evaluation and
/// Experiments", 2011
/// Can be applied during the search as well, can modify `k` appropriately and don't require any
/// specific form of the input.
pub fn full_param_independent_reduction(p: &mut ProblemInstance) {
    // TODO: Optimize at least rules 1-3 ! This is a super naive implementation, even the paper directly
    // describes a better strategy for doing it.

    let _k_start = p.k;
    dbg_trace_indent!(
        p,
        _k_start,
        "Starting full gen-param-indep-reduction, k {}, edits {:?}.",
        p.k,
        p.edits
    );

    let mut applied_any_rule = true;
    while applied_any_rule && p.k > 0.0 {
        applied_any_rule = false;

        let r5 = true;
        let r4 = true;
        let r3 = true;
        let r2 = true;
        let r1 = true;

        // Rule 1 (heavy non-edge rule)
        if r1 {
            applied_any_rule |= rule1(p);
        }

        // Rule 2 (heavy edge rule, single end)
        if r2 {
            applied_any_rule |= rule2(p);
        }

        // Rule 3 (heavy edge rule, both ends)
        if r3 {
            applied_any_rule |= rule3(p);
        }

        if p.g.present_node_count() <= 1 {
            break;
        }

        // Rule 4
        if r4 {
            applied_any_rule |= rule4(p);
        }

        // Try Rule 5 only if no other rules could be applied
        if applied_any_rule {
            continue;
        }

        // Rule 5
        if r5 {
            applied_any_rule = rule5(p);
        }
    }

    dbg_trace_indent!(
        p,
        _k_start,
        "Finished full gen-param-indep-reduction, {}, edits {:?}.",
        p.k,
        p.edits
    );
}

pub fn fast_param_independent_reduction(p: &mut ProblemInstance) {
    // TODO: Optimize at least rules 1-3 ! This is a super naive implementation, even the paper directly
    // describes a better strategy for doing it.

    let _k_start = p.k;
    dbg_trace_indent!(
        p,
        _k_start,
        "Starting fast gen-param-indep-reduction, k {}, edits {:?}.",
        p.k,
        p.edits
    );

    let mut applied_any_rule = true;
    while applied_any_rule && p.k > 0.0 {
        applied_any_rule = false;

        let r3 = true;
        let r2 = true;
        let r1 = true;

        // Rule 1 (heavy non-edge rule)
        if r1 {
            applied_any_rule |= rule1(p);
        }

        // Rule 2 (heavy edge rule, single end)
        if r2 {
            applied_any_rule |= rule2(p);
        }

        // Rule 3 (heavy edge rule, both ends)
        if r3 {
            applied_any_rule |= rule3(p);
        }

        if p.g.present_node_count() <= 1 {
            break;
        }
    }

    dbg_trace_indent!(
        p,
        _k_start,
        "Finished fast gen-param-indep-reduction, {}, edits {:?}.",
        p.k,
        p.edits
    );
}

pub fn rule1(p: &mut ProblemInstance) -> bool {
    let g = &mut p.g;
    let mut applied = false;
    for u in 0..g.size() {
        continue_if_not_present!(g, u);

        let sum = g.neighbors_with_weights(u).map(|(_, weight)| weight).sum();

        for v in 0..g.size() {
            if u == v {
                continue;
            }
            continue_if_not_present!(g, v);

            let uv = g.get(u, v);
            if uv >= Weight::ONE || !uv.is_finite() {
                continue;
            }

            if -uv >= sum {
                p.path_log.push_str(&format!(
                    "rule1, forbidding {:?}-{:?}\n",
                    p.imap[u], p.imap[v]
                ));
                dbg_trace_indent!(p, p.k, "rule1, forbidding {:?}-{:?}", p.imap[u], p.imap[v]);

                g.set(u, v, InfiniteNum::NEG_INFINITY);
                applied = true;
            }
        }
    }
    applied
}

pub fn rule2(p: &mut ProblemInstance) -> bool {
    let mut applied = false;
    for u in 0..p.g.size() {
        continue_if_not_present!(p.g, u);

        let total_sum =
            p.g.nodes()
                .filter_map(|w| {
                    if u == w {
                        None
                    } else {
                        Some(p.g.get(u, w).abs())
                    }
                })
                .sum::<Weight>();

        for v in 0..p.g.size() {
            if u == v {
                continue;
            }
            continue_if_not_present!(p.g, v);

            let uv = p.g.get(u, v);
            if uv <= Weight::ZERO {
                continue;
            }

            let sum = total_sum - uv;

            if uv >= sum {
                dbg_trace_indent!(p, p.k, "rule2 merge {:?}-{:?}", p.imap[u], p.imap[v]);
                p.path_log
                    .push_str(&format!("rule2, merge {:?}-{:?}\n", p.imap[u], p.imap[v]));

                p.merge(u, v);
                applied = true;
            }
        }
    }
    applied
}

pub fn rule3(p: &mut ProblemInstance) -> bool {
    let mut applied = false;
    for u in 0..p.g.size() {
        continue_if_not_present!(p.g, u);
        // This rule is already "symmetric" so we only go through pairs in one order, not
        // either order.

        let sum_u_total =
            p.g.neighbors_with_weights(u)
                .map(|(_, w)| w)
                .sum::<Weight>();

        for v in (u + 1)..p.g.size() {
            continue_if_not_present!(p.g, v);

            let uv = p.g.get(u, v);
            if uv <= Weight::ZERO {
                continue;
            }

            let sum_v_total =
                p.g.neighbors_with_weights(v)
                    .map(|(_, w)| w)
                    .sum::<Weight>();

            let sum = sum_u_total - uv + sum_v_total - uv;

            if uv >= sum {
                dbg_trace_indent!(p, p.k, "rule3 merge {:?}-{:?}", p.imap[u], p.imap[v]);
                p.path_log
                    .push_str(&format!("rule3, merge {:?}-{:?}\n", p.imap[u], p.imap[v]));

                p.merge(u, v);
                applied = true;
            }
        }
    }
    applied
}

pub fn rule4(p: &mut ProblemInstance) -> bool {
    let g = &mut p.g;
    let mut applied = false;
    // TODO: Think through if this doesn't do weird things if we already set some non-edges to
    // NEG_INFINITY
    // TODO: Comment this stuff (including MinCut) and probably clean it up a bit ^^'
    let mut c = HashSet::new();

    // Choose initial u
    let (mut first, mut max) = (usize::MAX, Weight::NEG_INFINITY);
    for u in g.nodes() {
        let mut sum = Weight::ZERO;
        for v in g.nodes() {
            sum += if u == v {
                Weight::ZERO
            } else {
                g.get(u, v).abs()
            };
        }
        if sum > max {
            first = u;
            max = sum;
        }
    }

    c.insert(first);

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
                            p,
                            p.k,
                            "rule4 merge. first {:?} - v {:?}, edits so far {:?}",
                            p.imap[first],
                            p.imap[v],
                            p.edits
                        );

                        p.path_log.push_str(&format!(
                            "rule4, merge {:?}-{:?}\n",
                            p.imap[first], p.imap[v]
                        ));

                        p.merge(first, v);
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

pub fn rule5(p: &mut ProblemInstance) -> bool {
    #[derive(Clone)]
    struct R5Data {
        delta_u: Weight,
        delta_v: Weight,
        max_x: Weight,
        relative_difference: Weight,
        relevant_pairs: Vec<(Weight, Weight)>,
    }

    fn compute_initial_data(g: &Graph<Weight>, u: usize, v: usize) -> R5Data {
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

            if x_edge == y_edge {
                relevant_pairs.push((x, y));
                relative_difference += (x - y).abs();
                if x.is_finite() {
                    max_x += x.abs();
                }
            } else {
                if x_edge {
                    delta_u += x;
                    delta_v -= y;
                } else {
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

    for u in 0..p.g.size() {
        continue_if_not_present!(p.g, u);

        for v in (u + 1)..p.g.size() {
            continue_if_not_present!(p.g, v);

            let uv = p.g.get(u, v);
            if uv <= Weight::ZERO {
                continue;
            }

            let d = compute_initial_data(&p.g, u, v);

            if uv <= Weight::min(d.delta_u, d.delta_v) {
                // No chance of satisfying condition
                continue;
            }

            if uv >= 0.5 * (d.relative_difference + d.delta_u + d.delta_v) {
                // Can always merge, no need to compute the more expensive stuff
                dbg_trace_indent!(
                    p,
                    p.k,
                    "merge before dynprog: {:?}-{:?}, uv {}, rel_diff {}, du {}, dv {}, edits so far: {:?}",
                    p.imap[u],
                    p.imap[v],
                    uv,
                    d.relative_difference,
                    d.delta_u,
                    d.delta_v,
                    p.edits
                );
                p.path_log.push_str(&format!(
                    "rule5, early merge {:?}-{:?}\n",
                    p.imap[u], p.imap[v]
                ));
                p.merge(u, v);
                applied = true;
                continue;
            }

            let _d = d.clone();
            let max = compute_max(d, uv);

            if uv > max {
                dbg_trace_indent!(
                    p,
                    p.k,
                    "merge after dynprog: {:?}-{:?}, uv {}, rel_diff {}, du {}, dv {}, max {}, edits so far: {:?}",
                    p.imap[u],
                    p.imap[v],
                    uv,
                    _d.relative_difference,
                    _d.delta_u,
                    _d.delta_v,
                    max,
                    p.edits
                );
                p.path_log
                    .push_str(&format!("rule5, merge {:?}-{:?}\n", p.imap[u], p.imap[v]));
                p.merge(u, v);
                applied = true;
                continue;
            }

            // TODO: Paper claims it might be good to use the two bounds above to decide in which
            // order the below computation should be executed for the edges, but the PEACE impl
            // doesn't seem to do anything of the sort either, afaict.
        }
    }

    applied
}

fn min_cut(g: &Graph<Weight>, c: &HashSet<usize>, a: usize) -> Weight {
    let mut g = g.fork();
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
        g.set_not_present(v);
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
