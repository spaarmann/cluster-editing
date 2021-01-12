use crate::{
    algo::{merge, Edit},
    critical_cliques,
    graph::{GraphWeight, IndexMap},
    util::InfiniteNum,
    Graph, Weight,
};

use std::collections::HashSet;

/// Performs initial parameter-independent reduction on the graph. A new graph and corresponding IndexMap
/// are returned, along with a list of edits performed. The reduction assumes an unweighted graph
/// as input (i.e. one with only weights 1 and -1).
/// The reduction may also allow placing a lower bounds on the optimal `k` parameter, which is also
/// returned.
pub fn initial_param_independent_reduction(
    g: &Graph<Weight>,
    imap: &IndexMap,
) -> (Graph<Weight>, IndexMap, Vec<Edit>, f32) {
    // Simply merging all critical cliques leads to a graph with at most 4 * k_opt vertices.
    let (mut g, mut imap) = critical_cliques::merge_cliques(g, imap);
    let k_start = (g.size() / 4) as f32;

    let mut edits = Vec::new();
    general_param_independent_reduction(&mut g, &mut imap, &mut edits, &mut 0.0);

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
) {
    // TODO: Optimize these! This is a super naive implementation, even the paper directly
    // describes a better strategy for doing it.

    let mut applied_any_rule = true;
    while applied_any_rule {
        applied_any_rule = false;

        // Rule 1 (heavy non-edge rule)
        applied_any_rule |= rule1(g);

        // Rule 2 (heavy edge rule, single end)
        applied_any_rule |= rule2(g, imap, edits, k);

        // Rule 3 (heavy edge rule, both ends)
        applied_any_rule |= rule3(g, imap, edits, k);

        if g.present_node_count() <= 1 {
            break;
        }

        // Rule 4
        applied_any_rule |= rule4(g, imap, edits, k);

        // Try Rule 5 only if no other rules could be applied
        if applied_any_rule {
            continue;
        }

        // Rule 5
    }
}

pub fn rule1(g: &mut Graph<Weight>) -> bool {
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

            let sum_neg_internal = c
                .iter()
                .map(|&u| {
                    c.iter()
                        .map(|&v| {
                            if u != v {
                                g.get(u, v).max(Weight::ZERO).abs()
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
                        .map(|v| g.get(u, v).min(Weight::ZERO))
                        .sum::<Weight>()
                })
                .sum::<Weight>();

            if k_c > sum_neg_internal + sum_pos_crossing {
                let mut nodes = c.into_iter();
                let first = nodes.next().unwrap();
                for v in nodes {
                    merge(g, imap, k, edits, first, v);
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
