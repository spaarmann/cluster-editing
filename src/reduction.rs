use crate::{
    algo::ProblemInstance, critical_cliques, graph::GraphWeight, util::InfiniteNum, Graph, Weight,
};

use std::collections::BTreeSet;

// This hash set is deterministic, unlike the `std` one. That means the same sequence
// of modifications will result in the same state, including iteration order.
// This isn't critical for correctness here, but it's useful to avoid results randomly
// fluctuating because of internal randomness.
use rustc_hash::FxHashSet;

#[derive(Clone, Debug, Default)]
pub struct ReductionStorage {
    r1: Vec<Weight>,
    r2: Vec<Weight>,
    r3: Vec<Weight>,
    r123_neighbors_u: Vec<(usize, Weight)>,
    r123_neighbors_v: Vec<(usize, Weight)>,
    r123_neighbors_u_new: Vec<(usize, Weight)>,
    r5_relevant_pairs: Vec<(Weight, Weight)>,
    r5_m: Vec<Option<isize>>,
    r5_m_prev: Vec<Option<isize>>,
}

/// The reduction assumes an unweighted graph as input (i.e. one with only weights 1 and -1).
pub fn initial_param_independent_reduction(p: &mut ProblemInstance) -> f32 {
    // Simply merging all critical cliques leads to a graph with at most 4 * k_opt vertices.
    let (g, imap) = critical_cliques::merge_cliques(&p.g, &p.imap, &mut p.path_log);
    p.g = g;
    p.imap = imap;

    let k_start = (p.g.size() / 4) as f32;

    // TODO: It would seem that merging steps above could potentially result in zero-edges in the
    // graph. The algorithm is generally described as requiring that *no* zero-edges are in the
    // input, so doesn't this pose a problem? -> Probably not, actually.
    // Still, for now, this checks if we do ever produce any zero-edges and logs an error if so.
    // A zero-edge here would now actually be a problem, for the edge cut reduction that comes
    // next, see the comments on its definition.
    for u in p.g.nodes() {
        for v in p.g.nodes() {
            if u == v {
                continue;
            }
            if p.g.get(u, v).is_zero() {
                panic!("Produced a zero-edge during parameter-independent reduction!");
            }
        }
    }

    p.k = k_start;
    edge_cut_reduction(p);

    full_param_independent_reduction(p, true);

    p.k.max(0.0)
}

pub fn param_dependent_reduction(p: &mut ProblemInstance) {
    let _k_start = p.k;

    if p.k <= Weight::ZERO {
        return;
    }

    dbg_trace_indent!(
        p,
        _k_start,
        "Starting param-dep-reduction, k {}, edits {:?}.",
        p.k,
        p.edits
    );

    induced_cost_reduction(p);

    p.params.stats.borrow_mut().k_red_from_ind_cost += _k_start - p.k;

    dbg_trace_indent!(
        p,
        _k_start,
        "Finished param-dep-reduction, {}, edits {:?}.",
        p.k,
        p.edits
    );
}

/// Rules from "Böcker et al., Exact Algorithms for Cluster Editing: Evaluation and
/// Experiments", 2011
/// Can be applied during the search as well, can modify `k` appropriately and don't require any
/// specific form of the input.
pub fn full_param_independent_reduction(p: &mut ProblemInstance, is_initial: bool) {
    let _k_start = p.k;
    dbg_trace_indent!(
        p,
        _k_start,
        "Starting full gen-param-indep-reduction, k {}, edits {:?}.",
        p.k,
        p.edits
    );

    let mut applied_any_rule = true;
    while applied_any_rule && p.k > 0.0 && p.g.present_node_count() > 1 {
        applied_any_rule = false;

        let k_before = p.k;
        // Don't repeatedly apply rules123 by themselves, one call will do all possible
        // applications.
        rules123(p);

        if !is_initial {
            p.params.stats.borrow_mut().k_red_from_rules123 += k_before - p.k;
        }

        let k_before = p.k;
        // Rule 4
        while rule4(p) {
            applied_any_rule = true;
        }

        if !is_initial {
            p.params.stats.borrow_mut().k_red_from_rule4 += k_before - p.k;
        }

        if applied_any_rule {
            continue;
        }

        let k_before = p.k;
        // Rule 5
        // TODO: Try looping this?
        applied_any_rule = rule5(p);

        if !is_initial {
            p.params.stats.borrow_mut().k_red_from_rule5 += k_before - p.k;
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
    let _k_start = p.k;
    dbg_trace_indent!(
        p,
        _k_start,
        "Starting fast gen-param-indep-reduction, k {}, edits {:?}.",
        p.k,
        p.edits
    );

    rules123(p);
    p.params.stats.borrow_mut().k_red_from_rule4 += _k_start - p.k;

    dbg_trace_indent!(
        p,
        _k_start,
        "Finished fast gen-param-indep-reduction, {}, edits {:?}.",
        p.k,
        p.edits
    );
}

pub fn rules123(p: &mut ProblemInstance) -> bool {
    let mut r = p.r.take().unwrap();

    r.r1.clear();
    r.r1.extend(std::iter::repeat(Weight::ZERO).take(p.g.size() * p.g.size()));
    r.r2.clear();
    r.r2.extend(std::iter::repeat(Weight::ZERO).take(p.g.size() * p.g.size()));
    r.r3.clear();
    r.r3.extend(std::iter::repeat(Weight::ZERO).take(p.g.size() * (p.g.size() - 1) / 2));

    fn r3idx(u: usize, v: usize) -> usize {
        let row_offset = v * (v - 1) / 2;
        row_offset + u
    }

    let mut edges_to_forbid = BTreeSet::new();
    let mut edges_to_merge2 = BTreeSet::new();
    let mut edges_to_merge3 = BTreeSet::new();

    for u in 0..p.g.size() {
        continue_if_not_present!(p.g, u);

        let r1sum =
            p.g.neighbors_with_weights(u)
                .map(|(_, weight)| weight)
                .sum::<Weight>();
        let r2total_sum =
            p.g.nodes()
                .filter_map(|w| {
                    if u == w {
                        None
                    } else {
                        Some(p.g.get(u, w).abs())
                    }
                })
                .sum::<Weight>();

        let r3sum_u =
            p.g.neighbors_with_weights(u)
                .map(|(_, w)| w)
                .sum::<Weight>();

        for v in 0..p.g.size() {
            if u == v {
                continue;
            }
            continue_if_not_present!(p.g, v);

            let uv = p.g.get(u, v);

            if v > u {
                // Rule 3 is symmetric, so only do it for one orderinp.g.

                let r3sum = r3sum_u - uv
                    + p.g
                        .neighbors_with_weights(v)
                        .map(|(_, w)| w)
                        .sum::<Weight>()
                    - uv;
                let r3val = uv - r3sum;
                r.r3[r3idx(u, v)] = r3val;
                if uv > Weight::ZERO && r3val >= Weight::ZERO {
                    edges_to_merge3.insert((u, v));
                }
            }
            // The other two aren't symmetric, so do them for all ordered pairs.

            if uv == Weight::NEG_INFINITY {
                r.r1[u * p.g.size() + v] = Weight::NEG_INFINITY;
            } else {
                let r1val = -uv - r1sum;
                r.r1[u * p.g.size() + v] = r1val;
                if uv < Weight::ZERO && r1val >= Weight::ZERO && uv.is_finite() {
                    edges_to_forbid.insert((u, v));
                }
            }

            let r2val = if r2total_sum == Weight::INFINITY {
                Weight::NEG_INFINITY
            } else {
                uv - (r2total_sum - uv.abs())
            };
            r.r2[u * p.g.size() + v] = r2val;

            if uv > Weight::ZERO && r2val >= Weight::ZERO {
                edges_to_merge2.insert((u, v));
            }
        }
    }

    let mut any_applied = false;
    // Repeatedly apply one reduction, and update the relevant fields, until no further application
    // is possible.
    // TODO: Play around with order here. (always do all forbids first? always do all merges first?
    // alternate?)
    let mut applied = true;
    while applied && p.k >= 0.0 {
        applied = false;

        // First, a couple little helpers for actually performing the updates once they've been computed.
        fn r1_update(
            p: &ProblemInstance,
            r: &mut ReductionStorage,
            u: usize,
            v: usize,
            diff: Weight,
            edge_set: &mut BTreeSet<(usize, usize)>,
        ) {
            let idx = u * p.g.size() + v;
            let prev = r.r1[idx];
            let new = prev + diff;
            r.r1[idx] = new;

            debug_assert!(
                new != Weight::INFINITY && diff != Weight::INFINITY,
                "Calculated new for {}-{} as {}, with diff {} and prev {}",
                u,
                v,
                new,
                diff,
                prev
            );

            let uv = p.g.get(u, v);

            if new >= Weight::ZERO && prev < Weight::ZERO && uv < Weight::ZERO {
                edge_set.insert((u, v));
            } else if (new < Weight::ZERO || uv >= Weight::ZERO) && prev >= Weight::ZERO {
                edge_set.remove(&(u, v));
            }
        }
        fn r2_update(
            p: &ProblemInstance,
            r: &mut ReductionStorage,
            u: usize,
            v: usize,
            diff: Weight,
            edge_set: &mut BTreeSet<(usize, usize)>,
        ) {
            let idx = u * p.g.size() + v;
            let prev = r.r2[idx];
            let new = prev + diff;
            r.r2[idx] = new;

            debug_assert!(
                new != Weight::INFINITY && diff != Weight::INFINITY && !new.is_nan(),
                "Calculated new for {}-{} as {}, with diff {} and prev {}",
                u,
                v,
                new,
                diff,
                prev
            );

            let uv = p.g.get(u, v);
            if new >= Weight::ZERO && prev < Weight::ZERO && uv > Weight::ZERO {
                edge_set.insert((u, v));
            } else if (new < Weight::ZERO || uv <= Weight::ZERO) && prev >= Weight::ZERO {
                edge_set.remove(&(u, v));
            }
        }
        fn r3_update(
            p: &ProblemInstance,
            r: &mut ReductionStorage,
            u: usize,
            v: usize,
            diff: Weight,
            edge_set: &mut BTreeSet<(usize, usize)>,
        ) {
            let idx = r3idx(u.min(v), u.max(v));
            let prev = r.r3[idx];

            let new = prev + diff;
            r.r3[idx] = new;

            debug_assert!(
                new != Weight::INFINITY && diff != Weight::INFINITY,
                "Calculated new for {}-{} as {}, with diff {} and prev {}",
                u,
                v,
                new,
                diff,
                prev
            );

            let uv = p.g.get(u, v);
            if new >= Weight::ZERO && prev < Weight::ZERO {
                edge_set.insert((u.min(v), u.max(v)));
            } else if (new < Weight::ZERO || uv <= Weight::ZERO) && prev >= Weight::ZERO {
                edge_set.remove(&(u.min(v), u.max(v)));
            }
        }

        if let Some(&(u, v)) = edges_to_forbid.iter().next() {
            edges_to_forbid.remove(&(u, v));

            trace_and_path_log!(p, p.k, "rule1, forbidding {:?}-{:?}", p.imap[u], p.imap[v]);

            debug_assert!(r.r1[u * p.g.size() + v] >= Weight::ZERO);
            debug_assert!(p.g.get(u, v) < Weight::ZERO);
            debug_assert!(p.g.get(u, v).abs() > 0.001);

            p.set(u, v, InfiniteNum::NEG_INFINITY);
            any_applied = true;
            applied = true;

            // Update other entries:
            // No effect on other r1 values, except:
            // Since r1 is asymmetric, u-v and v-u might both be in the list. Remove v-u if it is.
            edges_to_forbid.remove(&(v, u));

            // r2(u-v) and r2(v-u) both now neg-inf.
            r.r2[u * p.g.size() + v] = Weight::NEG_INFINITY;
            edges_to_merge2.remove(&(u, v));
            r.r2[v * p.g.size() + u] = Weight::NEG_INFINITY;
            edges_to_merge2.remove(&(v, u));
            // In addition, for all x in V: r2(u-x) and r2(v-x) now neg-inf.
            for x in p.g.nodes() {
                r.r2[u * p.g.size() + x] = Weight::NEG_INFINITY;
                edges_to_merge2.remove(&(u, x));
                r.r2[v * p.g.size() + x] = Weight::NEG_INFINITY;
                edges_to_merge2.remove(&(v, x));

                if x == u || x == v {
                    continue;
                }
                // For r3, the sums only contain existing edges, but we only forbid
                // already-not-existing edges, so they are not affected. However, for all x, r3(u-x)
                // and r3(v-x) also include s(ux) and s(vx) directly, so those change.
                r3_update(p, &mut r, u, x, Weight::NEG_INFINITY, &mut edges_to_merge3);
                r3_update(p, &mut r, v, x, Weight::NEG_INFINITY, &mut edges_to_merge3);
            }
        }

        // This was the easy part, now comes the update for merging.

        if let Some(&(u, v)) = edges_to_merge2.iter().chain(edges_to_merge3.iter()).next() {
            edges_to_merge2.remove(&(u, v));
            edges_to_merge3.remove(&(u.min(v), u.max(v)));

            trace_and_path_log!(p, p.k, "rule2/3 merge {:?}-{:?}", p.imap[u], p.imap[v]);

            debug_assert!(
                r.r2[u * p.g.size() + v] >= Weight::ZERO
                    || r.r3[r3idx(u.min(v), u.max(v))] >= Weight::ZERO
            );
            debug_assert!(p.g.get(u, v) > Weight::ZERO);

            // We'll need the previous neighbors of both u and v for the update.
            let u_neighbors = p.g.neighbors_with_weights(u).collect::<Vec<_>>();
            let v_neighbors = p.g.neighbors_with_weights(v).collect::<Vec<_>>();

            let uv = p.g.get(u, v);

            // For r2(x-y), the computation involves every single pair x-w, for all w != x.
            // Thus merging u-v will have an impact on literally all r2 values :(
            // (2.1) First, for every r2(x-y), the pairing x-v disappears from the sum.
            //
            // (2.2) For r2(u-y), for all y, the first term may have changed. Add (uy_new - uy_old)
            // to the value.
            // (2.3) For r2(y-u), for all y, the first term may have changed. Add (uy_new - uy_old)
            // to the value.
            // (2.4) For r2(x-y), x != u && y != u, the sum may have changed. Add (xu_new.abs() - xu_old.abs())
            // to the sum (subtract it from the value).
            //
            // Unlike r1 (see below) none of these operations require knowledge of both the the
            // result of the merge and previous values simultaneously, so we do the first half now,
            // and the second half after the merge.
            //
            // And then r3. Note this is symmetric, unlike the others.
            // We again split handling of this, like for r2, to avoid having to store neighbor sets
            // for all nodes.
            // For r3(u-x), for all x, the first term will have changed. Also, v disappeared from
            // u's neighbors, so the first sum changed. If v was a neighbor of x too, the second
            // sum changed too.
            // =>
            // (3.1) Before merging, for r3(u-x) for all x != v:
            //  r3 -= x-u ; r3 += u-v ; if x-v > 0: r3 -= x-v
            // (3.2) Then, after merging:
            //  r3 += x-u
            // For r3(x-y), x != u && y != u, the first term is unchanged. The sums changed if the
            // respective node (x or y) has u or v as a neighbor.
            // =>
            // (3.3) Before merging, for r3(x-y), x != u && x != v && y != u && y != v:
            //  if x-u > 0: r3 += x-u ; if y-u > 0: r3 += y-u ; if x-v > 0: r3 += x-v; if y-v > 0: r3 += y-v
            // (3.4) Then, after merging:
            //  if x-u > 0: r3 -= x-u ; if y-u > 0: r3 -= y-u
            for x in p.g.nodes() {
                if x == v {
                    continue;
                }

                let vx = p.g.get(v, x);
                let vx_abs = vx.abs();

                for y in p.g.nodes() {
                    if y == x || y == v {
                        continue;
                    }

                    // 2.1
                    if vx_abs.is_finite() {
                        // If x-v was -Infinity, we can't just add Infinity back in to counteract
                        // its effect on the sum obviously.
                        // To get an accurate r2 for this moment in time, we'd really have to
                        // recalculate it entirely, but we can cheat a little:
                        // If a node had a -Inf weight to v, after the merge it must have a -Inf
                        // weight to u, so the sum keeps being Infinity anyway, no matter what else
                        // we may try doing to it. We thus simply skip the update.
                        r2_update(p, &mut r, x, y, vx_abs, &mut edges_to_merge2);
                    }

                    if y == u || x == u {
                        continue;
                    }

                    // 2.4 (add xu_old.abs())
                    let ux = p.g.get(x, u);
                    let ux_abs = ux.abs();
                    if ux_abs.is_finite() {
                        // Essentially the same justification as above holds here too.
                        r2_update(p, &mut r, x, y, ux_abs, &mut edges_to_merge2);
                    }

                    // r3 only needs the pair in one direction.
                    if y > x {
                        // (3.3)
                        let uy = p.g.get(u, y);
                        let vy = p.g.get(v, y);

                        let mut diff = Weight::ZERO;
                        diff += ux.max(Weight::ZERO);
                        diff += uy.max(Weight::ZERO);
                        diff += vx.max(Weight::ZERO);
                        diff += vy.max(Weight::ZERO);
                        r3_update(p, &mut r, x, y, diff, &mut edges_to_merge3);
                    }
                }

                if x == u {
                    continue;
                }

                let ux = p.g.get(x, u);
                if ux.is_finite() {
                    // Pretty much the same infinity-related justification as above once more.

                    // 2.2 (u-y is named u-x here), subtract uy_old.
                    r2_update(p, &mut r, u, x, -ux, &mut edges_to_merge2);
                    // 2.3. (y-u is named x-u here), subtract uy_old.
                    r2_update(p, &mut r, x, u, -ux, &mut edges_to_merge2);
                }

                // (3.1)
                let mut diff = Weight::ZERO;
                if ux.is_finite() {
                    // Same infinity-related justification for ignoring this if ux is -Inf as above
                    // for r2.
                    diff -= ux;
                }
                diff += uv;
                if vx > Weight::ZERO {
                    diff -= vx;
                }

                r3_update(p, &mut r, u, x, diff, &mut edges_to_merge3);
            }

            p.merge(u, v);
            any_applied = true;
            applied = true;

            // Since v is now gone, all entries involving it are void.
            edges_to_forbid.retain(|&(x, y)| x != v && y != v);
            edges_to_merge2.retain(|&(x, y)| x != v && y != v);
            edges_to_merge3.retain(|&(x, y)| x != v && y != v);

            let u_new_neighbors = p.g.neighbors_with_weights(u).collect::<Vec<_>>();

            // For all x that were neighbors of u or v, or are *now* a neighbor of u, r1 may have
            // changed to any other node.
            // For r1(x-y), if x is or was neighbor of u or v, the sum will have changed:

            // Everything that was a neighbor of v now isn't anymore, so the previous weight to
            // v disappears from the sum.
            for &(x, xv) in v_neighbors.iter() {
                for y in p.g.nodes() {
                    if x == y {
                        continue;
                    }
                    r1_update(p, &mut r, x, y, xv, &mut edges_to_forbid);
                }
            }

            // TODO: Is it worth making these sets instead for the set intersection/difference?
            // Actually, perhaps even better, this could probably be split in pre-/post-merging,
            // like the other two?
            for &(x, xu_old) in u_neighbors.iter() {
                if x == v {
                    continue;
                }
                if let Some(&(_, xu_new)) = u_new_neighbors.iter().find(|&&(t, _)| t == x) {
                    // For x in old_n ∩ new_n: u was and still is part of the sum, add (new_weight -
                    // old_weight) to the sum (subtract it from the value).
                    for y in p.g.nodes() {
                        if x == y {
                            continue;
                        }
                        r1_update(p, &mut r, x, y, -(xu_new - xu_old), &mut edges_to_forbid);
                    }
                } else {
                    // For x in old_n - new_n: u dropped out of the sum entirely, subtract the old weight
                    // from the sum (add it to the value).
                    for y in p.g.nodes() {
                        if x == y {
                            continue;
                        }
                        r1_update(p, &mut r, x, y, xu_old, &mut edges_to_forbid);
                    }
                }
            }
            for &(x, xu_new) in u_new_neighbors.iter() {
                if let None = u_neighbors.iter().find(|&&(t, _)| t == x) {
                    // For x in new_n - old_n: u entered the sum as a new vertex, add the new weight to the
                    // sum (subtract it from the value).
                    for y in p.g.nodes() {
                        if x == y {
                            continue;
                        }
                        r1_update(p, &mut r, x, y, -xu_new, &mut edges_to_forbid);
                    }
                }
            }

            // Second half of the r2 and r3 updates.
            for x in p.g.nodes() {
                if x == u {
                    continue;
                }

                let ux = p.g.get(x, u);
                let ux_abs = ux.abs();

                // 2.2 (u-y is named u-x here), add uy_new.
                r2_update(p, &mut r, u, x, ux, &mut edges_to_merge2);
                // 2.3. (y-u is named x-u here), add uy_new.
                r2_update(p, &mut r, x, u, ux, &mut edges_to_merge2);

                // (3.2)
                let ux = p.g.get(u, x);
                r3_update(p, &mut r, u, x, ux, &mut edges_to_merge3);

                for y in p.g.nodes() {
                    if y == x || y == u {
                        continue;
                    }
                    // 2.4, subtract xu_new.abs()
                    r2_update(p, &mut r, x, y, -ux_abs, &mut edges_to_merge2);

                    if y > x {
                        // (3.3)
                        let uy = p.g.get(u, y);

                        let mut diff = Weight::ZERO;
                        diff -= ux.max(Weight::ZERO);
                        diff -= uy.max(Weight::ZERO);
                        r3_update(p, &mut r, x, y, diff, &mut edges_to_merge3);
                    }
                }
            }
        }
    }

    p.r = Some(r);
    any_applied
}

pub fn rule4(p: &mut ProblemInstance) -> bool {
    let g = &mut p.g;
    let mut applied = false;
    // TODO: Think through if this doesn't do weird things if we already set some non-edges to
    // NEG_INFINITY
    // TODO: Comment this stuff (including MinCut) and probably clean it up a bit ^^'
    let mut c = FxHashSet::default();

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

            let sum_neg_internal = c
                .iter()
                .map(|&u| {
                    c.iter()
                        .map(|&v| {
                            // Don't count both uv and vu
                            if u > v {
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
                        trace_and_path_log!(
                            p,
                            p.k,
                            "rule4 merge. first {:?} - v {:?}",
                            p.imap[first],
                            p.imap[v]
                        );

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
    struct R5Data<'a> {
        delta_u: Weight,
        delta_v: Weight,
        max_x: Weight,
        relative_difference: Weight,
        relevant_pairs: &'a mut Vec<(Weight, Weight)>,
    }

    fn compute_initial_data<'a>(
        g: &Graph<Weight>,
        relevant_pairs: &'a mut Vec<(Weight, Weight)>,
        u: usize,
        v: usize,
    ) -> R5Data<'a> {
        let mut delta_u = Weight::ZERO;
        let mut delta_v = Weight::ZERO;
        let mut max_x = Weight::ZERO;
        let mut relative_difference = Weight::ZERO;

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
            } else if x_edge {
                delta_u += x;
                delta_v -= y;
            } else {
                delta_u -= x;
                delta_v += y;
            }
        }

        R5Data {
            delta_u,
            delta_v,
            max_x,
            relative_difference,
            relevant_pairs,
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn compute_max(
        d: &R5Data,
        m: &mut Vec<Option<isize>>,
        m_prev: &mut Vec<Option<isize>>,
        upper_bound: Weight,
    ) -> Weight {
        let m_size = (2.0 * d.max_x + 1.0).ceil() as usize;

        //let mut m: Vec<Option<isize>> = vec![None; m_size];
        m.extend(std::iter::repeat(None).take(m_size));

        let mut m_inf: Option<isize> = None; // Extra entry "in" m for infinity values

        // [-max_x, ..., -1, 0, 1, ..., max_x]
        let max_x = d.max_x as isize;
        let get_idx = |x: isize| (x + max_x) as usize;
        let get_val = |i: usize| (i as isize - max_x);

        // Initialize array
        m[get_idx(0)] = Some(0);

        for &(x_j, y_j) in d.relevant_pairs.iter() {
            let mut max_value = 0;

            let x = x_j as isize;
            let y = y_j as isize;

            //let m_prev = m.clone();
            m_prev.clear();
            m_prev.extend_from_slice(m);

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

    let mut r = p.r.take().unwrap();

    for u in 0..p.g.size() {
        continue_if_not_present!(p.g, u);

        for v in (u + 1)..p.g.size() {
            continue_if_not_present!(p.g, v);

            let uv = p.g.get(u, v);
            if uv <= Weight::ZERO {
                continue;
            }

            r.r5_relevant_pairs.clear();
            let d = compute_initial_data(&p.g, &mut r.r5_relevant_pairs, u, v);

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
                append_path_log!(p, "rule5, early merge {:?}-{:?}\n", p.imap[u], p.imap[v]);
                p.merge(u, v);
                applied = true;
                continue;
            }

            r.r5_m.clear();
            r.r5_m_prev.clear();
            let max = compute_max(&d, &mut r.r5_m, &mut r.r5_m_prev, uv);

            if uv > max {
                dbg_trace_indent!(
                    p,
                    p.k,
                    "merge after dynprog: {:?}-{:?}, uv {}, rel_diff {}, du {}, dv {}, max {}, edits so far: {:?}",
                    p.imap[u],
                    p.imap[v],
                    uv,
                    d.relative_difference,
                    d.delta_u,
                    d.delta_v,
                    max,
                    p.edits
                );
                append_path_log!(p, "rule5, merge {:?}-{:?}\n", p.imap[u], p.imap[v]);
                p.merge(u, v);
                applied = true;
                continue;
            }

            // TODO: Paper claims it might be good to use the two bounds above to decide in which
            // order the below computation should be executed for the edges, but the PEACE impl
            // doesn't seem to do anything of the sort either, afaict.
        }
    }

    p.r = Some(r);
    applied
}

fn min_cut(g: &Graph<Weight>, c: &FxHashSet<usize>, a: usize) -> Weight {
    let mut g = g.clone();
    let mut c = c.clone();

    fn merge_mc(g: &mut Graph<Weight>, c: &mut FxHashSet<usize>, u: usize, v: usize) {
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

    fn min_cut_phase(g: &mut Graph<Weight>, c: &mut FxHashSet<usize>, a: usize) -> Weight {
        let mut set = FxHashSet::default();
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

fn induced_cost_reduction(p: &mut ProblemInstance) {
    let mut u = 0;
    'outer: while u < p.g.size() {
        if !p.g.is_present(u) {
            u += 1;
            continue;
        }

        for v in (u + 1)..p.g.size() {
            continue_if_not_present!(p.g, v);

            let uv = p.g.get(u, v);

            if uv.is_infinite() {
                // If the value is already forbidden, no point in trying anything here.
                continue;
            }

            let induced_costs = p.induced_costs.get_costs(u, v);

            // TODO: Could try skipping the bound calculation if icp/icf with the weight of uv
            // itself is *already* larger k.
            let bound = p
                .conflicts
                .min_cost_to_resolve_edge_disjoint_conflicts_ignoring(&p.g, u, v);

            if induced_costs.icp + bound > p.k {
                trace_and_path_log!(
                    p,
                    p.k,
                    "icp, forbid {:?}-{:?} ({}-{}), with icp {}, uv {}, bound {}, and k {}",
                    p.imap[u],
                    p.imap[v],
                    u,
                    v,
                    induced_costs.icp,
                    uv,
                    bound,
                    p.k
                );

                if uv > Weight::ZERO {
                    p.k -= uv;
                    p.make_delete_edit(u, v);

                    p.params.stats.borrow_mut().k_red_from_ind_cost += uv;
                }
                if uv.abs() < 0.001 {
                    p.k -= 0.5;

                    p.params.stats.borrow_mut().k_red_from_zeroes += 0.5;
                }

                p.set(u, v, Weight::NEG_INFINITY);

                if p.k <= 0.0 {
                    return;
                }

                // We changed the graph, so we may have invalidated u_neighbors.
                // We could just continue 'outer, but we'd possibly leave reduction
                // on the table if there is another v where u-v is applicable.
                // This way we recalculate u_neighbors, but still go through all v's
                // again.
                // TODO: Investigate if this is worth it for performance, or a pure continue
                // ends up being better.
                continue 'outer;
            }
            if induced_costs.icf + bound > p.k {
                trace_and_path_log!(
                    p,
                    p.k,
                    "icf, merge {:?}-{:?} ({}-{}), with icf {}, uv {}, bound {} and k {}",
                    p.imap[u],
                    p.imap[v],
                    u,
                    v,
                    induced_costs.icf,
                    uv,
                    bound,
                    p.k
                );

                let k_before = p.k;
                p.merge(u, v);

                p.params.stats.borrow_mut().k_red_from_ind_cost += k_before - p.k;

                if p.k <= 0.0 {
                    return;
                }

                // See above.
                continue 'outer;
            }
        }

        u += 1;
    }
}

// From "Cluster Editing: Kernelization Based on Edge Cuts", Cao and Chen, 2012.
// As described in the paper, this relies on the absence of zero-edges. As a starting point, we
// simply only apply it once during the initial reduction step, when the graph cannot contain any
// zero-edges yet.
// As possible future expansions, it might be possible to selectively apply the reduction for
// intermediate graphs or subgraphs where we have checked no zero-edges are present, or maybe even
// adjust the reduction to cope with zero-edges.
fn edge_cut_reduction(p: &mut ProblemInstance) {
    let mut potential_inserts = Vec::new();

    for v in 0..p.g.size() {
        continue_if_not_present!(p.g, v);

        let mut deficiency = Weight::ZERO;
        let mut gamma = Weight::ZERO;
        let mut closed_neighbor_count = 0;

        potential_inserts.clear();

        for x in p.g.closed_neighbors(v) {
            closed_neighbor_count += 1;

            for y in p.g.closed_neighbors(v) {
                if y >= x {
                    continue;
                }

                let xy = p.g.get(x, y);
                if xy <= Weight::ZERO {
                    deficiency += -xy;
                    potential_inserts.push((x, y, xy));
                }
            }

            for y in p.g.nodes() {
                if x == y || v == y || p.g.get(v, y) > Weight::ZERO {
                    continue;
                }

                let xy = p.g.get(x, y);
                if xy > Weight::ZERO {
                    gamma += xy;
                }
            }
        }

        let stable_cost = 2.0 * deficiency + gamma;
        let reducible = stable_cost < closed_neighbor_count as Weight;

        if reducible {
            // We can make N[v] a clique!

            // Rule 1: Insert missing edges:
            for &(x, y, xy) in &potential_inserts {
                p.g.set(x, y, -xy);
                p.make_insert_edit(x, y);
                trace_and_path_log!(
                    p,
                    p.k,
                    "edge cuts, insert {:?}-{:?} ({}-{})",
                    p.imap[x],
                    p.imap[y],
                    x,
                    y
                );
            }

            // Rule 2: Delete some edges around the clique:
            for x in 0..p.g.size() {
                if !p.g.is_present(x) || (x != v && p.g.get(v, x) <= Weight::ZERO) {
                    continue;
                }

                for y in 0..p.g.size() {
                    if !p.g.is_present(y) || x == y || v == y || p.g.get(x, y) <= Weight::ZERO {
                        continue;
                    }

                    let vy = p.g.get(v, y);
                    if vy > Weight::ZERO {
                        continue;
                    }

                    // y is in N(N[v])

                    let mut total_weight = Weight::ZERO;
                    for z in p.g.closed_neighbors(v) {
                        total_weight += p.g.get(y, z);
                    }

                    if total_weight <= Weight::ZERO {
                        // We can delete all the edges from y to N[v]
                        for z in 0..p.g.size() {
                            if !p.g.is_present(z) || (z != v && p.g.get(v, z) <= Weight::ZERO) {
                                continue;
                            }

                            let yz = p.g.get(y, z);
                            if yz > Weight::ZERO {
                                p.g.set(y, z, -yz);
                                p.make_delete_edit(y, z);
                                trace_and_path_log!(
                                    p,
                                    p.k,
                                    "edge cuts, delete {:?}-{:?} ({}-{})",
                                    p.imap[y],
                                    p.imap[z],
                                    y,
                                    z
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
