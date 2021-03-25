use crate::graph::{Graph, GraphWeight};

use std::ops::Not;

/// Effeciently maintains a collection of conflict triples (nodes v-u-w where uv and uw are edges,
/// but vw is a non-edge).
///
/// To avoid logic errors, users of this struct must:
/// - Always supply the same graph to the `update_for_*` functions as used for the `new_for_graph`
///   function.
/// - Call the appropriate update_for_* function for every modification to the graph that has a
///   corresponding function.
#[derive(Clone)]
pub struct ConflictStore {
    // Since the performance of iterating over conflicts is not a big concern, and memory use isn't
    // either, the simplest solution providing good performance for the cases we care about, is
    // just a big collection of flags.
    // TODO: Is it worth making this a bitset for cache reasons? This might also apply to the
    // present vec in Graph actually.
    // `conflicts[idx(v, u, w)]` indicates whether v-u-w is a conflict.
    // This is symmetric in the outer two indices, i.e.
    // `conflicts[idx(v, u, w)] == conflicts[idx(w, u, v)]`, but not in other permutations of the
    // indices.
    conflict_store: Vec<bool>,
    conflict_count: usize,
    // List of edge-disjoint conflicts in the graph.
    edge_disjoint_list: Vec<Option<(usize, usize, usize)>>,
    // Mapping from edge to an index into `edge_disjoint_list`, or `-1`.
    edge_disjoint_mask: Vec<i32>,
    edge_disjoint_count: usize,
    graph_size: usize,
}

/* # Performance goals
 * Should be as fast as possible:
 * - update_for_*, conflict_count
 * - edge_disjoint_conflict_count would be good if we supply it
 *
 * Faster is better, but not that critical:
 * - get_next_conflict
 *
 * Essentially doesn't matter:
 * - new_for_graph
 */

impl ConflictStore {
    pub fn new_for_graph<T: GraphWeight>(g: &Graph<T>) -> Self {
        let size = g.size();
        let mut conflict_store = vec![false; size * size * size];
        let mut count = 0;
        let mut edge_disjoint_mask = vec![-1; size * size];
        let mut edge_disjoint_list = Vec::new();

        // TODO: We could probably avoid a few iterations here by deduplicating

        // Look for conflicts `v-u-w`
        for u in 0..size {
            continue_if_not_present!(g, u);
            for v in 0..size {
                continue_if_not_present!(g, v);
                if u == v {
                    continue;
                }

                let uv = g.has_edge(u, v);

                // For `v-u-w` to be a conflict, uv must exist.
                if !uv {
                    continue;
                }

                for w in 0..size {
                    continue_if_not_present!(g, w);
                    if u == w || v == w {
                        continue;
                    }

                    let uw = g.has_edge(u, w);
                    let vw = g.has_edge(v, w);

                    if uw && !vw {
                        conflict_store[Self::idx_with_size(size, v, u, w)] = true;
                        count += 1;

                        let uv_idx = Self::edge_idx_with_size(size, u, v);
                        let uw_idx = Self::edge_idx_with_size(size, u, w);
                        let vw_idx = Self::edge_idx_with_size(size, v, w);
                        if edge_disjoint_mask[uv_idx] == -1
                            && edge_disjoint_mask[uw_idx] == -1
                            && edge_disjoint_mask[vw_idx] == -1
                        {
                            let conflict_idx = edge_disjoint_list.len() as i32;
                            edge_disjoint_list.push(Some((v.min(w), u, v.max(w))));
                            edge_disjoint_mask[uv_idx] = conflict_idx;
                            edge_disjoint_mask[uw_idx] = conflict_idx;
                            edge_disjoint_mask[vw_idx] = conflict_idx;
                            edge_disjoint_mask[Self::edge_idx_with_size(size, v, u)] = conflict_idx;
                            edge_disjoint_mask[Self::edge_idx_with_size(size, w, u)] = conflict_idx;
                            edge_disjoint_mask[Self::edge_idx_with_size(size, w, v)] = conflict_idx;
                        }
                    }
                }
            }
        }

        debug_assert!(count % 2 == 0, "Conflicts always occur in symmetric pairs");

        count = count / 2;

        let edge_disjoint_count = edge_disjoint_list.len();

        Self {
            conflict_store,
            conflict_count: count,
            edge_disjoint_list,
            edge_disjoint_mask,
            edge_disjoint_count,
            graph_size: size,
        }
    }

    pub fn update_for_insert<T: GraphWeight>(&mut self, g: &Graph<T>, x: usize, y: usize) {
        // When an edge xy is inserted, this resolves any conflicts x-u-y and y-u-x, for any u
        // where this was previously a conflict.
        // Additionally it can create new conflicts x-y-u, u-y-x (if ux does not exist) and
        // y-x-u, u-x-y (if uy does not exist).

        for u in g.nodes() {
            if u == x || u == y {
                continue;
            }

            self.remove_conflict_if_exists(x, u, y);

            let ux = g.has_edge(u, x);
            let uy = g.has_edge(u, y);

            if !ux && uy {
                self.create_conflict(x, y, u);
            } else if ux && !uy {
                self.create_conflict(y, x, u);
            }
        }
    }

    pub fn update_for_delete<T: GraphWeight>(&mut self, g: &Graph<T>, x: usize, y: usize) {
        // When an edge xy is deleted, this resolves any conflict x-y-u, u-y-x, y-x-u, u-x-y, for
        // any u where it was previously a conflict.
        // Additionally it can create new conflicts x-u-y, y-u-x if previously all 3 edges were
        // present.

        for u in g.nodes() {
            if u == x || u == y {
                continue;
            }

            self.remove_conflict_if_exists(x, y, u);
            self.remove_conflict_if_exists(y, x, u);

            let ux = g.has_edge(u, x);
            let uy = g.has_edge(u, y);

            if ux && uy {
                self.create_conflict(x, u, y);
            }
        }
    }

    pub fn update_for_not_present<T: GraphWeight>(&mut self, g: &Graph<T>, x: usize) {
        // This is pretty easy, but also the most expensive of the update functions (O(n^2) instead
        // of O(n)). We're not handling a full merge here, every actual change that a merge
        // produces in the graph should already be handled above. We just need to clear any
        // conflicts involving `x` here.

        for u in g.nodes() {
            if u == x {
                continue;
            }

            for v in (u + 1)..g.size() {
                continue_if_not_present!(g, v);
                if v == x {
                    continue;
                }

                self.remove_conflict_if_exists(u, x, v);
                self.remove_conflict_if_exists(x, u, v);
                self.remove_conflict_if_exists(x, v, u);
            }
        }
    }

    fn create_conflict(&mut self, v: usize, u: usize, w: usize) {
        let vuw_idx = self.idx(v, u, w);
        let wuv_idx = self.idx(w, u, v);

        debug_assert!(self.conflict_store[vuw_idx].not());
        debug_assert!(self.conflict_store[wuv_idx].not());

        self.conflict_count += 1;
        self.conflict_store[vuw_idx] = true;
        self.conflict_store[wuv_idx] = true;

        // If the new conflict can be added to our set of edge-disjoint conflicts, go and do so.
        let uv_idx = self.edge_idx(u, v);
        let uw_idx = self.edge_idx(u, w);
        let vw_idx = self.edge_idx(v, w);
        if self.edge_disjoint_mask[uv_idx] == -1
            && self.edge_disjoint_mask[uw_idx] == -1
            && self.edge_disjoint_mask[vw_idx] == -1
        {
            let conflict_idx = self.edge_disjoint_list.len() as i32;
            self.edge_disjoint_list.push(Some((v.min(w), u, v.max(w))));
            self.set_edge_disjoint_conflict_mask(v, u, w, conflict_idx);
            self.edge_disjoint_count += 1;
        }
    }

    fn remove_conflict_if_exists(&mut self, v: usize, u: usize, w: usize) {
        let vuw_idx = self.idx(v, u, w);
        let wuv_idx = self.idx(w, u, v);

        if self.conflict_store[vuw_idx] {
            debug_assert!(self.conflict_store[vuw_idx]);
            debug_assert!(self.conflict_store[wuv_idx]);

            self.conflict_count -= 1;
            self.conflict_store[vuw_idx] = false;
            self.conflict_store[wuv_idx] = false;

            // If this conflict was part of our set of edge-disjoint conflicts, remove it from
            // that.
            let uv_idx = self.edge_idx(u, v);
            let uw_idx = self.edge_idx(u, w);
            let vw_idx = self.edge_idx(v, w);
            let conflict_idx_uv = self.edge_disjoint_mask[uv_idx];
            let conflict_idx_uw = self.edge_disjoint_mask[uw_idx];
            let conflict_idx_vw = self.edge_disjoint_mask[vw_idx];
            if conflict_idx_uv != -1
                && conflict_idx_uv == conflict_idx_uw
                && conflict_idx_uv == conflict_idx_vw
            {
                let _ = self.edge_disjoint_list[conflict_idx_uv as usize]
                    .take()
                    .unwrap();

                self.set_edge_disjoint_conflict_mask(v, u, w, -1);
                self.edge_disjoint_count -= 1;

                // TODO: Removing a conflict from the set here could enable us to add a new one
                // instead, should definitely try if that's worth it.
            }
        }
    }

    fn set_edge_disjoint_conflict_mask(&mut self, v: usize, u: usize, w: usize, conflict_idx: i32) {
        let uv_idx = self.edge_idx(u, v);
        let vu_idx = self.edge_idx(v, u);
        let uw_idx = self.edge_idx(u, w);
        let wu_idx = self.edge_idx(w, u);
        let vw_idx = self.edge_idx(v, w);
        let wv_idx = self.edge_idx(w, v);
        self.edge_disjoint_mask[uv_idx] = conflict_idx;
        self.edge_disjoint_mask[vu_idx] = conflict_idx;
        self.edge_disjoint_mask[uw_idx] = conflict_idx;
        self.edge_disjoint_mask[wu_idx] = conflict_idx;
        self.edge_disjoint_mask[vw_idx] = conflict_idx;
        self.edge_disjoint_mask[wv_idx] = conflict_idx;
    }

    pub fn conflict_count(&self) -> usize {
        self.conflict_count
    }

    pub fn edge_disjoint_conflict_count(&self) -> usize {
        self.edge_disjoint_count
    }

    pub fn min_cost_to_resolve_edge_disjoint_conflicts<T: GraphWeight>(&self, g: &Graph<T>) -> T {
        let mut sum = T::ZERO;

        for &conflict in &self.edge_disjoint_list {
            if let Some((v, u, w)) = conflict {
                let uv = g.get(u, v);
                let uw = g.get(u, w);
                let vw = -g.get(v, w);

                let min = if uv < uw {
                    if uv < vw {
                        uv
                    } else {
                        vw
                    }
                } else {
                    if uw < vw {
                        uw
                    } else {
                        vw
                    }
                };

                sum += min;
            }
        }

        sum
    }

    pub fn min_cost_to_resolve_edge_disjoint_conflicts_ignoring<T: GraphWeight>(
        &self,
        g: &Graph<T>,
        x: usize,
        y: usize,
    ) -> T {
        let mut sum = T::ZERO;

        for &conflict in &self.edge_disjoint_list {
            if let Some((v, u, w)) = conflict {
                if u == x || v == x || w == x || u == y || v == y || w == y {
                    continue;
                }

                let uv = g.get(u, v);
                let uw = g.get(u, w);
                let vw = -g.get(v, w);

                let min = if uv < uw {
                    if uv < vw {
                        uv
                    } else {
                        vw
                    }
                } else {
                    if uw < vw {
                        uw
                    } else {
                        vw
                    }
                };

                sum += min;
            }
        }

        sum
    }

    pub fn get_next_conflict(&self) -> Option<(usize, usize, usize)> {
        // TODO: This can maybe also try to be clever about which conflict to supply.
        if self.conflict_count == 0 {
            return None;
        }

        for u in 0..self.graph_size {
            for v in 0..self.graph_size {
                for w in 0..self.graph_size {
                    let idx = self.idx(u, v, w);
                    if self.conflict_store[idx] {
                        return Some((u, v, w));
                    }
                }
            }
        }

        panic!(
            "Did not find a conflict, but conflict_count is {}",
            self.conflict_count
        );
    }

    #[inline(always)]
    fn idx(&self, u: usize, v: usize, w: usize) -> usize {
        w + self.graph_size * (v + self.graph_size * u)
    }

    #[inline(always)]
    fn idx_with_size(size: usize, u: usize, v: usize, w: usize) -> usize {
        w + size * (v + size * u)
    }

    #[inline(always)]
    fn edge_idx(&self, u: usize, v: usize) -> usize {
        v + u * self.graph_size
    }
    #[inline(always)]
    fn edge_idx_with_size(size: usize, u: usize, v: usize) -> usize {
        v + u * size
    }
}
