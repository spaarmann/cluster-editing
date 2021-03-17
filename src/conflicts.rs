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
                    }
                }
            }
        }

        debug_assert!(count % 2 == 0, "Conflicts always occur in symmetric pairs");

        count = count / 2;

        Self {
            conflict_store,
            conflict_count: count,
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

            let xuy_idx = self.idx(x, u, y);
            let yux_idx = self.idx(y, u, x);
            if self.conflict_store[xuy_idx] {
                debug_assert!(self.conflict_store[xuy_idx]);
                debug_assert!(self.conflict_store[yux_idx]);

                self.conflict_count -= 1;
                self.conflict_store[xuy_idx] = false;
                self.conflict_store[yux_idx] = false;
            }

            let ux = g.has_edge(u, x);
            let uy = g.has_edge(u, y);

            if !ux && uy {
                let xyu_idx = self.idx(x, y, u);
                let uyx_idx = self.idx(u, y, x);

                debug_assert!(self.conflict_store[xyu_idx].not());
                debug_assert!(self.conflict_store[uyx_idx].not());

                self.conflict_count += 1;
                self.conflict_store[xyu_idx] = true;
                self.conflict_store[uyx_idx] = true;
            } else if ux && !uy {
                let yxu_idx = self.idx(y, x, u);
                let uxy_idx = self.idx(u, x, y);

                debug_assert!(self.conflict_store[yxu_idx].not());
                debug_assert!(self.conflict_store[uxy_idx].not());

                self.conflict_count += 1;
                self.conflict_store[yxu_idx] = true;
                self.conflict_store[uxy_idx] = true;
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

            let xyu_idx = self.idx(x, y, u);
            let uyx_idx = self.idx(u, y, x);
            let yxu_idx = self.idx(y, x, u);
            let uxy_idx = self.idx(u, x, y);
            if self.conflict_store[xyu_idx] {
                debug_assert!(self.conflict_store[xyu_idx]);
                debug_assert!(self.conflict_store[uyx_idx]);

                self.conflict_count -= 1;
                self.conflict_store[xyu_idx] = false;
                self.conflict_store[uyx_idx] = false;
            }
            if self.conflict_store[yxu_idx] {
                debug_assert!(self.conflict_store[yxu_idx]);
                debug_assert!(self.conflict_store[uxy_idx]);

                self.conflict_count -= 1;
                self.conflict_store[yxu_idx] = false;
                self.conflict_store[uxy_idx] = false;
            }

            let ux = g.has_edge(u, x);
            let uy = g.has_edge(u, y);

            if ux && uy {
                let xuy_idx = self.idx(x, u, y);
                let yux_idx = self.idx(y, u, x);

                debug_assert!(self.conflict_store[xuy_idx].not());
                debug_assert!(self.conflict_store[yux_idx].not());

                self.conflict_count += 1;
                self.conflict_store[xuy_idx] = true;
                self.conflict_store[yux_idx] = true;
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

                let uxv_idx = self.idx(u, x, v);
                if self.conflict_store[uxv_idx] {
                    let vxu_idx = self.idx(v, x, u);

                    debug_assert!(self.conflict_store[uxv_idx]);
                    debug_assert!(self.conflict_store[vxu_idx]);

                    self.conflict_count -= 1;
                    self.conflict_store[uxv_idx] = false;
                    self.conflict_store[vxu_idx] = false;
                }

                let xuv_idx = self.idx(x, u, v);
                if self.conflict_store[xuv_idx] {
                    let vux_idx = self.idx(v, u, x);

                    debug_assert!(self.conflict_store[xuv_idx]);
                    debug_assert!(self.conflict_store[vux_idx]);

                    self.conflict_count -= 1;
                    self.conflict_store[xuv_idx] = false;
                    self.conflict_store[vux_idx] = false;
                }

                let xvu_idx = self.idx(x, v, u);
                if self.conflict_store[xvu_idx] {
                    let uvx_idx = self.idx(u, v, x);

                    debug_assert!(self.conflict_store[xvu_idx]);
                    debug_assert!(self.conflict_store[uvx_idx]);

                    self.conflict_count -= 1;
                    self.conflict_store[xvu_idx] = false;
                    self.conflict_store[uvx_idx] = false;
                }
            }
        }
    }

    pub fn conflict_count(&self) -> usize {
        self.conflict_count
    }

    // This would be nice for some things, but maybe isn't super critical to supply.
    pub fn edge_disjoint_conflict_count(&self) -> usize {
        todo!();
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
}
