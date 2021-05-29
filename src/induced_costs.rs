use crate::{
    graph::{Graph, GraphWeight},
    Weight,
};

use std::cmp::Ordering;

#[derive(Copy, Clone, Debug)]
pub struct Costs {
    pub icf: Weight,
    pub icp: Weight,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct EdgeWithBranchingNumber {
    u: usize,
    v: usize,
    branching_num: Weight,
}

// For this and Ord: We know we never get NaNs, and don't care much about any other floating point
// weirdness, so this should be fine.
impl Eq for EdgeWithBranchingNumber {}

impl PartialOrd for EdgeWithBranchingNumber {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(
            self.branching_num
                .partial_cmp(&other.branching_num)
                .unwrap()
                .then_with(|| self.u.cmp(&other.u))
                .then_with(|| self.v.cmp(&other.v)),
        )
    }
}

impl Ord for EdgeWithBranchingNumber {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Clone)]
pub struct InducedCosts {
    cost_store: Vec<Costs>,
    graph_size: usize,
    min_branching_num: Option<EdgeWithBranchingNumber>,
    oplog: Vec<Op>,
}

#[derive(Clone)]
enum Op {
    UpdateCosts {
        u: usize,
        v: usize,
        prev_costs: Costs,
    },
    UpdateMinBranchingNum(Option<EdgeWithBranchingNumber>),
}

impl InducedCosts {
    pub fn new_for_graph(g: &Graph<Weight>) -> Self {
        let size = g.size();
        let cost_store = vec![
            Costs {
                icf: Weight::ZERO,
                icp: Weight::ZERO
            };
            size * size
        ];

        let mut this = Self {
            graph_size: size,
            cost_store,
            min_branching_num: None,
            oplog: Vec::new(),
        };

        this.calculate_all_costs(g);

        this
    }

    fn calculate_all_costs(&mut self, g: &Graph<Weight>) {
        let mut u_neighbors = Vec::new();

        //self.branching_nums.clear();
        self.min_branching_num = None;

        let mut u = 0;
        while u < g.size() {
            if !g.is_present(u) {
                u += 1;
                continue;
            }

            u_neighbors.clear();
            u_neighbors.extend(g.neighbors_with_weights(u));

            for v in (u + 1)..g.size() {
                continue_if_not_present!(g, v);

                let uv = g.get(u, v);

                let mut icf = Weight::ZERO;
                let mut icp = Weight::ZERO;

                if uv.is_infinite() {
                    icf = Weight::ZERO;
                    // If it's forbidden, the costs of setting it permanent would be infinite.
                    icp = Weight::INFINITY;
                } else {
                    for &(w, uw) in u_neighbors.iter() {
                        if w == v {
                            continue;
                        }

                        let vw = g.get(v, w);
                        if vw > Weight::ZERO {
                            // w in intersection of neighborhoods.
                            icf += uw.min(vw);
                        } else {
                            // w in symmetric difference of neighborhoods.
                            icp += uw.abs().min(vw.abs());
                        }
                    }
                    for (w, vw) in g.neighbors_with_weights(v) {
                        if w == u {
                            continue;
                        }

                        let uw = g.get(u, w);
                        if uw <= Weight::ZERO {
                            // w in second part of symmetric difference of neighborhoods.
                            icp += vw.abs().min(uw.abs());
                        }
                    }

                    icf += uv.max(Weight::ZERO);
                    icp += (-uv).max(Weight::ZERO);
                }

                let costs = Costs { icf, icp };

                let idx = self.idx(u, v);
                self.cost_store[idx] = costs;
                let idx = self.idx(v, u);
                self.cost_store[idx] = costs;

                let edge_with_branching_num = EdgeWithBranchingNumber {
                    u,
                    v,
                    branching_num: Self::get_branching_number(costs, uv),
                };

                if edge_with_branching_num.branching_num.is_infinite() {
                    continue;
                }

                if self
                    .min_branching_num
                    .map(|b| edge_with_branching_num < b)
                    .unwrap_or(true)
                {
                    self.min_branching_num = Some(edge_with_branching_num);
                }
            }

            u += 1;
        }

        /*for u in g.nodes() {
            for v in (u + 1)..g.size() {
                continue_if_not_present!(g, v);

                let idx = self.idx(u, v);
                let cost = self.cost_store[idx];
                log::debug!(
                    "calculate_all {}-{}: icf {}, icp {}",
                    u,
                    v,
                    cost.icf,
                    cost.icp
                );
            }
        }*/
    }

    pub fn update(&mut self, g: &Graph<Weight>, x: usize, y: usize, prev: Weight, new: Weight) {
        self.add_diff_to_costs(
            x,
            y,
            new.max(Weight::ZERO) - prev.max(Weight::ZERO),
            (-new).max(Weight::ZERO) - (-prev).max(Weight::ZERO),
            new,
        );

        for u in g.nodes() {
            if u == x || u == y {
                continue;
            }

            let uy = g.get(u, y);
            let ux = g.get(u, x);

            // Start with icf(xu) and icp(xu):

            if uy > Weight::ZERO {
                // If uy is an edge, xy may appear in icf(xu) if it is or was also an edge. Update
                // accordingly.
                let icf_contrib_new = if new > Weight::ZERO {
                    new.min(uy)
                } else {
                    Weight::ZERO
                };

                let icf_contrib_prev = if prev > Weight::ZERO {
                    prev.min(uy)
                } else {
                    Weight::ZERO
                };

                // Also, if uy is an edge but xy isn't or wasn't, xy appears in icp(xu).
                let icp_contrib_new = if new <= Weight::ZERO {
                    new.abs().min(uy.abs())
                } else {
                    Weight::ZERO
                };

                let icp_contrib_prev = if prev <= Weight::ZERO {
                    prev.abs().min(uy.abs())
                } else {
                    Weight::ZERO
                };

                self.add_diff_to_costs(
                    x,
                    u,
                    icf_contrib_new - icf_contrib_prev,
                    icp_contrib_new - icp_contrib_prev,
                    ux,
                );
            } else {
                // Alternatively, if uy is not an edge, if xy is or was, xy appears in icp(xu).
                let icp_contrib_new = if new > Weight::ZERO {
                    new.abs().min(uy.abs())
                } else {
                    Weight::ZERO
                };

                let icp_contrib_prev = if prev > Weight::ZERO {
                    prev.abs().min(uy.abs())
                } else {
                    Weight::ZERO
                };

                self.add_diff_to_costs(x, u, Weight::ZERO, icp_contrib_new - icp_contrib_prev, ux);
            }

            // And then icf(yu) and icp(yu):

            if ux > Weight::ZERO {
                // If ux is an edge, xy may appear in icf(yu) if it is or was also an edge. Update
                // accordingly.
                let icf_contrib_new = if new > Weight::ZERO {
                    new.min(ux)
                } else {
                    Weight::ZERO
                };

                let icf_contrib_prev = if prev > Weight::ZERO {
                    prev.min(ux)
                } else {
                    Weight::ZERO
                };

                // Also, if ux is an edge but xy isn't or wasn't, xy appears in icp(yu).
                let icp_contrib_new = if new <= Weight::ZERO {
                    new.abs().min(ux.abs())
                } else {
                    Weight::ZERO
                };

                let icp_contrib_prev = if prev <= Weight::ZERO {
                    prev.abs().min(ux.abs())
                } else {
                    Weight::ZERO
                };

                self.add_diff_to_costs(
                    y,
                    u,
                    icf_contrib_new - icf_contrib_prev,
                    icp_contrib_new - icp_contrib_prev,
                    uy,
                );
            } else {
                // Alternatively, if ux is not an edge, if xy is or was, xy appears in icp(yu).
                let icp_contrib_new = if new > Weight::ZERO {
                    new.abs().min(ux.abs())
                } else {
                    Weight::ZERO
                };

                let icp_contrib_prev = if prev > Weight::ZERO {
                    prev.abs().min(ux.abs())
                } else {
                    Weight::ZERO
                };

                self.add_diff_to_costs(y, u, Weight::ZERO, icp_contrib_new - icp_contrib_prev, uy);
            }
        }
    }

    // Called while `x` is still present in the graph, so we can still know what relationships it
    // had to other nodes previously.
    pub fn update_for_not_present(&mut self, g: &Graph<Weight>, x: usize) {
        // Removing x means that for all u != x, icf(xu) and icp(xu) disappear.
        // Also, for a pair uv, if before xu and xv existed, x was in the common neighborhood of u
        // and v and thus icf(uv) changes.
        // For a pair uv, if before xu existed and xv didn't, or vice-versa, x was in the symmetric
        // difference of the neighborhoods, and thus icp(uv) changes.

        for u in g.nodes() {
            if x == u {
                continue;
            }

            // icf(xu) and icp(xu) are not a thing anymore. We don't need to store anything
            // regarding that however, as we never iterate over all our stored values or something
            // similar. All operations are always based on what is present in the graph passed in,
            // so future ops will ignore these values automatically.
            // The only exception is the min_branching_num; if that edge is current xu we have to
            // discard it.
            if let Some(min_branching_num) = self.min_branching_num {
                if (min_branching_num.u, min_branching_num.v) == (x, u)
                    || (min_branching_num.u, min_branching_num.v) == (u, x)
                {
                    self.oplog
                        .push(Op::UpdateMinBranchingNum(self.min_branching_num));
                    self.min_branching_num = None;
                }
            }

            let xu = g.get(x, u);

            for v in (u + 1)..g.size() {
                if !g.is_present(v) || v == x {
                    continue;
                }

                let xv = g.get(x, v);
                let uv = g.get(u, v);

                if xu > Weight::ZERO && xv > Weight::ZERO {
                    // x has a term in icf(uv) which we need to take out:
                    self.add_diff_to_costs(u, v, -(xu.min(xv)), Weight::ZERO, uv);
                } else if !(xu <= Weight::ZERO && xv <= Weight::ZERO) {
                    // x has a term in icp(uv) which we need to take out:
                    self.add_diff_to_costs(u, v, Weight::ZERO, -(xu.abs().min(xv.abs())), uv);
                }
            }
        }
    }

    fn add_diff_to_costs(
        &mut self,
        u: usize,
        v: usize,
        icf_diff: Weight,
        icp_diff: Weight,
        uv_new: Weight,
    ) {
        let idx = self.idx(u, v);
        let uv_costs = &mut self.cost_store[idx];

        self.oplog.push(Op::UpdateCosts {
            u,
            v,
            prev_costs: *uv_costs,
        });

        uv_costs.icf += icf_diff;
        uv_costs.icp += icp_diff;

        let idx = self.idx(v, u);
        let uv_costs = &mut self.cost_store[idx];
        uv_costs.icf += icf_diff;
        uv_costs.icp += icp_diff;

        let new_branching_num = EdgeWithBranchingNumber {
            u: u.min(v),
            v: u.max(v),
            branching_num: Self::get_branching_number(*uv_costs, uv_new),
        };

        if let Some(current_min) = self.min_branching_num {
            self.oplog
                .push(Op::UpdateMinBranchingNum(self.min_branching_num));
            if new_branching_num <= current_min {
                self.min_branching_num = Some(new_branching_num);
            } else if (current_min.u, current_min.v) == (new_branching_num.u, new_branching_num.v) {
                self.min_branching_num = None;
            }
        } else if new_branching_num.branching_num.is_finite() {
            // This means we may sometimes not actually have the real minimum in there.
            // Might still be worth a try like this, it shouldn't impact correctness.
            // To avoid that, just remove this else block.
            self.oplog
                .push(Op::UpdateMinBranchingNum(self.min_branching_num));
            self.min_branching_num = Some(new_branching_num);
        }
    }

    pub fn get_costs(&self, u: usize, v: usize) -> Costs {
        let idx = self.idx(u, v);
        self.cost_store[idx]
    }

    pub fn get_edge_with_min_branching_number(
        &mut self,
        g: &Graph<Weight>,
    ) -> Option<(usize, usize)> {
        if let Some(min_branching_num) = self.min_branching_num {
            return Some((min_branching_num.u, min_branching_num.v));
        }

        // If we don't currently have a stored min branching number, we'll have to search for it.
        // If we ever remove the else branch in add_diff_to_costs, we may want to set the
        // next-best value as future min here.
        for u in g.nodes() {
            for v in (u + 1)..g.size() {
                continue_if_not_present!(g, v);

                let uv = g.get(u, v);

                let idx = self.idx(u, v);
                let costs = self.cost_store[idx];

                let edge_with_branching_num = EdgeWithBranchingNumber {
                    u,
                    v,
                    branching_num: Self::get_branching_number(costs, uv),
                };

                if edge_with_branching_num.branching_num.is_infinite() {
                    continue;
                }

                if self
                    .min_branching_num
                    .map(|b| edge_with_branching_num < b)
                    .unwrap_or(true)
                {
                    self.oplog
                        .push(Op::UpdateMinBranchingNum(self.min_branching_num));
                    self.min_branching_num = Some(edge_with_branching_num);
                }
            }
        }

        return self.min_branching_num.map(|b| (b.u, b.v));
    }

    pub fn get_branching_number(costs: Costs, uv: Weight) -> Weight {
        let delete_costs = uv.max(Weight::ZERO);

        if delete_costs < 0.0001 || costs.icp.abs() < 0.0001 {
            Weight::INFINITY
        } else {
            // We calculated branching numbers for [0, 50] x [0, 50] in MATLAB and fitted a poly44
            // surface to the results for faster approximation:
            let x = delete_costs;
            let y = costs.icp;

            1.478
                + -0.03714 * x
                + -0.03714 * y
                + 0.001421 * x * x
                + 0.001493 * x * y
                + 0.001421 * y * y
                + -2.543e-05 * x * x * x
                + -2.732e-05 * x * x * y
                + -2.732e-05 * x * y * y
                + -2.543e-05 * y * y * y
                + 1.712e-07 * x * x * x * x
                + 1.858e-07 * x * x * x * y
                + 1.903e-07 * x * x * y * y
                + 1.858e-07 * x * y * y * y
                + 1.712e-07 * y * y * y * y
        }
    }

    pub fn oplog_len(&self) -> usize {
        self.oplog.len()
    }

    pub fn rollback_to(&mut self, oplog_len: usize) {
        for op in self.oplog.drain(oplog_len..).rev().collect::<Vec<_>>() {
            match op {
                Op::UpdateCosts { u, v, prev_costs } => {
                    let idx = self.idx(u, v);
                    self.cost_store[idx] = prev_costs;
                    let idx = self.idx(v, u);
                    self.cost_store[idx] = prev_costs;
                }
                Op::UpdateMinBranchingNum(prev_min) => self.min_branching_num = prev_min,
            }
        }
    }

    #[inline(always)]
    fn idx(&self, u: usize, v: usize) -> usize {
        v + u * self.graph_size
    }
}
