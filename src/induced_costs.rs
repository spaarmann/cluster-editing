use crate::{
    graph::{Graph, GraphWeight, IndexMap},
    Weight,
};

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

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
            Reverse(self.branching_num)
                .partial_cmp(&Reverse(other.branching_num))
                .unwrap()
                .then_with(|| Reverse(self.u).cmp(&Reverse(other.u)))
                .then_with(|| Reverse(self.v).cmp(&Reverse(other.v))),
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
    branching_nums: BinaryHeap<EdgeWithBranchingNumber>,
    // TODO: Do Oplog instead of cloning once update_for_not_present exists properly.
    //oplog: Vec<Op>,
}

/*#[derive(Clone)]
pub enum Op {
    Update {
        u: usize,
        v: usize,
        prev_icf: Weight,
        prev_icp: Weight,
    },
}*/

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
            branching_nums: BinaryHeap::new(),
            //       oplog: Vec::new(),
        };

        this.calculate_all_costs(g);

        this
    }

    fn calculate_all_costs(&mut self, g: &Graph<Weight>) {
        let mut u_neighbors = Vec::new();

        self.branching_nums.clear();

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
                    branching_num: Self::get_branching_number(costs, uv, false),
                };
                self.branching_nums.push(edge_with_branching_num);
            }

            u += 1;
        }
    }

    pub fn update(&mut self, g: &Graph<Weight>, x: usize, y: usize, prev: Weight, new: Weight) {
        /*if (x, y) == (7, 11) || (x, y) == (11, 7) {
            let idx = self.idx(x, y);
            log::debug!(
                "Setting {}-{} to {} from {}, previous costs {:?}, prev branch number {}",
                x,
                y,
                prev,
                new,
                self.cost_store[idx],
                self.branching_nums
                    .iter()
                    .find(|&b| (b.u, b.v) == (x, y) || (b.v, b.u) == (x, y))
                    .unwrap()
                    .branching_num
            );
        }*/

        self.add_diff_to_costs(
            x,
            y,
            new.max(Weight::ZERO) - prev.max(Weight::ZERO),
            (-new).max(Weight::ZERO) - (-prev).max(Weight::ZERO),
            new,
        );

        /*if (x, y) == (7, 11) || (x, y) == (11, 7) {
            let idx = self.idx(x, y);
            log::debug!(
                "{}-{} new costs {:?}, new branching number {}",
                x,
                y,
                self.cost_store[idx],
                self.branching_nums
                    .iter()
                    .find(|&b| (b.u, b.v) == (x, y) || (b.v, b.u) == (x, y))
                    .unwrap()
                    .branching_num
            );
        }*/

        for u in g.nodes() {
            if u == x || u == y {
                continue;
            }

            let uy = g.get(u, y);
            let ux = g.get(u, x);

            // icf(xu) = sum_{v in N(x)+N(u)} xv.min(uv)
            // We changed xy, there is potentially a term with v=y in that sum, which we would need
            // to update.

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

            // First update icf(xu)
            /*if uy > Weight::ZERO {
                // y was and is a neighbor u

                if new > Weight::ZERO && prev > Weight::ZERO {
                    // y also was and is a neighbor of x, it thus appears in the icf(xu) calculation
                    // both before and after
                    self.add_diff_to_costs(x, u, new.min(uy) - prev.min(uy), Weight::ZERO);
                } else if new > Weight::ZERO && prev <= Weight::ZERO {
                    // y wasn't a neighbor of x, but it is now, so it now appears in icf(xu)
                    self.add_diff_to_costs(x, u, new.min(uy), Weight::ZERO);
                } else if new <= Weight::ZERO && prev > Weight::ZERO {
                    // y was a neighbor of x, but isn't now, so it used to appear in icf(xu) but
                    // now doesn't
                    self.add_diff_to_costs(x, u, -prev.min(uy), Weight::ZERO);
                } else {
                    // y isn't and wasn't a neighbor of x, so it didn't and doesn't appear in
                    // icf(xu)
                }
            }*/
        }
    }

    pub fn update_for_not_present(&mut self, g: &Graph<Weight>, _x: usize) {
        // TODO: Write an update for this
        self.calculate_all_costs(g);
    }

    fn add_diff_to_costs(
        &mut self,
        u: usize,
        v: usize,
        icf_diff: Weight,
        icp_diff: Weight,
        uv: Weight,
    ) {
        let idx = self.idx(u, v);
        let uv_costs = &mut self.cost_store[idx];
        uv_costs.icf += icf_diff;
        uv_costs.icp += icp_diff;

        let idx = self.idx(v, u);
        let uv_costs = &mut self.cost_store[idx];
        uv_costs.icf += icf_diff;
        uv_costs.icp += icp_diff;

        // Remove old entry from binary heap
        self.branching_nums
            .retain(|&b| !(b.u == u && b.v == v) && !(b.u == v && b.v == u));

        // Add new entry
        self.branching_nums.push(EdgeWithBranchingNumber {
            u: u.min(v),
            v: u.max(v),
            branching_num: Self::get_branching_number(
                *uv_costs,
                uv,
                (u, v) == (7, 11) || (u, v) == (11, 7),
            ),
        });
    }

    pub fn get_costs(&self, u: usize, v: usize) -> Costs {
        let idx = self.idx(u, v);
        self.cost_store[idx]
    }

    pub fn get_edge_with_min_branching_number(
        &mut self, // TODO: This sohuldn't need to be mutable when done debugging
        g: &Graph<Weight>,
        imap: &IndexMap,
    ) -> Option<(usize, usize)> {
        // TODO: Using the heap here currently results in exact003 going way past k=42 for some
        // reason. First diff in execution is choosing a different edge to branch on at some
        // points, but the two have the same branching number...
        // In theory that shouldn't matter? But it makes comparing to figure out where else
        // something goes wrong harder.
        // We could probably force the same order by sorting the elements of the heap by vertex
        // index if the numbers are equal, since the loop below will pick the *first* it finds and
        // iterates in order.

        let res = self
            .branching_nums
            .peek()
            .map(|&b| if b.u < b.v { (b.u, b.v) } else { (b.v, b.u) });
        //return res;

        //if let Some((1, 8)) = res {
        /*if let Some((u, v)) = res {
            log::debug!(
                "Chose ({}, {}) = ({:?}, {:?}) as min branching number, heap is currently: {:?}",
                u,
                v,
                "foo", //imap[u],
                "foo", //imap[v],
                self.branching_nums
            );

            /*let top = self.branching_nums.pop().unwrap();
            let second = *self.branching_nums.peek().unwrap();

            // Restore value on the heap
            self.branching_nums.push(top);

            log::debug!(
                "Top is {:?}, second is {:?}, brnum partial_cmp {:?}, whole partial_cmp {:?}",
                top,
                second,
                top.branching_num.partial_cmp(&second.branching_num),
                top.partial_cmp(&second)
            );

            let idx = self.idx(u, v);
            log::debug!(
                "{}-{} edge is {}, costs are {:?}",
                u,
                v,
                g.get(u, v),
                self.cost_store[idx]
            );*/
        }*/

        return res;

        /*let mut best = None;
        let mut best_val = Weight::INFINITY;
        for u in g.nodes() {
            for v in (u + 1)..g.size() {
                continue_if_not_present!(g, v);

                let idx = self.idx(u, v);
                let costs = self.cost_store[idx];
                let branching_num = Self::get_branching_number(costs, g.get(u, v), false);

                /*if (u, v) == (7, 11) || (u, v) == (3, 4) {
                    log::debug!(
                        "Branching number for {}-{} is {} with costs {:?} and uv {}",
                        u,
                        v,
                        branching_num,
                        costs,
                        g.get(u, v)
                    );
                }*/

                if branching_num < best_val {
                    best = Some((u, v));
                    best_val = branching_num;
                }
            }
        }

        best*/
    }

    fn get_branching_number(costs: Costs, uv: Weight, log: bool) -> Weight {
        let delete_costs = uv.max(Weight::ZERO);

        if delete_costs < 0.0001 || costs.icp.abs() < 0.0001 {
            /*if log {
                log::debug!(
                    "BranchNum INFINITY from del_cost {}, {:?}, uv {}",
                    delete_costs,
                    costs,
                    uv
                );
            }*/
            Weight::INFINITY
        } else {
            // We calculated branching numbers for [0, 50] x [0, 50] in MATLAB and fitted a poly44
            // surface to the results for faster approximation:
            let x = delete_costs;
            let y = costs.icp;

            /*if log {
                log::debug!(
                    "Non-INFINITY branchnum from del_cost {}, {:?}, uv {}",
                    delete_costs,
                    costs,
                    uv
                );
            }*/

            1.478
                + -0.03714 * x
                + -0.3714 * y
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

    /*pub fn oplog_len(&self) -> usize {
        self.oplog.len()
    }

    pub fn rollback_to(&mut self, oplog_len: usize) {
        for op in self.oplog.drain(oplog_len..).rev().collect::<Vec<_>>() {
            match op {
                // TODO
            }
        }
    }*/

    #[inline(always)]
    fn idx(&self, u: usize, v: usize) -> usize {
        v + u * self.graph_size
    }
}
