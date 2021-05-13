use crate::{
    graph::{Graph, GraphWeight},
    Weight,
};

#[derive(Copy, Clone, Debug)]
pub struct Costs {
    pub icf: Weight,
    pub icp: Weight,
}

#[derive(Clone)]
pub struct InducedCosts {
    cost_store: Vec<Costs>,
    graph_size: usize,
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
            //       oplog: Vec::new(),
        };

        this.calculate_all_costs(g);

        this
    }

    fn calculate_all_costs(&mut self, g: &Graph<Weight>) {
        let mut u_neighbors = Vec::new();

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

                let idx = self.idx(u, v);
                self.cost_store[idx] = Costs { icf, icp };
                let idx = self.idx(v, u);
                self.cost_store[idx] = Costs { icf, icp };
            }

            u += 1;
        }
    }

    pub fn update(&mut self, g: &Graph<Weight>, x: usize, y: usize, prev: Weight, new: Weight) {
        self.add_diff_to_costs(
            x,
            y,
            new.max(Weight::ZERO) - prev.max(Weight::ZERO),
            (-new).max(Weight::ZERO) - (-prev).max(Weight::ZERO),
        );

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

                self.add_diff_to_costs(x, u, Weight::ZERO, icp_contrib_new - icp_contrib_prev);
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

                self.add_diff_to_costs(y, u, Weight::ZERO, icp_contrib_new - icp_contrib_prev);
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

    fn add_diff_to_costs(&mut self, u: usize, v: usize, icf_diff: Weight, icp_diff: Weight) {
        let idx = self.idx(u, v);
        let uv_costs = &mut self.cost_store[idx];
        uv_costs.icf += icf_diff;
        uv_costs.icp += icp_diff;

        let idx = self.idx(v, u);
        let uv_costs = &mut self.cost_store[idx];
        uv_costs.icf += icf_diff;
        uv_costs.icp += icp_diff;
    }

    pub fn get_costs(&self, u: usize, v: usize) -> Costs {
        let idx = self.idx(u, v);
        self.cost_store[idx]
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
