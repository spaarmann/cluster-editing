use crate::{algo::ProblemInstance, graph::GraphWeight, Weight};

use log::info;

pub fn calculate_upper_bound(mut p: ProblemInstance) -> f32 {
    // TODO: It's a bit awkward to just hardcode a max here, but oh well
    p.k = 100000.0;
    p.k_max = p.k;

    info!("Starting upper bound computation...");

    while p.conflicts.conflict_count() > 0 {
        let mut max: Option<(usize, usize, Weight)> = None;

        for u in p.g.nodes() {
            for v in (u + 1)..p.g.size() {
                continue_if_not_present!(p.g, v);

                let uv = p.g.get(u, v);
                let costs = p.induced_costs.get_costs(u, v);
                let bound = p
                    .conflicts
                    .min_cost_to_resolve_edge_disjoint_conflicts_ignoring(&p.g, u, v);

                let max_op = costs.icp.max(costs.icf) + bound;

                if uv.is_finite() && max.map(|(_, _, w)| max_op > w).unwrap_or(true) {
                    max = Some((u, v, max_op));
                }
            }
        }

        let max = max.unwrap();
        let (u, v) = (max.0, max.1);
        let max_costs = p.induced_costs.get_costs(u, v);
        let uv = p.g.get(u, v);

        assert!(
            uv.is_finite(),
            "Chose an already forbidden edge to modify: {}-{} with weight {} and {:?}!",
            u,
            v,
            uv,
            max_costs,
        );

        if max_costs.icf.is_finite() && max_costs.icf > max_costs.icp {
            if uv > Weight::ZERO {
                p.k -= uv;
                p.make_delete_edit(u, v);
            }
            if uv.abs() < 0.001 {
                p.k -= 0.5;
            }

            p.set(u, v, Weight::NEG_INFINITY);
        } else if max_costs.icp.is_finite() && max_costs.icp >= max_costs.icf {
            // Merge
            p.merge(u, v);
        } else {
            panic!("Max edge has both induced costs infinite: {:?}", max_costs);
        }
    }

    info!("Finished calculating upper bound.");

    p.k_max - p.k
}
