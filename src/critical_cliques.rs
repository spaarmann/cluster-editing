use crate::Graph;

use std::collections::HashSet;

use petgraph::graph::NodeIndex;

#[derive(Debug, Clone, Default)]
pub struct CritClique {
    pub vertices: Vec<u32>,
    pub node_indices: Vec<NodeIndex>,
}

pub fn find_crit_cliques(g: &Graph) -> Vec<CritClique> {
    let mut cliques = Vec::new();

    // TODO: This looks at least O(n^2) but should apparently be do-able in O(n + m), so have
    // another look at making this more efficient.

    let mut visited = vec![false; g.node_count()];

    for u in g.node_indices() {
        if visited[u.index()] {
            continue;
        }

        visited[u.index()] = true;
        let mut clique = CritClique::default();
        clique.node_indices.push(u);
        clique.vertices.push(*g.node_weight(u).unwrap());

        // We need to be able to compare neighborhoods, so we have to collect them into some
        // collection. Since I'm not sure we can rely on a particular order being returned by
        // `.neighbors`, use a `HashSet` for now.
        // TODO: If we do transition to using a Graph representation that guarantees order of
        // neighbors, it's probably more efficient to not use HashSets to store them and instead either
        // use a Vec or just compare the iterators directly.
        // TODO: If that doesn't happen, it'd be nice to at least not recompute them multiple
        // times, but the borrow checker has been making that a bit annoying.

        let neighbors = std::iter::once(u)
            .chain(g.neighbors(u))
            .collect::<HashSet<_>>();

        for v in g.node_indices() {
            if visited[v.index()] {
                continue;
            }

            let v_neighbors = std::iter::once(v)
                .chain(g.neighbors(v))
                .collect::<HashSet<_>>();

            if neighbors == v_neighbors {
                clique.node_indices.push(v);
                clique.vertices.push(*g.node_weight(v).unwrap());

                visited[v.index()] = true;
            }
        }

        cliques.push(clique);
    }

    cliques
}
