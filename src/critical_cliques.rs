use crate::Graph;

#[derive(Debug, Clone, Default)]
pub struct CritClique {
    pub vertices: Vec<usize>,
}

pub fn find_crit_cliques(g: &Graph) -> Vec<CritClique> {
    let mut cliques = Vec::new();

    // TODO: This looks at least O(n^2) but should apparently be do-able in O(n + m), so have
    // another look at making this more efficient.

    let mut visited = vec![false; g.node_count()];

    for u in g.nodes() {
        if visited[u] {
            continue;
        }

        visited[u] = true;
        let mut clique = CritClique::default();
        clique.vertices.push(u);

        for v in g.nodes() {
            if visited[v] {
                continue;
            }

            // TODO: Is it maybe worth storing neighbor sets instead of recomputing them?

            if g.closed_neighbors(u).eq(g.closed_neighbors(v)) {
                clique.vertices.push(v);

                visited[v] = true;
            }
        }

        cliques.push(clique);
    }

    cliques
}
