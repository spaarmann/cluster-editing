use crate::Graph;

#[derive(Debug, Clone, Default)]
pub struct CritClique {
    pub vertices: Vec<usize>,
}

pub struct CritCliqueGraph {
    pub cliques: Vec<CritClique>,
    pub graph: Graph,
}

pub fn build_crit_clique_graph(g: &Graph) -> CritCliqueGraph {
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

    let mut crit_graph = Graph::new(cliques.len());

    for c1 in 0..cliques.len() {
        for c2 in 0..cliques.len() {
            if c1 == c2 {
                continue;
            }

            if should_be_neighbors(g, &cliques[c1], &cliques[c2]) {
                crit_graph.set(c1, c2, 1.0);
            }
        }
    }

    CritCliqueGraph {
        cliques,
        graph: crit_graph,
    }
}

fn should_be_neighbors(g: &Graph, c1: &CritClique, c2: &CritClique) -> bool {
    for &u in &c1.vertices {
        for &v in &c2.vertices {
            if !g.has_edge(u, v) {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crit_graph() {
        // This is the example from "Guo: A more effective linear kernelization for cluster
        // editing, 2009", Fig. 1

        let mut graph = Graph::new(9);
        graph.set(0, 1, 1.0);
        graph.set(0, 2, 1.0);
        graph.set(1, 2, 1.0);
        graph.set(2, 3, 1.0);
        graph.set(2, 4, 1.0);
        graph.set(3, 4, 1.0);
        graph.set(3, 5, 1.0);
        graph.set(3, 6, 1.0);
        graph.set(4, 5, 1.0);
        graph.set(4, 6, 1.0);
        graph.set(5, 6, 1.0);
        graph.set(5, 7, 1.0);
        graph.set(5, 8, 1.0);

        let crit = build_crit_clique_graph(&graph);

        assert_eq!(crit.cliques[0].vertices, vec![0, 1]);
        assert_eq!(crit.cliques[1].vertices, vec![2]);
        assert_eq!(crit.cliques[2].vertices, vec![3, 4]);
        assert_eq!(crit.cliques[3].vertices, vec![5]);
        assert_eq!(crit.cliques[4].vertices, vec![6]);
        assert_eq!(crit.cliques[5].vertices, vec![7]);
        assert_eq!(crit.cliques[6].vertices, vec![8]);

        assert_eq!(crit.graph.neighbors(0).collect::<Vec<_>>(), vec![1]);
        assert_eq!(crit.graph.neighbors(1).collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(crit.graph.neighbors(2).collect::<Vec<_>>(), vec![1, 3, 4]);
        assert_eq!(
            crit.graph.neighbors(3).collect::<Vec<_>>(),
            vec![2, 4, 5, 6]
        );
        assert_eq!(crit.graph.neighbors(4).collect::<Vec<_>>(), vec![2, 3]);
        assert_eq!(crit.graph.neighbors(5).collect::<Vec<_>>(), vec![3]);
        assert_eq!(crit.graph.neighbors(6).collect::<Vec<_>>(), vec![3]);
    }
}
