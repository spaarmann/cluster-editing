use crate::algo::Edit;
use crate::{graph::IndexMap, Graph};

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

// Chen and Meng: A 2k Kernel for the Cluster Editing Problem, 2010
pub fn apply_reductions(g: &mut Graph, imap: &mut IndexMap, k: &mut f32) -> Option<Vec<Edit>> {
    let crit = build_crit_clique_graph(g);

    let mut removed_g = vec![false; g.node_count()];
    let mut edits = Vec::new();

    //log::trace!("[reduce] graph before: {:?}", g);

    for clique in crit.cliques {
        // Rule 1
        if clique.vertices.len() as f32 > *k {
            make_clique_and_neighborhood_disjoint_and_remove(
                g,
                imap,
                k,
                &mut removed_g,
                &mut edits,
                &clique,
            );

            if *k < 0.0 {
                return None;
            }
        }

        // Rule 2
        //if clique.vertices.len() >= |N(K)|
    }

    //log::trace!("[reduce] graph after: {:?}", g);

    let removed_count = removed_g.iter().filter(|&&b| b).count();

    if removed_count == 0 {
        return Some(edits);
    }

    // Construct a new graph and imap with the vertices we marked for removal actually removed. The
    // new imap still maps from indices into that new graph to the vertices of the original graph
    // the algorithm got as input.

    // TODO: Possibly test whether it's faster to just keep the removed_g map around in a larger
    // scope rather than creating the graph here.

    let mut new_g = Graph::new(g.node_count() - removed_count);
    let mut new_imap = IndexMap::new(new_g.node_count());
    let mut new_vertex = 0;

    let mut reverse_imap = IndexMap::new(g.node_count());

    for u in 0..g.node_count() {
        if removed_g[u] {
            continue;
        }

        for v in g.neighbors(u) {
            if v > u || removed_g[v] {
                continue;
            }

            new_g.set_direct(reverse_imap[v], new_vertex, g.get_direct(v, u));
        }

        reverse_imap[u] = new_vertex;
        new_imap[new_vertex] = imap[u];
        new_vertex += 1;
    }

    *g = new_g;
    *imap = new_imap;

    Some(edits)
}

/// This (somewhat awkwardly named) function will make clique + N(clique) a disjoint clique in g by
/// performing appropriate edge edits, mark those vertices as removed, adjust k accordingly, and
/// store the set of edits made.
fn make_clique_and_neighborhood_disjoint_and_remove(
    g: &mut Graph,
    imap: &mut IndexMap,
    k: &mut f32,
    removed_g: &mut Vec<bool>,
    edits: &mut Vec<Edit>,
    clique: &CritClique,
) {
    // Contains the critical clique and all vertices neighboring the clique.
    let vertices = g
        .closed_neighbors(*clique.vertices.first().unwrap())
        .collect::<Vec<_>>();

    /*log::trace!(
        "[reduce] isolating {:?} based on clique {:?}",
        vertices,
        clique.vertices
    );*/

    for i in 0..vertices.len() {
        let u = vertices[i];

        // Skip vertices that were already "removed" by earlier reduction steps.
        if removed_g[u] {
            continue;
        }

        // Make sure u is connected to every other vertex in vertices,
        for j in (i + 1)..vertices.len() {
            let v = vertices[j];
            if removed_g[u] {
                continue;
            }

            let uv = g.get_mut_direct(u, v);

            if *uv < 0.0 {
                *k += *uv;
                edits.push(Edit::insert(&imap, u, v));
                *uv = f32::INFINITY;
            }
        }

        // ... but not to anything else. (We want to transform vertices into a disjoint clique!)
        for v in 0..g.node_count() {
            if removed_g[v] {
                continue;
            }

            // TODO: Try using a BTreeSet for vertices, or using some kind of other iteration
            // strategy to avoid the linear search here.
            if vertices.contains(&v) {
                continue;
            }

            let uv = g.get_mut(u, v);
            if *uv > 0.0 {
                *k -= *uv;
                //log::trace!("[reduce] deleting {}-{} which has weight {}", u, v, *uv);
                edits.push(Edit::delete(&imap, u, v));
                *uv = f32::NEG_INFINITY;
            }
        }

        // And at the end, all of the disjoint clique should just be removed from the graph.
        removed_g[u] = true;
    }
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
