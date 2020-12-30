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

    for (clique_idx, clique) in crit.cliques.iter().enumerate() {
        let clique_neighbors = get_clique_neighborhood(clique_idx, &crit);
        let edit_set = calculate_edits_to_remove_clique_and_neighborhood(
            g,
            imap,
            &removed_g,
            clique,
            &clique_neighbors,
        );

        let clique_len = clique.vertices.len();
        let neighbors_len = clique_neighbors.len();
        let total_edit_degree = edit_set.total_edit_degree;

        let rule1_applicable = clique_len as f32 > *k;
        let rule2_applicable =
            clique_len >= neighbors_len && clique_len + neighbors_len > total_edit_degree;

        if rule1_applicable || rule2_applicable {
            /*log::warn!(
                "Applying reduction, applicability: [{}, {}], k before: {}",
                rule1_applicable,
                rule2_applicable,
                *k
            );*/

            make_clique_and_neighborhood_disjoint_and_remove(
                g,
                imap,
                k,
                &mut removed_g,
                &mut edits,
                edit_set,
                &clique,
                &clique_neighbors,
            );

            //log::warn!("k after: {}", *k);

            if *k < 0.0 {
                return None;
            }

            continue;
        }
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

/// Gets all the vertices that are neighbors of the critical clique, but not in the clique
/// themselves. No specific order is guaranteed.
fn get_clique_neighborhood(clique_idx: usize, crit_graph: &CritCliqueGraph) -> Vec<usize> {
    crit_graph
        .graph
        .neighbors(clique_idx)
        .flat_map(|n| &crit_graph.cliques[n].vertices)
        .copied()
        .collect()
}

struct EditSet {
    inserts: Vec<(usize, usize)>,
    deletions: Vec<(usize, usize)>,
    total_edit_degree: usize,
}

fn calculate_edits_to_remove_clique_and_neighborhood(
    g: &Graph,
    imap: &IndexMap,
    removed_g: &[bool],
    clique: &CritClique,
    clique_neighbors: &[usize],
) -> EditSet {
    // Everything in the clique is already connected with the rest of the clique (it's a clique!).
    // All the neighbors are also connected to all the vertices in the clique, because all the
    // clique vertices have the *same set* of neighbors outside the clique (it's a *critical*
    // clique!).
    // So we only need to add edges between the different groups of neighbors.
    //
    // The only edges that we need to remove are between the neighbors of the clique to any nodes
    // that are neither in the neighbors nor the clique itself. (The vertices in the clique
    // obviously don't have any such neighbors, so there's nothing to remove.)

    let mut edits = EditSet {
        inserts: Vec::new(),
        deletions: Vec::new(),
        total_edit_degree: 0,
    };

    for i in 0..clique_neighbors.len() {
        let u = clique_neighbors[i];
        if removed_g[u] {
            continue;
        }

        // Add edges to other clique neighbors.
        for j in (i + 1)..clique_neighbors.len() {
            let v = clique_neighbors[j];
            if removed_g[v] {
                continue;
            }

            if g.get(u, v) < 0.0 {
                edits.inserts.push((u, v));

                // Increase total degree twice: we only add the (u, v) edge once but it would be
                // counted in the edit degree for both u and v
                edits.total_edit_degree += 2;
            }
        }

        // Remove edges to unrelated vertices.
        // TODO: Try using a BTreeSet for neighbors and vertices, or using some kind of other iteration
        // strategy to avoid the linear search here.
        for v in 0..g.node_count() {
            if u == v || removed_g[v] {
                continue;
            }

            if clique_neighbors.contains(&v) || clique.vertices.contains(&v) {
                continue;
            }

            if g.get(u, v) > 0.0 {
                edits.deletions.push((u, v));

                // Here the degree is only increased once: it would only count for u, since v isn't
                // even in the neighborhood and thus not considered.
                edits.total_edit_degree += 1;
            }
        }
    }

    edits
}

fn make_clique_and_neighborhood_disjoint_and_remove(
    g: &mut Graph,
    imap: &IndexMap,
    k: &mut f32,
    removed_g: &mut [bool],
    edits: &mut Vec<Edit>,
    edits_to_perform: EditSet,
    clique: &CritClique,
    clique_neighbors: &[usize],
) {
    for (u, v) in edits_to_perform.inserts {
        let uv = g.get_mut(u, v);
        if *uv < 0.0 {
            *k += *uv;
            edits.push(Edit::insert(&imap, u, v));
            *uv = f32::INFINITY;
        }
    }

    for (u, v) in edits_to_perform.deletions {
        let uv = g.get_mut(u, v);
        if *uv > 0.0 {
            *k -= *uv;
            edits.push(Edit::delete(&imap, u, v));
            *uv = f32::NEG_INFINITY;
        }
    }

    // Now mark the clique and its neighbors as "removed" from the graph, so future reduction and
    // algorithm steps ignore it. (It is now a disjoint clique, i.e. already done.)
    for &u in clique_neighbors {
        removed_g[u] = true;
    }
    for &u in &clique.vertices {
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
