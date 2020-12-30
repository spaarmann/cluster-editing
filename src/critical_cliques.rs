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
    let mut edits = Vec::new();

    let mut any_rules_applied = true;
    while any_rules_applied {
        any_rules_applied = false;

        let mut rule5_state = None;

        let crit = build_crit_clique_graph(g);

        let mut removed_g = vec![false; g.node_count()];

        //log::trace!("[reduce] graph before: {:?}", g);

        for (clique_idx, clique) in crit.cliques.iter().enumerate() {
            let (clique_neighbors, clique_crit_neighbor_count) =
                get_clique_neighbors(clique_idx, &crit, &removed_g);
            let edit_set = calculate_edits_to_remove_clique_and_neighborhood(
                g,
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

            let mut rule3_applicable = false;
            let mut rule4_applicable = false;
            let mut rule4_vertex = None;
            let mut clique_neighbors2 = None;
            if !rule1_applicable && !rule2_applicable {
                // Only calculate this if the other two aren't already true since it's a bit more work
                if clique_len < neighbors_len && clique_len + neighbors_len > total_edit_degree {
                    // TODO: If we keep applying removed_g every while iteration, these calls here
                    // don't actually need to deal with it.

                    let neighbors2 = get_clique_neighbors2(clique_idx, &crit, &removed_g);

                    let threshold = (clique_len + neighbors_len) / 2;
                    for &u in &neighbors2 {
                        let count =
                            count_intersection(g.neighbors(u), &clique_neighbors, &removed_g);
                        if count > threshold {
                            rule4_vertex = Some(u);
                            break;
                        }
                    }

                    if rule5_state.is_none() {
                        rule5_state = Some((
                            clique.clone(),
                            clique_neighbors.clone(),
                            clique_crit_neighbor_count,
                            neighbors2.clone(),
                        ));
                    }

                    rule3_applicable = rule4_vertex.is_none();
                    rule4_applicable = rule4_vertex.is_some();
                    clique_neighbors2 = Some(neighbors2);
                }
            }

            /*log::warn!(
                "rule applicability: [{}, {}, {}, {}]",
                rule1_applicable,
                rule2_applicable,
                rule3_applicable,
                rule4_applicable
            );*/

            if rule1_applicable || rule2_applicable || rule3_applicable {
                let has_reduced = make_clique_and_neighborhood_disjoint_and_remove(
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

                if has_reduced {
                    any_rules_applied = true;
                    break;
                }
            }

            if rule4_applicable {
                // TODO: If we stop applying removed_g every `while` iteration, this needs to take
                // it into account.
                let has_reduced = apply_rule4(
                    g,
                    imap,
                    k,
                    &mut edits,
                    &clique_neighbors,
                    &clique_neighbors2.unwrap(),
                    rule4_vertex.unwrap(),
                );

                if *k < 0.0 {
                    return None;
                }

                if has_reduced {
                    any_rules_applied = true;
                    break;
                }
            }
        }

        if !any_rules_applied && rule5_state.is_some() {
            // If we got here, either no rule was applicable or they did not result in any further
            // reduction, but we found a case where rule 5 should now be applicable.
            // The paper claims that the above condition and the fact that the other rules
            // don#t reduce it further is sufficient to imply this condition. Let's check to be
            // safe for now :)
            // TODO: Might remove this check if I'm convinced it's safe.
            let (clique, clique_neighbors, clique_crit_neighbor_count, clique_neighbors2) =
                rule5_state.unwrap();
            assert!(clique_crit_neighbor_count == 1 && clique_neighbors2.len() == 1);
            log::error!("Applying rule 5");
            let has_reduced = apply_rule5(
                g,
                imap,
                k,
                &mut removed_g,
                &mut edits,
                &clique,
                &clique_neighbors,
            );

            if !has_reduced {
                // All the other rules didn't apply, so we got here, and now 5 didn't do anything
                // either. We're done now.
                break;
            }
            any_rules_applied = true;
        }

        //log::trace!("[reduce] graph after: {:?}", g);

        let removed_count = removed_g.iter().filter(|&&b| b).count();

        if removed_count == 0 {
            continue;
        }

        // Construct a new graph and imap with the vertices we marked for removal actually removed. The
        // new imap still maps from indices into that new graph to the vertices of the original graph
        // the algorithm got as input.

        // TODO: Figure out if it's necessary to do this every `while` iteration or if the
        // reductions are all still valid without it; would also be nice to avoid recomputing the
        // crit clique graph when it's not necessary.

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
    }

    Some(edits)
}

// TODO: COOOOMMMEEEENNNNTTTTSSSS!!!!

/// Gets all the vertices that are neighbors of the critical clique, but not in the clique
/// themselves. No specific order is guaranteed.
fn get_clique_neighbors(
    clique_idx: usize,
    crit_graph: &CritCliqueGraph,
    removed_g: &[bool],
) -> (Vec<usize>, usize) {
    let crit_neighbors = crit_graph.graph.neighbors(clique_idx);
    let mut count = 0;
    let neighborhood = crit_neighbors
        .flat_map(|n| {
            count += 1;
            &crit_graph.cliques[n].vertices
        })
        .copied()
        .filter(|&u| !removed_g[u])
        .collect();
    (neighborhood, count)
}

fn get_clique_neighbors2(
    clique_idx: usize,
    crit_graph: &CritCliqueGraph,
    removed_g: &[bool],
) -> Vec<usize> {
    let crit_neighbors = crit_graph.graph.neighbors(clique_idx).collect::<Vec<_>>();

    crit_neighbors
        .iter()
        .flat_map(|&n| {
            crit_graph
                .graph
                .neighbors(n)
                .filter(|n2| !crit_neighbors.contains(n2))
                .flat_map(|n2| &crit_graph.cliques[n2].vertices)
        })
        .copied()
        .filter(|&u| !removed_g[u])
        .collect()
}

fn count_intersection(n1: impl Iterator<Item = usize>, n2: &[usize], removed_g: &[bool]) -> usize {
    let mut count = 0;
    for u in n1 {
        if !removed_g[u] && n2.contains(&u) {
            count += 1;
        }
    }
    count
}

struct EditSet {
    inserts: Vec<(usize, usize)>,
    deletions: Vec<(usize, usize)>,
    total_edit_degree: usize,
}

fn calculate_edits_to_remove_clique_and_neighborhood(
    g: &Graph,
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
) -> bool {
    for (u, v) in edits_to_perform.inserts {
        let uv = g.get_mut(u, v);
        *k += *uv;
        edits.push(Edit::insert(&imap, u, v));
        *uv = f32::INFINITY;
    }

    for (u, v) in edits_to_perform.deletions {
        let uv = g.get_mut(u, v);
        *k -= *uv;
        edits.push(Edit::delete(&imap, u, v));
        *uv = f32::NEG_INFINITY;
    }

    // Now mark the clique and its neighbors as "removed" from the graph, so future reduction and
    // algorithm steps ignore it. (It is now a disjoint clique, i.e. already done.)
    for &u in clique_neighbors {
        removed_g[u] = true;
    }
    for &u in &clique.vertices {
        removed_g[u] = true;
    }

    clique_neighbors.len() > 0 || clique.vertices.len() > 0
}

fn apply_rule4(
    g: &mut Graph,
    imap: &IndexMap,
    k: &mut f32,
    edits: &mut Vec<Edit>,
    clique_neighbors: &[usize],
    clique_neighbors2: &[usize],
    u: usize,
) -> bool {
    // Insert edges in neighborhood to make clique+neighborhood a clique.
    let mut has_done_edit = false;

    for i in 0..clique_neighbors.len() {
        let v = clique_neighbors[i];

        // Add edges to other clique neighbors.
        for j in (i + 1)..clique_neighbors.len() {
            let w = clique_neighbors[j];

            let vw = g.get_mut(v, w);
            if *vw < 0.0 {
                *k += *vw;
                edits.push(Edit::insert(&imap, v, w));
                *vw = f32::INFINITY;

                has_done_edit = true;
            }
        }
    }

    // Remove edges between clique_neighbors and clique_neighbors2-u
    for &v in clique_neighbors {
        for &w in clique_neighbors2 {
            if w == u {
                continue;
            }

            let vw = g.get_mut(v, w);
            if *vw > 0.0 {
                *k -= *vw;
                edits.push(Edit::delete(&imap, v, w));
                *vw = f32::NEG_INFINITY;

                has_done_edit = true;
            }
        }
    }

    has_done_edit
}

fn apply_rule5(
    g: &mut Graph,
    imap: &IndexMap,
    k: &mut f32,
    removed_g: &mut [bool],
    edits: &mut Vec<Edit>,
    clique: &CritClique,
    clique_neighbors: &[usize],
) -> bool {
    // Can pick any set of |clique| vertices in clique_neighbors, we'll just use the first |clique|
    // verts.
    // Then, remove (clique + that set) from G, and set k = k - |clique|.
    // Note that the modification to k does not actually correspond directly to the edge edits we
    // do, but this is what the paper has proven to be correct *shrug*.

    let clique_size = clique.vertices.len();
    let to_remove = clique
        .vertices
        .iter()
        .chain(clique_neighbors[..clique_size].iter())
        .copied()
        .collect::<Vec<_>>();

    for &u in &to_remove {
        removed_g[u] = true;

        for v in 0..g.node_count() {
            if removed_g[v] {
                continue;
            }

            let uv = g.get_mut(u, v);
            if *uv > 0.0 {
                edits.push(Edit::delete(imap, u, v));
                *uv = f32::NEG_INFINITY;
            }
        }
    }

    *k = *k - clique_size as f32;

    to_remove.len() > 0
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
