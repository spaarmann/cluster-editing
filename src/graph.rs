/// An undirected graph without self-loops that stores weights for every pair of vertices (so, for
/// edges and for non-edges).
/// Edges have weights >0, non-edges have weights <0.
///
/// Accessing self-loop weights will panic!
///
/// TODO: How exactly do we handle edges with weight =0?
#[derive(Clone, Debug)]
pub struct Graph {
    /// The graph is internally stored as a linearized triangular matrix.
    /// An example matrix for a graph with four vertices might look like this:
    /// ```txt
    ///     u v w x
    ///   +--------
    /// u | - - - -
    /// v | a - - -
    /// w | b c - -
    /// x | d e f -
    /// ```
    /// This would be stored as `[a, b, c, d, e, f]` in the `matrix` field.
    matrix: Vec<f32>,
    size: usize,
    /// Stores a mapping from row index to the starting index of that row in the `matrix` list.
    row_offsets: Vec<usize>,
}

impl Graph {
    /// Creates a new empty (without any edges) graph with `size` vertices, with all weights set to
    /// -1.0.
    pub fn new(size: usize) -> Self {
        let mat_size = (size * (size - 1) / 2) as usize;

        let row_offsets =
            // Row 0 does not exist, so use a marker value that will definitely panic if any code
            // tries to index using it.
            std::iter::once(usize::MAX)
            // For all other rows, calculate the correct offset.
            .chain((1..size).map(|i| i * (i - 1) / 2))
            .collect();
        Graph {
            size,
            matrix: vec![-1.0; mat_size],
            row_offsets,
        }
    }

    /// Creates a new graph from an existing petgraph graph. Edges existing in the input graph are
    /// given weight 1.0, non-edges are given weight -1.0.
    /// A corresponding `IndexMap` is created that maps vertex indices from the returned graph to
    /// the weights associated with the vertices in the petgraph graph.
    pub fn new_from_petgraph(pg: &crate::PetGraph) -> (Self, IndexMap) {
        let mut g = Self::new(pg.node_count());
        let mut imap = IndexMap::new(pg.node_count());

        for v in pg.node_indices() {
            imap[v.index()] = *pg.node_weight(v).unwrap();
        }

        for e in pg.edge_indices() {
            let (u, v) = pg.edge_endpoints(e).unwrap();
            g.set(u.index(), v.index(), 1.0);
        }

        (g, imap)
    }

    /// Creates a petgraph graph from this graph.
    pub fn into_petgraph(&self, imap: Option<&IndexMap>) -> crate::PetGraph {
        use petgraph::prelude::NodeIndex;

        let mut pg = crate::PetGraph::with_capacity(self.size, 0);

        for u in 0..self.size {
            pg.add_node(imap.map(|m| m[u]).unwrap_or(u));
        }

        for u in 0..self.size {
            for v in (u + 1)..self.size {
                if self.get_direct(u, v) > 0.0 {
                    pg.add_edge(NodeIndex::new(u), NodeIndex::new(v), 0);
                }
            }
        }

        pg
    }

    /// Get the weight associated with pair `(u, v)`.
    /// u and v can be in any order, panics if `u == v`.
    pub fn get(&self, u: usize, v: usize) -> f32 {
        assert_ne!(u, v);

        if u < v {
            self.matrix[self.row_offsets[v] + u]
        } else {
            self.matrix[self.row_offsets[u] + v]
        }
    }

    /// Get the weight associated with pair `(u, v)`.
    /// u and v can be in any order, panics if `u == v`.
    pub fn get_ref(&self, u: usize, v: usize) -> &f32 {
        assert_ne!(u, v);

        if u < v {
            &self.matrix[self.row_offsets[v] + u]
        } else {
            &self.matrix[self.row_offsets[u] + v]
        }
    }

    /// Get a mutable reference to the weight associated with pair `(u, v)`.
    /// u and v can be in any order, panics if `u == v`.
    pub fn get_mut(&mut self, u: usize, v: usize) -> &mut f32 {
        assert_ne!(u, v);

        if u < v {
            &mut self.matrix[self.row_offsets[v] + u]
        } else {
            &mut self.matrix[self.row_offsets[u] + v]
        }
    }

    /// Set the weight associated with pair `(u, v)`.
    /// u and v can be in any order, panics if `u == v`.
    pub fn set(&mut self, u: usize, v: usize, w: f32) {
        assert_ne!(u, v);

        if u < v {
            self.matrix[self.row_offsets[v] + u] = w;
        } else {
            self.matrix[self.row_offsets[u] + v] = w;
        }
    }

    /// Like `get`, but assumes `u != v` and `u < v` instead of checking both.
    pub fn get_direct(&self, u: usize, v: usize) -> f32 {
        self.matrix[self.row_offsets[v] + u]
    }
    /// Like `get_ref`, but assumes `u != v` and `u < v` instead of checking both.
    pub fn get_ref_direct(&self, u: usize, v: usize) -> &f32 {
        &self.matrix[self.row_offsets[v] + u]
    }
    /// Like `get_mut`, but assumes `u != v` and `u < v` instead of checking both.
    pub fn get_mut_direct(&mut self, u: usize, v: usize) -> &mut f32 {
        &mut self.matrix[self.row_offsets[v] + u]
    }
    /// Like `set`, but assumes `u != v` and `u < v` instead of checking both.
    pub fn set_direct(&mut self, u: usize, v: usize, w: f32) {
        self.matrix[self.row_offsets[v] + u] = w;
    }

    /// Returns an iterator over the open neighborhood of `u` (i.e., not including `u` itself). The
    /// neighbors are guaranteed to be in ascending order.
    pub fn neighbors(&self, u: usize) -> impl Iterator<Item = usize> + '_ {
        (0..u)
            .filter(move |&v| self.get_direct(v, u) > 0.0)
            .chain(((u + 1)..self.size).filter(move |&v| self.get_direct(u, v) > 0.0))
    }

    /// Returns an iterator over the closed neighborhood of `u` (i.e., including `u` itself). The
    /// neighbors are guaranteed to be in ascending order.
    pub fn closed_neighbors(&self, u: usize) -> impl Iterator<Item = usize> + '_ {
        (0..u)
            .filter(move |&v| self.get_direct(v, u) > 0.0)
            .chain(std::iter::once(u))
            .chain(((u + 1)..self.size).filter(move |&v| self.get_direct(u, v) > 0.0))
    }

    /// Returns the number of nodes in this graph.
    pub fn node_count(&self) -> usize {
        self.size
    }

    /// Returns an iterator over all the nodes present in the graph.
    pub fn nodes(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.size
    }

    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.get(u, v) > 0.0
    }

    pub fn has_edge_direct(&self, u: usize, v: usize) -> bool {
        self.get_direct(u, v) > 0.0
    }
}

impl std::ops::Index<(usize, usize)> for Graph {
    type Output = f32;
    /// Semantics equivalent to `Graph::get_ref`.
    fn index(&self, (u, v): (usize, usize)) -> &Self::Output {
        self.get_ref(u, v)
    }
}

impl std::ops::IndexMut<(usize, usize)> for Graph {
    /// Semantics equivalent to `Graph::get_mut`.
    fn index_mut(&mut self, (u, v): (usize, usize)) -> &mut Self::Output {
        self.get_mut(u, v)
    }
}

/// Companion to the `Graph` struct for remapping to different indices.
///
/// The `Graph` struct uses indices `0..size` for the vertices stored within.
/// This is fine (and even advantageous) for operating on a single `Graph` instance,
/// but in some contexts it is necessary to e.g. map those indices to those of the original input
/// graph. An `IndexMap` can store such a mapping.
pub struct IndexMap {
    map: Vec<usize>,
}

impl IndexMap {
    pub fn new(size: usize) -> Self {
        Self { map: vec![0; size] }
    }
}

impl std::ops::Index<usize> for IndexMap {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.map[index]
    }
}

impl std::ops::IndexMut<usize> for IndexMap {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.map[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_graph() -> Graph {
        // 0 -- 1
        // |    |
        // 2    3

        let mut g = Graph::new(4);
        g.set(0, 1, 1.0);
        g.set(0, 2, 1.0);
        g.set(1, 3, 1.0);
        g
    }

    #[test]
    fn neighbors() {
        let g = example_graph();

        assert_eq!(g.neighbors(0).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(g.neighbors(1).collect::<Vec<_>>(), vec![0, 3]);
        assert_eq!(g.neighbors(2).collect::<Vec<_>>(), vec![0]);
        assert_eq!(g.neighbors(3).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn closed_neighbors() {
        let g = example_graph();

        assert_eq!(g.closed_neighbors(0).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(g.closed_neighbors(1).collect::<Vec<_>>(), vec![0, 1, 3]);
        assert_eq!(g.closed_neighbors(2).collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(g.closed_neighbors(3).collect::<Vec<_>>(), vec![1, 3]);
    }
}
