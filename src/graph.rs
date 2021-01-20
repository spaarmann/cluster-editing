pub trait GraphWeight: PartialOrd + Copy {
    const ZERO: Self;
    const ONE: Self;
    const NEG_ONE: Self;
    fn is_zero(self) -> bool;
}

impl GraphWeight for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NEG_ONE: Self = -1.0;
    fn is_zero(self) -> bool {
        self.abs() < 0.001
    }
}

impl GraphWeight for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const NEG_ONE: Self = -1;
    fn is_zero(self) -> bool {
        self == 0
    }
}

/// An undirected graph without self-loops that stores weights for every pair of vertices (so, for
/// edges and for non-edges).
/// Edges have weights >0, non-edges have weights <0.
///
/// Accessing self-loop weights will panic!
#[derive(Clone, Debug)]
pub struct Graph<T: GraphWeight> {
    size: usize,
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
    matrix: Vec<T>,
    /// The data structure is treated as very fixed-size at the moment, but on order to efficiently
    /// merge vertices in the algorithm, it is possible to mark vertices as "removed"/not-present.
    /// Such vertices will not be returned as part of any neighbor-sets etc. Accessing the weight
    /// of an edge involving a removed vertex is an error, but only checked in debug mode for
    /// performance reasons.
    /// Any users of this struct that iterate manually over some index range must check themself
    /// whether a vertex is removed, using `.is_present(u)`.
    present: Vec<bool>,
    // Stores a mapping from row index to the starting index of that row in the `matrix` list.
    //row_offsets: Vec<usize>,
}

impl<T: GraphWeight> Graph<T> {
    /// Creates a new empty (without any edges) graph with `size` vertices, with all weights set to
    /// -1.0.
    pub fn new(size: usize) -> Self {
        assert!(size > 0);
        let mat_size = size * size; //(size * (size - 1) / 2) as usize;

        /*let row_offsets =
        // Row 0 does not exist, so use a marker value that will definitely panic if any code
        // tries to index using it.
        std::iter::once(usize::MAX)
        // For all other rows, calculate the correct offset.
        .chain((1..size).map(|i| i * (i - 1) / 2))
        .collect();*/
        Graph {
            size,
            matrix: vec![T::NEG_ONE; mat_size],
            present: vec![true; size],
            //row_offsets,
        }
    }

    /// Creates a new graph from an existing petgraph graph. Edges existing in the input graph are
    /// given weight 1, non-edges are given weight -1.
    /// A corresponding `IndexMap` is created that maps vertex indices from the returned graph to
    /// the weights associated with the vertices in the petgraph graph.
    pub fn new_from_petgraph(pg: &crate::PetGraph) -> (Self, IndexMap) {
        let mut g = Self::new(pg.node_count());
        let mut imap = IndexMap::new(pg.node_count());

        for v in pg.node_indices() {
            imap[v.index()] = vec![*pg.node_weight(v).unwrap()];
        }

        for e in pg.edge_indices() {
            let (u, v) = pg.edge_endpoints(e).unwrap();
            g.set(u.index(), v.index(), T::ONE);
        }

        (g, imap)
    }

    /// Creates a petgraph graph from this graph and an optional IndexMap.
    pub fn into_petgraph(
        &self,
        imap: Option<&IndexMap>,
    ) -> petgraph::Graph<Vec<usize>, T, petgraph::Undirected> {
        use petgraph::prelude::NodeIndex;

        let mut pg = petgraph::Graph::<_, _, _>::with_capacity(self.size, 0);

        let mut map = vec![NodeIndex::new(0); self.size];
        for u in 0..self.size {
            if self.present[u] {
                map[u] = pg.add_node(imap.map(|m| m[u].clone()).unwrap_or(vec![u]));
            }
        }

        for u in 0..self.size {
            if self.present[u] {
                for v in (u + 1)..self.size {
                    if self.present[v] {
                        let uv = self.get_direct(u, v);
                        if uv > T::ZERO {
                            pg.add_edge(map[u], map[v], uv);
                        }
                    }
                }
            }
        }

        pg
    }

    /// Get the weight associated with pair `(u, v)`.
    /// u and v can be in any order, panics if `u == v`.
    pub fn get(&self, u: usize, v: usize) -> T {
        debug_assert!(self.present[u]);
        debug_assert!(self.present[v]);
        assert_ne!(u, v);

        /*if u < v {
            self.matrix[self.row_offsets[v] + u]
        } else {
            self.matrix[self.row_offsets[u] + v]
        }*/
        self.matrix[u + self.size * v]
    }

    /// Get the weight associated with pair `(u, v)`.
    /// u and v can be in any order, panics if `u == v`.
    pub fn get_ref(&self, u: usize, v: usize) -> &T {
        debug_assert!(self.present[u]);
        debug_assert!(self.present[v]);
        assert_ne!(u, v);

        /*if u < v {
            &self.matrix[self.row_offsets[v] + u]
        } else {
            &self.matrix[self.row_offsets[u] + v]
        }*/
        &self.matrix[u + self.size * v]
    }

    /// Get a mutable reference to the weight associated with pair `(u, v)`.
    /// u and v can be in any order, panics if `u == v`.
    /*pub fn get_mut(&mut self, u: usize, v: usize) -> &mut T {
        debug_assert!(self.present[u]);
        debug_assert!(self.present[v]);
        assert_ne!(u, v);

        if u < v {
            &mut self.matrix[self.row_offsets[v] + u]
        } else {
            &mut self.matrix[self.row_offsets[u] + v]
        }
    }*/

    /// Set the weight associated with pair `(u, v)`.
    /// u and v can be in any order, panics if `u == v`.
    pub fn set(&mut self, u: usize, v: usize, w: T) {
        debug_assert!(self.present[u]);
        debug_assert!(self.present[v]);
        assert_ne!(u, v);

        /*if u < v {
            self.matrix[self.row_offsets[v] + u] = w;
        } else {
            self.matrix[self.row_offsets[u] + v] = w;
        }*/
        self.matrix[u + self.size * v] = w;
        self.matrix[v + self.size * u] = w;
    }

    /// Like `get`, but assumes `u != v` and `u < v` instead of checking both.
    pub fn get_direct(&self, u: usize, v: usize) -> T {
        debug_assert!(self.present[u]);
        debug_assert!(self.present[v]);
        //self.matrix[self.row_offsets[v] + u]
        self.matrix[u + self.size * v]
    }
    /// Like `get_ref`, but assumes `u != v` and `u < v` instead of checking both.
    pub fn get_ref_direct(&self, u: usize, v: usize) -> &T {
        debug_assert!(self.present[u]);
        debug_assert!(self.present[v]);
        //&self.matrix[self.row_offsets[v] + u]
        &self.matrix[u + self.size * v]
    }
    /// Like `get_mut`, but assumes `u != v` and `u < v` instead of checking both.
    /*pub fn get_mut_direct(&mut self, u: usize, v: usize) -> &mut T {
        debug_assert!(self.present[u]);
        debug_assert!(self.present[v]);
        &mut self.matrix[self.row_offsets[v] + u]
    }*/
    /// Like `set`, but assumes `u != v` and `u < v` instead of checking both.
    pub fn set_direct(&mut self, u: usize, v: usize, w: T) {
        debug_assert!(self.present[u]);
        debug_assert!(self.present[v]);
        //self.matrix[self.row_offsets[v] + u] = w;
        self.matrix[u + self.size * v] = w;
        self.matrix[v + self.size * u] = w;
    }

    /// Returns an iterator over the open neighborhood of `u` (i.e., not including `u` itself). The
    /// neighbors are guaranteed to be in ascending order.
    pub fn neighbors(&self, u: usize) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(self.present[u]);
        (0..u)
            .filter(move |&v| self.present[v] && self.get_direct(v, u) > T::ZERO)
            .chain(
                ((u + 1)..self.size)
                    .filter(move |&v| self.present[v] && self.get_direct(u, v) > T::ZERO),
            )
    }

    /// Returns an iterator over the closed neighborhood of `u` (i.e., including `u` itself). The
    /// neighbors are guaranteed to be in ascending order.
    pub fn closed_neighbors(&self, u: usize) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(self.present[u]);
        (0..u)
            .filter(move |&v| self.present[v] && self.get_direct(v, u) > T::ZERO)
            .chain(std::iter::once(u))
            .chain(
                ((u + 1)..self.size)
                    .filter(move |&v| self.present[v] && self.get_direct(u, v) > T::ZERO),
            )
    }

    /// Returns the size of the graph.
    /// Note that this also includes nodes marked as "not present".
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn present_node_count(&self) -> usize {
        (0..self.size).filter(move |&v| self.present[v]).count()
    }

    /// Returns an iterator over all the nodes present in the graph.
    pub fn nodes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.size).filter(move |&v| self.present[v])
    }

    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        debug_assert!(self.present[u]);
        debug_assert!(self.present[v]);
        self.get(u, v) > T::ZERO
    }

    pub fn has_edge_direct(&self, u: usize, v: usize) -> bool {
        debug_assert!(self.present[u]);
        debug_assert!(self.present[v]);
        self.get_direct(u, v) > T::ZERO
    }

    pub fn is_present(&self, u: usize) -> bool {
        self.present[u]
    }

    pub fn set_present(&mut self, u: usize, present: bool) {
        self.present[u] = present;
    }

    /// Splits a graph into its connected components.
    /// An associated index map is also split into equivalent maps for each individual component.
    /// In addition, this also provides a mapping from vertices in this graph to the index of the
    /// component they are a part of.
    pub fn split_into_components(&self, imap: &IndexMap) -> (Vec<(Self, IndexMap)>, Vec<usize>) {
        let mut visited = vec![false; self.size];
        let mut stack = Vec::new();
        let mut components = Vec::new();

        let mut current = Vec::new();

        let mut component_map = vec![0; self.size];

        for u in 0..self.size {
            if visited[u] || !self.present[u] {
                continue;
            }

            stack.push(u);

            while let Some(v) = stack.pop() {
                if visited[v] {
                    continue;
                }

                visited[v] = true;
                current.push(v);

                for n in self.neighbors(v) {
                    if !visited[n] {
                        stack.push(n);
                    }
                }
            }

            let mut comp = Self::new(current.len());
            let mut comp_imap = IndexMap::new(current.len());
            for i in 0..current.len() {
                let v = current[i];
                comp_imap[i] = imap[v].clone();
                component_map[v] = components.len();

                for j in 0..i {
                    let w = current[j];

                    comp.set(i, j, self.get(v, w));
                }
            }

            components.push((comp, comp_imap));
            current.clear();
        }

        (components, component_map)
    }
}

impl<T: GraphWeight> std::ops::Index<(usize, usize)> for Graph<T> {
    type Output = T;
    /// Semantics equivalent to `Graph::get_ref`.
    fn index(&self, (u, v): (usize, usize)) -> &Self::Output {
        self.get_ref(u, v)
    }
}

/*impl<T: GraphWeight> std::ops::IndexMut<(usize, usize)> for Graph<T> {
    /// Semantics equivalent to `Graph::get_mut`.
    fn index_mut(&mut self, (u, v): (usize, usize)) -> &mut Self::Output {
        self.get_mut(u, v)
    }
}*/

/// Companion to the `Graph` struct for remapping to different indices.
///
/// The `Graph` struct uses indices `0..size` for the vertices stored within.
/// This is fine (and even advantageous) for operating on a single `Graph` instance,
/// but in some contexts it is necessary to e.g. map those indices to those of the original input
/// graph. An `IndexMap` can store such a mapping.
///
/// Most of the time, users of the struct and the struct itself may assume each entry is
/// `Some(vec)`, otherwise e.g. the indexers will just panic.
/// Note that this is even true for the mutable indexer (since it gives a reference to the `Vec`
/// inside). To overwrite a `None` value, use `set`.
///
/// It is however also possible to temporarily have `None` entries, for performance reasons. In
/// these cases the caller must make sure the map is ultimately left in a valid state before being
/// passed on.
#[derive(Clone, Debug)]
pub struct IndexMap {
    map: Vec<Option<Vec<usize>>>,
}

impl IndexMap {
    /// Creates an a default map, mapping every index to 0.
    pub fn new(size: usize) -> Self {
        Self {
            map: vec![Some(vec![0]); size],
        }
    }

    /// Creates an identity map.
    pub fn identity(size: usize) -> Self {
        Self {
            map: (0..size).map(|i| Some(vec![i])).collect(),
        }
    }

    /// Creates an empty map, where each entry is `None`. *Must* be initialized fully before being
    /// passed to any code that assumes indexing is valid etc.
    pub fn empty(size: usize) -> Self {
        Self {
            map: vec![None; size],
        }
    }

    /// Takes an entry out of the map, leaving `None`. Should only be used if the corresponding
    /// entry will never be accessed again, or if it is immediately replaced by a new valid entry.
    pub fn take(&mut self, i: usize) -> Vec<usize> {
        self.map[i].take().unwrap()
    }

    /// Directly overwrites an entry. Use this to replace a `None` entry with a new valid entry.  
    pub fn set(&mut self, i: usize, entry: Vec<usize>) {
        self.map[i] = Some(entry);
    }
}

impl std::ops::Index<usize> for IndexMap {
    type Output = Vec<usize>;
    fn index(&self, index: usize) -> &Self::Output {
        self.map[index].as_ref().unwrap()
    }
}

impl std::ops::IndexMut<usize> for IndexMap {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.map[index].as_mut().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_graph() -> Graph<i32> {
        // 0 -- 1
        // |    |
        // 2    3

        let mut g = Graph::new(4);
        g.set(0, 1, 1);
        g.set(0, 2, 1);
        g.set(1, 3, 1);
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
