use lifeguard::Pool;

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
pub struct Graph<'a, T: GraphWeight> {
    size: usize,
    /// The graph is internally stored as a full square matrix for each node pair.
    matrix: Vec<T>,
    /// The data structure is treated as very fixed-size at the moment, but on order to efficiently
    /// merge vertices in the algorithm, it is possible to mark vertices as "removed"/not-present.
    /// Such vertices will not be returned as part of any neighbor-sets etc. Accessing the weight
    /// of an edge involving a removed vertex is an error, but only checked in debug mode for
    /// performance reasons.
    /// Any users of this struct that iterate manually over some index range must check themself
    /// whether a vertex is removed, using `.is_present(u)`.
    present: Vec<bool>,
    present_count: usize,
    adjacency_lists: Vec<Vec<usize>>,
    pub adj_pool: &'a Pool<Vec<usize>>,
}

impl<'a, T: GraphWeight> Drop for Graph<'a, T> {
    fn drop(&mut self) {
        for adj in self.adjacency_lists.drain(..) {
            self.adj_pool.attach(adj);
        }
    }
}

impl<'a, T: GraphWeight + std::fmt::Display> Graph<'a, T> {
    /// Creates a new empty (without any edges) graph with `size` vertices, with all weights set to
    /// -1.0.
    pub fn new(size: usize, adj_pool: &'a Pool<Vec<usize>>) -> Self {
        let mat_size = size * size;

        let adjacency_lists = (0..size).map(|_| adj_pool.new().detach()).collect();
        //let adjacency_lists = vec![];

        Graph {
            size,
            matrix: vec![T::NEG_ONE; mat_size],
            present: vec![true; size],
            present_count: size,
            adjacency_lists,
            adj_pool,
        }
    }

    pub fn fork(&self) -> Self {
        let adjacency_lists = (0..self.size)
            .map(|u| {
                let mut adj = self.adj_pool.new().detach();
                adj.extend_from_slice(&self.adjacency_lists[u]);
                adj
            })
            .collect();
        //let adjacency_lists = vec![];

        Graph {
            size: self.size,
            matrix: self.matrix.clone(),
            present: self.present.clone(),
            present_count: self.present_count,
            adjacency_lists,
            adj_pool: self.adj_pool,
        }
    }

    /// Creates a new graph from an existing petgraph graph. Edges existing in the input graph are
    /// given weight 1, non-edges are given weight -1.
    /// A corresponding `IndexMap` is created that maps vertex indices from the returned graph to
    /// the weights associated with the vertices in the petgraph graph.
    pub fn new_from_petgraph(
        pg: &crate::PetGraph,
        adj_pool: &'a Pool<Vec<usize>>,
    ) -> (Self, IndexMap) {
        let mut g = Self::new(pg.node_count(), adj_pool);
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
                        let uv = self.get(u, v);
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

        self.matrix[u * self.size + v]
    }

    /// Set the weight associated with pair `(u, v)`.
    /// u and v can be in any order, panics if `u == v`.
    pub fn set(&mut self, u: usize, v: usize, w: T) {
        debug_assert!(self.present[u]);
        debug_assert!(self.present[v]);
        assert_ne!(u, v);

        let prev = self.matrix[u * self.size + v];
        self.matrix[u * self.size + v] = w;
        self.matrix[v * self.size + u] = w;

        if prev <= T::ZERO && w > T::ZERO {
            self.adjacency_lists[u].push(v);
            self.adjacency_lists[v].push(u);
        } else if prev > T::ZERO && w <= T::ZERO {
            let idx_in_u = self.adjacency_lists[u]
                .iter()
                .position(|&x| x == v)
                .unwrap_or_else(|| {
                    panic!(
                        "Failed finding idx_in_u, in set({}, {}, {}) (was {}), adj list of u is {:?}",
                        u, v, w, prev, self.adjacency_lists[u]
                    )
                });
            self.adjacency_lists[u].swap_remove(idx_in_u);

            let idx_in_v = self.adjacency_lists[v]
                .iter()
                .position(|&x| x == u)
                .unwrap();
            self.adjacency_lists[v].swap_remove(idx_in_v);
        }
    }

    /// Returns an iterator over the open neighborhood of `u` (i.e., not including `u` itself).
    pub fn neighbors(&self, u: usize) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(self.present[u]);
        self.adjacency_lists[u].iter().copied()

        /*self.nodes()
        .filter(move |&v| u != v && self.get(u, v) > T::ZERO)*/

        /*(0..u)
        .filter(move |&v| self.present[v] && self.get(v, u) > T::ZERO)
        .chain(
            ((u + 1)..self.size).filter(move |&v| self.present[v] && self.get(u, v) > T::ZERO),
        )*/
    }

    pub fn neighbors_with_weights(&self, u: usize) -> impl Iterator<Item = (usize, T)> + '_ {
        debug_assert!(self.present[u]);
        self.adjacency_lists[u]
            .iter()
            .map(move |&x| (x, self.get(u, x)))

        /*self.nodes().filter_map(move |v| {
            if u == v {
                return None;
            }
            let uv = self.get(u, v);
            if uv > T::ZERO {
                Some((v, uv))
            } else {
                None
            }
        })*/
    }

    /// Returns an iterator over the closed neighborhood of `u` (i.e., including `u` itself).
    pub fn closed_neighbors(&self, u: usize) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(self.present[u]);

        self.neighbors(u).chain(std::iter::once(u))

        /*self.nodes()
        .filter(move |&v| u == v || self.get(u, v) > T::ZERO)*/

        /*(0..u)
        .filter(move |&v| self.present[v] && self.get(v, u) > T::ZERO)
        .chain(std::iter::once(u))
        .chain(
            ((u + 1)..self.size).filter(move |&v| self.present[v] && self.get(u, v) > T::ZERO),
        )*/
    }

    /// Returns the size of the graph.
    /// Note that this also includes nodes marked as "not present".
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn present_node_count(&self) -> usize {
        self.present_count
        //(0..self.size).filter(move |&v| self.present[v]).count()
    }

    /// Returns an iterator over all the nodes present in the graph.
    // TODO: Is it worth making this an ExactSizeIterator ?
    pub fn nodes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.size).filter(move |&v| self.present[v])
    }

    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        debug_assert!(self.present[u]);
        debug_assert!(self.present[v]);
        self.get(u, v) > T::ZERO
    }

    pub fn is_present(&self, u: usize) -> bool {
        self.present[u]
    }

    pub fn set_not_present(&mut self, u: usize) {
        assert!(self.present[u]);
        self.present[u] = false;
        self.present_count -= 1;

        for i in 0..self.adjacency_lists[u].len() {
            let v = self.adjacency_lists[u][i];
            let idx_in_v = self.adjacency_lists[v]
                .iter()
                .position(|&x| x == u)
                .unwrap();
            self.adjacency_lists[v].swap_remove(idx_in_v);
        }
    }

    pub fn collapse(mut self, mut imap: IndexMap) -> (Self, IndexMap) {
        if self.size == self.present_count {
            return (self, imap);
        }

        if self.present_count == 0 {
            return (Self::new(0, self.adj_pool), IndexMap::new(0));
        }

        let mut first_empty = 0;
        // Find first empty/not-present slot.
        // By the first if above, we know there is one, so this won't go out of bounds.
        while self.present[first_empty] {
            first_empty += 1;
        }

        let mut last_present = self.size - 1;
        // Find last slot that is still present.
        // By the second if above, we know there is one, so this won't go past 0.
        while !self.present[last_present] {
            last_present -= 1;
        }

        while first_empty < last_present {
            // TODO: I wonder if the weight copying here and the one below can be combined into one pass?

            // 1. Swap last_present vertex into slot first_empty.

            // 1.1. Update adjacency lists of every neighbor of last_present.
            for i in 0..self.adjacency_lists[last_present].len() {
                let v = self.adjacency_lists[last_present][i];
                let idx_in_v = self.adjacency_lists[v]
                    .iter()
                    .position(|&x| x == last_present)
                    .unwrap();
                self.adjacency_lists[v][idx_in_v] = first_empty;
            }
            // 1.2. Move adjacency list of last_present itself.
            self.adjacency_lists.swap(first_empty, last_present);
            // 1.3. Move imap of last_present.
            let im = imap.take(last_present);
            imap.set(first_empty, im);
            // 1.4. Update present list.
            self.present[first_empty] = true;
            self.present[last_present] = false;
            // 1.5. Update weights to all other present nodes
            for v in 0..last_present {
                self.matrix[v * self.size + first_empty] =
                    self.matrix[v * self.size + last_present];
                self.matrix[first_empty * self.size + v] =
                    self.matrix[last_present * self.size + v];
            }

            // 2. Find new first_empty and last_present.
            while first_empty < self.size && self.present[first_empty] {
                first_empty += 1;
            }
            while last_present > 0 && !self.present[last_present] {
                last_present -= 1;
            }
        }

        // TODO: Try that weird matrix layout idea to reduce the work done here.

        // Entries are now arranged as desired, just truncate the storage.
        let new_size = last_present + 1;

        for y in 0..new_size {
            for x in 0..new_size {
                self.matrix[y * new_size + x] = self.matrix[y * self.size + x];
            }
        }

        self.size = new_size;
        self.matrix.truncate(self.size * self.size);
        self.present.truncate(self.size);
        imap.map.truncate(self.size);

        for adj in self.adjacency_lists.drain(self.size..) {
            self.adj_pool.attach(adj);
        }

        return (self, imap);
    }

    /// Splits a graph into its connected components.
    /// An associated index map is also split into equivalent maps for each individual component.
    /// In addition, this also provides a mapping from vertices in this graph to the index of the
    /// component they are a part of.
    pub fn split_into_components(
        &self,
        imap: &IndexMap,
    ) -> Option<(Vec<(Self, IndexMap)>, Vec<usize>)> {
        //) -> (Vec<(Self, IndexMap)>, Vec<usize>) {
        let mut visited = vec![false; self.size];
        let mut stack = Vec::new();
        let mut components = Vec::new();

        let mut current = Vec::new();

        let mut component_map = vec![0; self.size];

        let mut would_have_been_none = false;

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

            if current.len() == self.present_count {
                would_have_been_none = true;
            }

            if current.len() == self.present_count {
                return None;
            }

            let mut comp = Self::new(current.len(), self.adj_pool);
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

        assert_eq!(would_have_been_none, components.len() == 1);

        Some((components, component_map))
        //(components, component_map)
    }
}

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

    fn example_graph(adj_pool: &Pool<Vec<usize>>) -> Graph<i32> {
        //       1
        //   0 ----- 1
        //   |       |
        // 5 |       | 3
        //   |       |
        //   2       3
        //       -2

        let mut g = Graph::new(4, adj_pool);
        g.set(0, 1, 1);
        g.set(0, 2, 5);
        g.set(1, 3, 3);
        g.set(2, 3, -2);
        g
    }

    #[test]
    fn neighbors() {
        let p = Pool::with_size(4);
        let g = example_graph(&p);

        assert_eq!(g.neighbors(0).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(g.neighbors(1).collect::<Vec<_>>(), vec![0, 3]);
        assert_eq!(g.neighbors(2).collect::<Vec<_>>(), vec![0]);
        assert_eq!(g.neighbors(3).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn closed_neighbors() {
        let p = Pool::with_size(4);
        let g = example_graph(&p);

        assert_eq!(g.closed_neighbors(0).collect::<Vec<_>>(), vec![1, 2, 0]);
        assert_eq!(g.closed_neighbors(1).collect::<Vec<_>>(), vec![0, 3, 1]);
        assert_eq!(g.closed_neighbors(2).collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(g.closed_neighbors(3).collect::<Vec<_>>(), vec![1, 3]);
    }

    /*#[test]
    fn collapse() {
        let _ = env_logger::builder().is_test(true).try_init();

        let p = Pool::with_size(4);
        let mut g = example_graph(&p);
        let size = g.size();

        g.set_not_present(1);
        let (g, _) = g.collapse(IndexMap::identity(size));

        assert_eq!(g.neighbors(0).collect::<Vec<_>>(), vec![2]);
        assert_eq!(g.neighbors(1).collect::<Vec<_>>(), vec![]);
        assert_eq!(g.neighbors(2).collect::<Vec<_>>(), vec![0]);

        assert_eq!(g.get(0, 1), -1);
        assert_eq!(g.get(2, 1), -2);
        assert_eq!(g.get(0, 2), 5);
    }*/
}
