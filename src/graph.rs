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

#[derive(Debug, Copy, Clone)]
pub enum Op<T: GraphWeight> {
    Set { u: usize, v: usize, prev: T },
    NotPresent(usize),
}

/// An undirected graph without self-loops that stores weights for every pair of vertices (so, for
/// edges and for non-edges).
/// Edges have weights >0, non-edges have weights <0 (zero-edges are also permitted and considered
/// to be non-edges by functions such has `neighbors`).
///
/// Accessing self-loop weights will panic!
#[derive(Clone, Debug)]
pub struct Graph<T: GraphWeight> {
    size: usize,
    /// The graph is internally stored as a full square adjacency matrix.
    matrix: Vec<T>,
    /// The data structure is treated as very fixed-size at the moment, but on order to efficiently
    /// merge vertices in the algorithm, it is possible to mark vertices as "removed"/not-present.
    /// Such vertices will not be returned as part of any neighbor-sets etc. Accessing the weight
    /// of an edge involving a removed vertex is an error, but only checked in debug mode for
    /// performance reasons.
    /// Any users of this struct that iterate manually over some index range must check themself
    /// whether a vertex is removed, using `.is_present(u)`.
    //present: Vec<bool>,
    //present_count: usize,
    oplog: Vec<Op<T>>,
}

// TODO: Docs
#[derive(Clone)]
pub struct GraphViewState {
    present: Vec<bool>,
    present_count: usize,
}

pub struct GraphView<'g, T: GraphWeight> {
    graph: &'g mut Graph<T>,
    state: GraphViewState,
}

impl<T: GraphWeight> Graph<T> {
    /// Creates a new empty (without any edges) graph with `size` vertices, with all weights set to
    /// -1.0.
    pub fn new(size: usize) -> Self {
        assert!(size > 0);
        let mat_size = size * size;

        Graph {
            size,
            matrix: vec![T::NEG_ONE; mat_size],
            //present: vec![true; size],
            //present_count: size,
            oplog: Vec::new(),
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

    /// Set the weight associated with pair `(u, v)`.
    /// u and v can be in any order, panics if `u == v`.
    pub fn set(&mut self, u: usize, v: usize, w: T) {
        assert_ne!(u, v);

        let prev = self.matrix[u * self.size + v];

        self.matrix[u * self.size + v] = w;
        self.matrix[v * self.size + u] = w;

        self.oplog.push(Op::Set { u, v, prev });
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl GraphViewState {
    pub fn realize<'g, T: GraphWeight>(self, graph: &'g mut Graph<T>) -> GraphView<'g, T> {
        GraphView { graph, state: self }
    }
}

impl<'g, T: GraphWeight> GraphView<'g, T> {
    pub fn new(graph: &'g mut Graph<T>) -> Self {
        let present = vec![true; graph.size];
        let present_count = graph.size;

        Self {
            graph,
            state: GraphViewState {
                present,
                present_count,
            },
        }
    }

    pub fn into_state(self) -> GraphViewState {
        self.state
    }

    pub fn into_parts(self) -> (&'g mut Graph<T>, GraphViewState) {
        (self.graph, self.state)
    }

    pub fn cloned_state(&self) -> GraphViewState {
        self.state.clone()
    }

    pub fn cloned_graph(&self) -> Graph<T> {
        self.graph.clone()
    }

    /*pub fn clone_graph(&self) -> Self {
        let new_graph = self.graph.borrow().clone();
        let new_graph = Rc::new(RefCell::new(new_graph));
        Self {
            graph: new_graph,
            present: self.present.clone(),
            present_count: self.present_count,
        }
    }

    pub fn clone_view(&self) -> Self {
        Self {
            graph: Rc::clone(&self.graph),
            present: self.present.clone(),
            present_count: self.present_count,
        }
    }*/

    /// Creates a petgraph graph from this graph and an optional IndexMap.
    pub fn into_petgraph(
        &self,
        imap: Option<&IndexMap>,
    ) -> petgraph::Graph<Vec<usize>, T, petgraph::Undirected> {
        use petgraph::prelude::NodeIndex;

        let mut pg = petgraph::Graph::<_, _, _>::with_capacity(self.graph.size, 0);

        let mut map = vec![NodeIndex::new(0); self.graph.size];
        for u in 0..self.graph.size {
            if self.state.present[u] {
                map[u] = pg.add_node(imap.map(|m| m[u].clone()).unwrap_or(vec![u]));
            }
        }

        for u in 0..self.graph.size {
            if self.state.present[u] {
                for v in (u + 1)..self.graph.size {
                    if self.state.present[v] {
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
        debug_assert!(self.state.present[u]);
        debug_assert!(self.state.present[v]);
        assert_ne!(u, v);

        self.graph.matrix[u * self.graph.size + v]
    }

    /// Set the weight associated with pair `(u, v)`.
    /// u and v can be in any order, panics if `u == v`.
    pub fn set(&mut self, u: usize, v: usize, w: T) {
        debug_assert!(self.state.present[u]);
        debug_assert!(self.state.present[v]);
        assert_ne!(u, v);

        let prev = self.graph.matrix[u * self.graph.size + v];
        let size = self.graph.size;

        self.graph.matrix[u * size + v] = w;
        self.graph.matrix[v * size + u] = w;

        self.graph.oplog.push(Op::Set { u, v, prev });
    }

    /// Returns an iterator over the open neighborhood of `u` (i.e., not including `u` itself).
    pub fn neighbors(&self, u: usize) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(self.state.present[u]);
        self.nodes()
            .filter(move |&v| u != v && self.get(u, v) > T::ZERO)
        /*(0..u)
        .filter(move |&v| self.present[v] && self.get(v, u) > T::ZERO)
        .chain(
            ((u + 1)..self.size).filter(move |&v| self.present[v] && self.get(u, v) > T::ZERO),
        )*/
    }

    pub fn neighbors_with_weights(&self, u: usize) -> impl Iterator<Item = (usize, T)> + '_ {
        debug_assert!(self.state.present[u]);
        self.nodes().filter_map(move |v| {
            if u == v {
                return None;
            }
            let uv = self.get(u, v);
            if uv > T::ZERO {
                Some((v, uv))
            } else {
                None
            }
        })
    }

    /// Returns an iterator over the closed neighborhood of `u` (i.e., including `u` itself).
    pub fn closed_neighbors(&self, u: usize) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(self.state.present[u]);
        self.nodes()
            .filter(move |&v| u == v || self.get(u, v) > T::ZERO)
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
        self.graph.size
    }

    pub fn present_node_count(&self) -> usize {
        self.state.present_count
        //(0..self.size).filter(move |&v| self.present[v]).count()
    }

    pub fn edge_count(&self) -> usize {
        let mut count = 0;
        for u in self.nodes() {
            for v in (u + 1)..self.graph.size {
                continue_if_not_present!(self, v);
                if self.get(u, v) > T::ZERO {
                    count += 1;
                }
            }
        }

        count
    }

    /// Returns an iterator over all the nodes present in the graph.
    pub fn nodes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.graph.size).filter(move |&v| self.state.present[v])
    }

    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        debug_assert!(self.state.present[u]);
        debug_assert!(self.state.present[v]);
        self.get(u, v) > T::ZERO
    }

    pub fn is_present(&self, u: usize) -> bool {
        self.state.present[u]
    }

    // TODO: Clean up oplog handling with GraphView vs Graph

    pub fn set_not_present(&mut self, u: usize) {
        assert!(self.state.present[u]);
        self.state.present[u] = false;
        self.state.present_count -= 1;

        self.graph.oplog.push(Op::NotPresent(u));
    }

    pub fn oplog_len(&self) -> usize {
        self.graph.oplog.len()
    }

    pub fn rollback_to(&mut self, oplog_len: usize) {
        for op in self.graph.oplog.drain(oplog_len..).rev() {
            match op {
                Op::Set { u, v, prev } => {
                    // Don't use `set` here, that would modify the oplog again!
                    self.graph.matrix[u * self.graph.size + v] = prev;
                    self.graph.matrix[v * self.graph.size + u] = prev;
                }
                Op::NotPresent(u) => {
                    self.state.present[u] = true;
                    self.state.present_count += 1;
                }
            }
        }
    }

    /// Splits a graph into its connected components.
    /// An associated index map is also split into equivalent maps for each individual component.
    /// In addition, this also provides a mapping from vertices in this graph to the index of the
    /// component they are a part of.
    pub fn split_into_components(&self) -> (Vec<GraphViewState>, Vec<usize>) {
        let mut visited = vec![false; self.graph.size];
        let mut stack = Vec::new();
        let mut components = Vec::new();

        let mut current_present = vec![false; self.graph.size];
        let mut current_size = 0;

        let mut component_map = vec![0; self.graph.size];

        for u in 0..self.graph.size {
            if visited[u] || !self.state.present[u] {
                continue;
            }

            stack.push(u);

            while let Some(v) = stack.pop() {
                if visited[v] {
                    continue;
                }

                visited[v] = true;
                current_present[v] = true;
                current_size += 1;
                component_map[v] = components.len();

                for n in self.neighbors(v) {
                    if !visited[n] {
                        stack.push(n);
                    }
                }
            }

            let comp = GraphViewState {
                present: current_present,
                present_count: current_size,
            };
            components.push(comp);
            current_present = vec![false; self.graph.size];
        }

        (components, component_map)
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
        let g = GraphView::new(&mut example_graph());

        assert_eq!(g.neighbors(0).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(g.neighbors(1).collect::<Vec<_>>(), vec![0, 3]);
        assert_eq!(g.neighbors(2).collect::<Vec<_>>(), vec![0]);
        assert_eq!(g.neighbors(3).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn closed_neighbors() {
        let g = GraphView::new(&mut example_graph());

        assert_eq!(g.closed_neighbors(0).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(g.closed_neighbors(1).collect::<Vec<_>>(), vec![0, 1, 3]);
        assert_eq!(g.closed_neighbors(2).collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(g.closed_neighbors(3).collect::<Vec<_>>(), vec![1, 3]);
    }
}
