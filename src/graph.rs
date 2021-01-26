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
    /// The graph is internally stored as a full square matrix for each node pair.
    matrix: Vec<T>,
    /// The data structure is treated as very fixed-size at the moment, but on order to efficiently
    /// merge vertices in the algorithm, it is possible to mark vertices as "removed"/not-present.
    /// Such vertices will not be returned as part of any neighbor-sets etc. Accessing the weight
    /// of an edge involving a removed vertex is an error, but only checked in debug mode for
    /// performance reasons.
    /// Any users of this struct that iterate manually over some index range must check themself
    /// whether a vertex is removed, using `.is_present(u)`.
    present_markers: Vec<bool>,
    present_nodes: Vec<usize>,
    adjacency_lists: Vec<Vec<usize>>,
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
            present_markers: vec![true; size],
            present_nodes: (0..size).collect::<Vec<_>>(),
            adjacency_lists: vec![Vec::new(); size],
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
            if self.present_markers[u] {
                map[u] = pg.add_node(imap.map(|m| m[u].clone()).unwrap_or(vec![u]));
            }
        }

        for u in 0..self.size {
            if self.present_markers[u] {
                for v in (u + 1)..self.size {
                    if self.present_markers[v] {
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
        debug_assert!(self.present_markers[u]);
        debug_assert!(self.present_markers[v]);
        assert_ne!(u, v);

        self.matrix[u * self.size + v]
    }

    /// Set the weight associated with pair `(u, v)`.
    /// u and v can be in any order, panics if `u == v`.
    pub fn set(&mut self, u: usize, v: usize, w: T) {
        debug_assert!(self.present_markers[u]);
        debug_assert!(self.present_markers[v]);
        assert_ne!(u, v);

        let prev = self.matrix[u * self.size + v];
        self.matrix[u * self.size + v] = w;
        self.matrix[v * self.size + u] = w;

        if prev <= T::ZERO && w > T::ZERO {
            self.adjacency_lists[u].push(v);
            self.adjacency_lists[v].push(u);
        } else if prev > T::ZERO && w <= T::ZERO {
            let idx_v_in_u = self.adjacency_lists[u]
                .iter()
                .position(|&x| x == v)
                .unwrap();
            self.adjacency_lists[u].swap_remove(idx_v_in_u);

            let idx_u_in_v = self.adjacency_lists[v]
                .iter()
                .position(|&x| x == u)
                .unwrap();
            self.adjacency_lists[v].swap_remove(idx_u_in_v);
        }
    }

    /// Returns an iterator over the open neighborhood of `u` (i.e., not including `u` itself).
    pub fn neighbors(&self, u: usize) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(self.present_markers[u]);

        self.adjacency_lists[u]
            .iter()
            .copied()
            .filter(move |&v| self.present_markers[v])

        //self.adjacency_lists[u].iter().copied()

        /*self.nodes()
        .filter(move |&v| u != v && self.get(u, v) > T::ZERO)*/

        /*(0..u)
        .filter(move |&v| self.present[v] && self.get(v, u) > T::ZERO)
        .chain(
            ((u + 1)..self.size).filter(move |&v| self.present[v] && self.get(u, v) > T::ZERO),
        )*/
    }

    pub fn neighbors_with_weights(&self, u: usize) -> impl Iterator<Item = (usize, T)> + '_ {
        debug_assert!(self.present_markers[u]);
        self.adjacency_lists[u]
            .iter()
            .filter(move |&&v| self.present_markers[v])
            .map(move |&v| (v, self.get(u, v)))
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
        debug_assert!(self.present_markers[u]);
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
        self.present_nodes.len()
        /*(0..self.size)
        .filter(move |&v| self.present_markers[v])
        .count()*/
    }

    /// Returns an iterator over all the nodes present in the graph.
    pub fn nodes(&self) -> impl Iterator<Item = usize> + '_ {
        self.present_nodes.iter().copied()
        //(0..self.size).filter(move |&v| self.present_markers[v])
    }

    pub fn nodes_from(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        self.present_nodes[i..].iter().copied()
    }

    // TODO: This function has some interesting invariants, make sure to document its usage
    // properly.
    pub fn iterate_nodes<F>(&mut self, start: usize, mut f: F)
    where
        F: FnMut(&mut Self, &mut dyn FnMut(&mut Self, usize), usize, usize),
    {
        let mut i = start;
        while i < self.present_nodes.len() {
            let mut set_not_present = move |this: &mut Self, x: usize| {
                let x_idx = this.present_nodes.iter().position(|&u| u == x).unwrap();

                // We only allow setting elements to not present that have already been iterated
                // over (or are the current iteration element). If a past element was removed,
                // swap_remove would move a not-yet-visited element back into that slot and that
                // just results in all kinds of headaches.
                if x_idx < i {
                    panic!("Tried to set an earlier element ({} with idx {}) to not present, currently at idx {}!", x, x_idx, i);
                }

                this.present_markers[x] = false;
                this.present_nodes.swap_remove(x_idx);

                // If this is the current element, make sure to iterate the replacement too.
                // Otherwise, x_idx > i, in which case everything is fine anyway.
                if x_idx == i {
                    i -= 1;
                }
            };

            f(self, &mut set_not_present, i, self.present_nodes[i]);

            i += 1;
        }
    }

    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        debug_assert!(self.present_markers[u]);
        debug_assert!(self.present_markers[v]);
        self.get(u, v) > T::ZERO
    }

    pub fn is_present(&self, u: usize) -> bool {
        self.present_markers[u]
    }

    pub fn set_present(&mut self, u: usize, present: bool) -> usize {
        let prev = self.present_markers[u];
        self.present_markers[u] = present;

        // TODO: If we keep all the weird iteration stuff, this should just become
        // a `set_not_present` function that does not allow the two unimplemented! cases.

        if prev && !present {
            let idx = self.present_nodes.iter().position(|&x| x == u).unwrap();
            self.present_nodes.swap_remove(idx);
            idx
        } else if !prev && present {
            unimplemented!();
        } else {
            unimplemented!();
        }
    }

    /// Splits a graph into its connected components.
    /// An associated index map is also split into equivalent maps for each individual component.
    /// In addition, this also provides a mapping from vertices in this graph to the index of the
    /// component they are a part of.
    /// If the graph consists of only a single component, this returns None instead to avoid extra
    /// allocations and work.
    pub fn split_into_components(
        &self,
        imap: &IndexMap,
    ) -> Option<(Vec<(Self, IndexMap)>, Vec<usize>)> {
        let mut visited = vec![false; self.size];
        let mut stack = Vec::new();
        let mut components = Vec::new();

        let mut current = Vec::new();

        let mut component_map = vec![0; self.size];

        let present_count = self.present_node_count();

        for u in 0..self.size {
            if visited[u] || !self.present_markers[u] {
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

            if current.len() == present_count {
                // Complete graph is this single component.
                return None;
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

        Some((components, component_map))
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
