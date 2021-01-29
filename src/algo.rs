use crate::{
    graph::{GraphWeight, IndexMap},
    reduction,
    util::InfiniteNum,
    Graph, PetGraph, Weight,
};

use log::info;
use petgraph::graph::NodeIndex;

#[derive(Copy, Clone, Debug)]
pub enum Edit {
    Insert(usize, usize),
    Delete(usize, usize),
}

impl Edit {
    pub fn insert(edits: &mut Vec<Edit>, imap: &IndexMap, u: usize, v: usize) {
        for &orig_u in &imap[u] {
            for &orig_v in &imap[v] {
                edits.push(Self::Insert(orig_u, orig_v));
            }
        }
    }

    pub fn delete(edits: &mut Vec<Edit>, imap: &IndexMap, u: usize, v: usize) {
        for &orig_u in &imap[u] {
            for &orig_v in &imap[v] {
                edits.push(Self::Delete(orig_u, orig_v));
            }
        }
    }
}

#[derive(Debug)]
pub struct Parameters {
    pub full_reduction_interval: i32,
}

pub fn execute_algorithm(graph: &PetGraph, params: Parameters) -> PetGraph {
    let mut result = graph.clone();
    let (g, imap) = Graph::<Weight>::new_from_petgraph(&graph);
    let components = g
        .split_into_components(&imap)
        .map(|(c, _)| c)
        .unwrap_or_else(|| vec![(g, imap)]);

    info!(
        "Decomposed input graph into {} components",
        components.len()
    );

    for (i, c) in components.into_iter().enumerate() {
        let (cg, imap) = c;
        info!("Solving component {}...", i);
        let (k, edits) = find_optimal_cluster_editing(&cg, &imap, &params);

        info!(
            "Found a cluster editing with k={} and {} edits for component {}: {:?}",
            k,
            edits.len(),
            i,
            edits
        );

        for edit in edits {
            match edit {
                Edit::Insert(u, v) => {
                    if let None = result.find_edge(NodeIndex::new(u), NodeIndex::new(v)) {
                        result.add_edge(NodeIndex::new(u), NodeIndex::new(v), 0);
                    }
                }
                Edit::Delete(u, v) => {
                    if let Some(e) = result.find_edge(NodeIndex::new(u), NodeIndex::new(v)) {
                        result.remove_edge(e);
                    }
                }
            };
        }
    }

    // The algorithm can produce "overlapping" edits. It might e.g. have a "delete(uv)"
    // edit followed later by an "insert(uv)" edit.
    // Find and print the actual diff of the graphs, in terms of the vertex indices of the original input.
    let mut edits = Vec::new();
    for u in graph.node_indices() {
        for v in graph.node_indices() {
            if u == v {
                continue;
            }
            if v > u {
                continue;
            }

            let original = graph.find_edge(u, v);
            let new = result.find_edge(u, v);

            match (original, new) {
                (None, Some(_)) => edits.push(Edit::Insert(
                    *graph.node_weight(u).unwrap(),
                    *graph.node_weight(v).unwrap(),
                )),
                (Some(_), None) => edits.push(Edit::Delete(
                    *graph.node_weight(u).unwrap(),
                    *graph.node_weight(v).unwrap(),
                )),
                _ => { /* no edit */ }
            }
        }
    }

    info!(
        "Final set of {} de-duplicated edits: {:?}",
        edits.len(),
        edits
    );
    result
}

// The imap is used to always have a mapping from the current indices used by the graph to
// what indices those vertices have in the original graph.
// The algorithm works on reduced/modified graphs in parts, but when editing those we want
// to create `Edit` values that are usable on the original graph; we can create those by
// using the imap.
pub fn find_optimal_cluster_editing(
    g: &Graph<Weight>,
    imap: &IndexMap,
    params: &Parameters,
) -> (i32, Vec<Edit>) {
    // TODO: Not sure if executing the algo once with k = 0 is the best
    // way of handling already-disjoint-clique-components.

    let original_node_count = g.size();
    info!(
        "Computing optimal solution for graph with {} nodes.",
        original_node_count
    );

    let mut _path_debugs = String::new();
    let mut instance = ProblemInstance {
        params,
        g: g.clone(),
        imap: imap.clone(),
        k: 0.0,
        k_max: 0.0,
        full_reduction_counter: 0,
        edits: Vec::new(),
        path_log: String::new(),
    };
    let k_start = reduction::initial_param_independent_reduction(&mut instance);

    info!(
        "Reduced graph from {} nodes to {} nodes using parameter-independent reduction.",
        original_node_count,
        instance.g.size()
    );

    let mut k = k_start;
    loop {
        info!("[driver] Starting search with k={}...", k);

        let mut instance = instance.fork_new_branch();
        instance.k = k;
        if let Some(instance) = instance.find_cluster_editing() {
            if !instance.path_log.is_empty() {
                log::info!("Final path debug log:\n{}", instance.path_log);
            }

            return (instance.k as i32, instance.edits);
        }

        k += 1.0;
    }
}

#[derive(Clone, Debug)]
pub struct ProblemInstance<'a> {
    pub params: &'a Parameters,
    pub g: Graph<Weight>,
    pub imap: IndexMap,
    pub k: f32,
    pub k_max: f32,
    pub full_reduction_counter: i32,
    pub edits: Vec<Edit>,
    pub path_log: String,
}

impl<'a> ProblemInstance<'a> {
    fn fork_new_branch(&self) -> Self {
        self.clone()
    }

    /// Tries to find a solution of size <= k for this problem instance.
    // `k` is stored as float because it needs to be compared with and changed by values from
    // the WeightMap a lot, which are floats.
    fn find_cluster_editing(mut self) -> Option<Self> {
        // If k is already 0, we can only if we currently have a solution; there is no point in trying
        // to do further reductions or splitting as we can't afford any edits anyway.

        if self.k > 0.0 {
            if let Some((components, component_map)) = self.g.split_into_components(&self.imap) {
                //if components.len() > 1 {
                // If a connected component decomposes into two components, we calculate
                // the optimum solution for these components separately.
                // TODO: Still not entirely convinced why this is actually *correct*.

                let _k_start = self.k;
                for (_i, (comp, comp_imap)) in components.into_iter().enumerate() {
                    dbg_trace_indent!(
                        self,
                        _k_start,
                        "Starting component {}, remaining k is {}",
                        _i,
                        self.k
                    );
                    self.path_log.push_str(&format!(
                        "Starting component {}, remaining k is {}\n",
                        _i, self.k
                    ));

                    let comp_instance = ProblemInstance {
                        params: self.params,
                        g: comp,
                        imap: comp_imap,
                        k: self.k,
                        k_max: self.k_max,
                        full_reduction_counter: self.full_reduction_counter,
                        edits: self.edits,
                        path_log: self.path_log,
                    };

                    // returns early if we can't even find a solution for the component,
                    // otherwise take the remaining k and proceed to the next component.
                    match comp_instance.find_cluster_editing() {
                        Some(comp_instance) => {
                            self.k = comp_instance.k;
                            self.edits = comp_instance.edits;
                            self.path_log = comp_instance.path_log;
                        }
                        None => {
                            dbg_trace_indent!(
                                self,
                                _k_start,
                                "Finished component {} with 'no solution found', returning.",
                                _i
                            );
                            return None;
                        }
                    }

                    dbg_trace_indent!(
                        self,
                        _k_start,
                        "Finished component {}, remaining k is {}",
                        _i,
                        self.k
                    );
                    self.path_log.push_str(&format!(
                        "Finished component {}, remaining k is {}\n",
                        _i, self.k
                    ));
                }

                // We still need to "cash in" any zero-edges that connect the different components.
                let mut zero_count = 0.0;
                for u in 0..self.g.size() {
                    continue_if_not_present!(self.g, u);
                    for v in (u + 1)..self.g.size() {
                        continue_if_not_present!(self.g, v);
                        if component_map[u] != component_map[v] {
                            if self.g.get(u, v) == Weight::ZERO {
                                zero_count += 1.0;
                            }
                        }
                    }
                }

                self.k -= zero_count / 2.0;

                dbg_trace_indent!(
                    self,
                    _k_start,
                    "After component split, cashed zero-edges, k now {}",
                    self.k
                );

                if self.k >= 0.0 {
                    return Some(self);
                } else {
                    return None;
                }
                //}
            }
        }

        /*if self.g.size() < 5 {
            log::warn!("g too small");
        } else if self.g.is_present(3) {
            log::warn!(
                "neighbors of 3 before collapse: {:?}, (3,4) = {}, 4 present {}",
                self.g.neighbors(3).collect::<Vec<_>>(),
                self.g.get(3, 4),
                self.g.is_present(4)
            );
        } else {
            log::warn!("3 not present before collapse");
        }*/

        let collapsed = self.g.collapse(self.imap);
        self.g = collapsed.0;
        self.imap = collapsed.1;

        /*if self.g.size() < 5 {
            log::warn!("g too small");
        } else if self.g.is_present(3) {
            log::warn!(
                "neighbors of 3 after collapse: {:?}, (3,4) = {}, 4 present {}",
                self.g.neighbors(3).collect::<Vec<_>>(),
                self.g.get(3, 4),
                self.g.is_present(4)
            );
        } else {
            log::warn!("3 not present after collapse");
        }*/

        dbg_trace_indent!(self, self.k, "Performing reduction");
        let _k_start = self.k;

        if self.full_reduction_counter == 0 {
            reduction::full_param_independent_reduction(&mut self);
            self.full_reduction_counter = self.params.full_reduction_interval;
        } else {
            reduction::fast_param_independent_reduction(&mut self);
            self.full_reduction_counter -= 1;
        }

        dbg_trace_indent!(
            self,
            _k_start,
            "Reduced from k={} to k={}",
            _k_start,
            self.k
        );
        self.path_log
            .push_str(&format!("Reduced from k={} to k={}\n", _k_start, self.k));

        if self.k < 0.0 {
            return None;
        }

        dbg_trace_indent!(self, _k_start, "Searching triple");

        // Search for a conflict triple
        // TODO: Surely this can be done a little smarter?
        let mut triple = None;
        'outer: for u in self.g.nodes() {
            for v in self.g.nodes() {
                if u == v {
                    continue;
                }

                if !self.g.has_edge(u, v) {
                    continue;
                }

                for w in self.g.nodes() {
                    if v == w || u == w {
                        continue;
                    }

                    if self.g.has_edge(u, w) && !self.g.has_edge(v, w) {
                        // `vuw` is a conflict triple!
                        triple = Some((v, u, w));
                        break 'outer;
                    }
                }
            }
        }

        let (v, u, _w) = match triple {
            None => {
                // No more conflict triples.
                // If there are still zero-edges, we need to "cash in" the 0.5 edit cost we deferred
                // when merging.
                let mut zero_count = 0.0;
                for u in 0..self.g.size() {
                    continue_if_not_present!(self.g, u);
                    for v in (u + 1)..self.g.size() {
                        continue_if_not_present!(self.g, v);
                        if self.g.get(u, v).is_zero() {
                            zero_count += 1.0;
                        }
                    }
                }

                self.k -= zero_count / 2.0;

                dbg_trace_indent!(
                    self,
                    _k_start,
                    "Found no triple, realized {} zero-edges.\n",
                    zero_count
                );

                self.path_log.push_str(&format!(
                    "Found no triple, realized {} zero-edges.\n",
                    zero_count
                ));

                if self.k < 0.0 {
                    // not enough cost left over to actually realize those zero-edges.
                    return None;
                }

                return Some(self);
            }
            Some(t) => t,
        };

        dbg_trace_indent!(
            self,
            _k_start,
            "Found triple ({:?}-{:?}-{:?}), branching",
            self.imap[v],
            self.imap[u],
            self.imap[_w]
        );

        // Found a conflict triple, now branch into 2 cases:
        // 1. Set uv to forbidden
        let res1 = {
            let mut branched = self.fork_new_branch();
            //let uv = branched.g.get_mut(u, v);
            let uv = branched.g.get(u, v);
            // TODO: Might not need this check after edge merging is in? Maybe?
            if uv.is_finite() {
                branched.k = self.k - uv as f32;

                if branched.k >= 0.0 {
                    //log_indent!(
                    dbg_trace_indent!(
                        self,
                        _k_start,
                        //log::Level::Warn,
                        "Branch: Set {:?}-{:?} forbidden, k after edit: {} !",
                        branched.imap[u],
                        branched.imap[v],
                        branched.k
                    );
                    branched.path_log.push_str(&format!(
                        "Branch: Set {:?}-{:?} forbidden, k after edit: {} !\n",
                        branched.imap[u], branched.imap[v], self.k
                    ));

                    Edit::delete(&mut branched.edits, &branched.imap, u, v);
                    branched.g.set(u, v, InfiniteNum::NEG_INFINITY);

                    branched.find_cluster_editing()
                } else {
                    dbg_trace_indent!(
                        self,
                        _k_start,
                        "Skipping Branch: Setting {:?}-{:?} to forbidden reduces k past 0!",
                        self.imap[u],
                        self.imap[v]
                    );
                    None
                }
            } else {
                None
            }
        };

        // 2. Merge uv
        let res2 = {
            //let mut branched = self.fork_new_branch();
            //let uv = branched.g.get_mut(u, v);
            let uv = self.g.get(u, v);
            // TODO: Might not need this check after edge merging is in? Maybe?
            if uv.is_finite() {
                self.path_log.push_str(&format!(
                    "Branch: Merge {:?}-{:?}, k after merging: {} !\n",
                    self.imap[u], self.imap[v], self.k
                ));

                let _imap_u = self.imap[u].clone();
                let _imap_v = self.imap[v].clone();
                self.merge(u, v);

                if self.k >= 0.0 {
                    //log_indent!(
                    dbg_trace_indent!(
                        self,
                        _k_start,
                        //log::Level::Warn,
                        "Branch: Merge {:?}-{:?}, k after merging: {} !",
                        _imap_u,
                        _imap_v,
                        self.k
                    );
                    self.find_cluster_editing()
                } else {
                    dbg_trace_indent!(
                        self,
                        _k_start,
                        "Skipping Branch: Merging {:?}-{:?} reduces k past 0!",
                        _imap_u,
                        _imap_v
                    );
                    None
                }
            } else {
                None
            }
        };

        match (res1, res2) {
            (None, None) => None,
            (None, Some(r)) => Some(r),
            (Some(r), None) => Some(r),
            (Some(r1), Some(r2)) => {
                /*dbg_trace_indent!(
                    self,
                    _k_start,
                    "Both branches succeeded, with k1={} and k2={}.",
                    r1.k,
                    r2.k
                );*/

                if r1.k > r2.k {
                    Some(r1)
                } else {
                    Some(r2)
                }
            }
        }
    }

    /// Merge u and v. The merged vertex becomes the new vertex at index u in the graph, while v is
    /// marked as not present anymore.
    pub fn merge(&mut self, u: usize, v: usize) {
        assert!(self.g.is_present(u));
        assert!(self.g.is_present(v));

        let _start_k = self.k;
        let _start_edit_len = self.edits.len();

        for w in 0..self.g.size() {
            if w == u || w == v || !self.g.is_present(w) {
                continue;
            }

            let uw = self.g.get(u, w);
            let vw = self.g.get(v, w);

            if uw > Weight::ZERO {
                if vw < Weight::ZERO {
                    // (+, -)
                    let new_weight = self.merge_nonmatching_nonzero(u, v, w);
                    self.g.set(u, w, new_weight);
                } else if vw > Weight::ZERO {
                    // (+, +)
                    self.g.set(u, w, uw + vw);
                } else {
                    // (+, 0)
                    self.k -= 0.5;
                    Edit::insert(&mut self.edits, &self.imap, v, w);
                }
            } else if uw < Weight::ZERO {
                if vw < Weight::ZERO {
                    // (-, -)
                    self.g.set(u, w, uw + vw);
                } else if vw > Weight::ZERO {
                    // (-, +)
                    let new_weight = self.merge_nonmatching_nonzero(v, u, w);
                    self.g.set(u, w, new_weight);
                } else {
                    // (-, 0)
                    self.k -= 0.5;
                }
            } else {
                if vw < Weight::ZERO {
                    // (0, -)
                    self.k -= 0.5;
                    self.g.set(u, w, vw);
                } else if vw > Weight::ZERO {
                    // (0, +)
                    self.k -= 0.5;
                    self.g.set(u, w, vw);
                    Edit::insert(&mut self.edits, &self.imap, u, w);
                } else {
                    // (0, 0)
                    self.k -= 0.5;
                }
            }
        }

        dbg_trace_indent!(
            self,
            _start_k,
            "Merged {:?} and {:?}. k was {}, is now {}. New edits: {:?}",
            self.imap[u],
            self.imap[v],
            _start_k,
            self.k,
            &self.edits[_start_edit_len..]
        );

        self.g.set_not_present(v);
        let mut imap_v = self.imap.take(v);
        self.imap[u].append(&mut imap_v);
    }

    // `merge` helper. Merge uw and vw, under the assumption that weight(uw) > 0 and weight(vw) < 0.
    // Adds appropriate edits to `edits` and modifies `k` and returns the correct new weight for the
    // merged edge.
    fn merge_nonmatching_nonzero(&mut self, u: usize, v: usize, w: usize) -> Weight {
        let uw = self.g.get(u, w);
        let vw = self.g.get(v, w);

        if (uw + vw).is_zero() {
            self.k -= uw as f32 - 0.5;
            Edit::delete(&mut self.edits, &self.imap, u, w);
            return Weight::ZERO;
        } else {
            if uw > -vw {
                self.k -= -vw as f32;
                Edit::insert(&mut self.edits, &self.imap, v, w);
            } else {
                self.k -= uw as f32;
                Edit::delete(&mut self.edits, &self.imap, u, w);
            }
            return uw + vw;
        }
    }
}
