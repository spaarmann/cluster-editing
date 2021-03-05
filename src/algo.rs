use crate::{
    graph::{GraphView, GraphViewState, GraphWeight, IndexMap},
    reduction,
    util::{FloatKey, InfiniteNum},
    Graph, PetGraph, Weight,
};

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{prelude::*, BufWriter};
use std::path::PathBuf;

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

#[derive(Default, Debug)]
struct ComponentStatistics {
    // fast_param_indep_reduction[max_k][k] = x => in the run starting with k=max_k, at k=k
    // the fast reduction achieved a reduction in `k` of `x`.
    fast_param_indep_reduction: HashMap<usize, HashMap<FloatKey, f32>>,
    // full_param_indep_reduction[max_k][k] = x => in the run starting with k=max_k, at k=k
    // the full reduction achieved a reduction in `k` of `x`.
    full_param_indep_reduction: HashMap<usize, HashMap<FloatKey, f32>>,
    // param_dep_reduction[max_k][k] = x => in the run starting with k=max_k, at k=k
    // the param-dependent reduction achieved a reduction in `k` of `x`.
    param_dep_reduction: HashMap<usize, HashMap<FloatKey, f32>>,
    // Reduction of node count from the initial param-independent reduction.
    initial_reduction: usize,
    component_node_count: usize,
    component_edge_count: usize,
}

#[derive(Debug, Default)]
pub struct Parameters {
    pub full_reduction_interval: i32,
    pub fast_reduction_interval: i32,
    pub debug_opts: HashMap<String, String>,
    pub stats_dir: Option<PathBuf>,

    // This doesn't really belong here, but it's a convenient place
    // to access from everywhere, throughout the whole algorithm.
    stats: RefCell<ComponentStatistics>,
}

impl Parameters {
    pub fn new(
        full_reduction_interval: i32,
        fast_reduction_interval: i32,
        debug_opts: HashMap<String, String>,
        stats_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            full_reduction_interval,
            fast_reduction_interval,
            debug_opts,
            stats_dir,
            stats: Default::default(),
        }
    }
}

pub fn execute_algorithm(graph: &PetGraph, mut params: Parameters) -> PetGraph {
    let mut result = graph.clone();
    let (mut graph_storage, imap) = Graph::<Weight>::new_from_petgraph(&graph);
    let g = GraphView::new(&mut graph_storage);
    let (components, _) = g.split_into_components();

    info!(
        "Decomposed input graph into {} components",
        components.len()
    );

    for (i, c_state) in components.into_iter().enumerate() {
        let c = c_state.realize(&mut graph_storage);

        if params.stats_dir.is_some() {
            let mut comp_statistics = ComponentStatistics::default();
            comp_statistics.component_node_count = c.present_node_count();
            comp_statistics.component_edge_count = c.edge_count();
            params.stats = RefCell::new(comp_statistics);
        }

        if let Some(ref stats_dir) = params.stats_dir {
            write_stat_initial(&stats_dir, &params.stats.borrow(), i);
        }

        info!("Solving component {}...", i);
        let c_state = c.into_state();
        let (k, edits) =
            find_optimal_cluster_editing(&mut graph_storage, &c_state, &imap, &params, i);

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
    graph_storage: &mut Graph<Weight>,
    g: &GraphViewState,
    imap: &IndexMap,
    params: &Parameters,
    comp_index: usize,
) -> (i32, Vec<Edit>) {
    // TODO: Not sure if executing the algo once with k = 0 is the best
    // way of handling already-disjoint-clique-components.

    let initial_state = g.clone();
    let initial_view = initial_state.realize(graph_storage);

    let original_node_count = initial_view.size();
    info!(
        "Computing optimal solution for graph with {} nodes.",
        original_node_count
    );

    let mut path_log = String::new();
    let (mut my_storage, imap, k_start) =
        reduction::initial_crit_clique_reduction(graph_storage.clone(), &g, imap, &mut path_log);

    let cloned_graph = my_storage.clone();
    let initial_reduce_view = GraphView::new(&mut my_storage);

    let mut instance = ProblemInstance::new(params, initial_reduce_view, imap.clone());
    reduction::initial_param_independent_reduction(&mut instance);

    info!(
        "Reduced graph from {} nodes to {} nodes using parameter-independent reduction.",
        original_node_count,
        instance.g.size()
    );
    params.stats.borrow_mut().initial_reduction = original_node_count - instance.g.size();

    if let Some(only_k) = params.debug_opts.get("only_k") {
        let only_k = only_k.parse::<usize>().unwrap();
        info!("[driver] Doing a single search with k={}...", only_k);

        let mut instance_storage = cloned_graph.clone();
        let mut instance = instance.fork_new_branch(&mut instance_storage);
        instance.k = only_k as f32;
        instance.k_max = only_k as f32;
        if let (true, instance) = instance.find_cluster_editing() {
            log::info!("Found solution, final path log:\n{}", instance.path_log);
            return (instance.k as i32, instance.edits);
        } else {
            log::warn!("Found no solution!");
            return (0, Vec::new());
        }
    }

    let mut k = k_start;
    loop {
        info!("[driver] Starting search with k={}...", k);

        let mut instance_storage = cloned_graph.clone();
        let mut instance = instance.fork_new_branch(&mut instance_storage);
        instance.k = k;
        instance.k_max = k;
        let (success, instance) = instance.find_cluster_editing();

        if let Some(ref stats_dir) = params.stats_dir {
            write_stat_block(stats_dir, &params.stats.borrow(), comp_index, k as usize);
        }

        if success {
            if !instance.path_log.is_empty() {
                log::info!("Final path debug log:\n{}", instance.path_log);
            }

            return (instance.k as i32, instance.edits);
        }

        k += 1.0;
    }
}

pub struct ProblemInstance<'a, 'g> {
    pub params: &'a Parameters,
    pub g: GraphView<'g, Weight>,
    pub imap: IndexMap,
    pub k: f32,
    pub k_max: f32,
    pub full_reduction_counter: i32,
    pub fast_reduction_counter: i32,
    pub edits: Vec<Edit>,
    pub path_log: String,
    // Helpers for `reduction`, stored here to avoid allocating new ones as much.
    pub r: Option<reduction::ReductionStorage>,
}

impl<'a, 'g> ProblemInstance<'a, 'g> {
    pub fn new(params: &'a Parameters, g: GraphView<'g, Weight>, imap: IndexMap) -> Self {
        Self {
            params,
            g,
            imap,
            k: 0.0,
            k_max: 0.0,
            full_reduction_counter: 0,
            fast_reduction_counter: 0,
            edits: Vec::new(),
            path_log: String::new(),
            r: Some(Default::default()),
        }
    }

    // TODO: Rename, we're not actually using this for new branches ^^'
    fn fork_new_branch<'n>(&self, new_storage: &'n mut Graph<Weight>) -> ProblemInstance<'a, 'n> {
        let new_state = self.g.cloned_state();
        let new_g = new_state.realize(new_storage);
        ProblemInstance {
            params: self.params,
            g: new_g,
            imap: self.imap.clone(),
            k: self.k,
            k_max: self.k_max,
            full_reduction_counter: self.full_reduction_counter,
            fast_reduction_counter: self.fast_reduction_counter,
            edits: self.edits.clone(),
            path_log: self.path_log.clone(),
            r: self.r.clone(),
        }
    }

    /// Tries to find a solution of size <= k for this problem instance.
    // `k` is stored as float because it needs to be compared with and changed by values from
    // the WeightMap a lot, which are floats.
    fn find_cluster_editing(mut self) -> (bool, Self) {
        // If k is already 0, we can only succeed if we currently have a solution; there is no point in trying
        // to do further reductions or splitting as we can't afford any edits anyway.
        let _k_start = self.k;

        if self.k > 0.0 {
            let full_view = self.g;
            let (components, component_map) = full_view.split_into_components();

            let (mut graph_storage, full_state) = full_view.into_parts();

            if components.len() > 1 {
                // If a connected component decomposes into two components, we calculate
                // the optimum solution for these components separately.

                let _k_start = self.k;
                for (_i, comp_state) in components.into_iter().enumerate() {
                    dbg_trace_indent!(
                        self,
                        _k_start,
                        "Starting component {}, remaining k is {}",
                        _i,
                        self.k
                    );
                    append_path_log!(
                        self,
                        "Starting component {}, remaining k is {}\n",
                        _i,
                        self.k
                    );

                    let comp_instance = ProblemInstance {
                        //params: self.params,
                        g: comp_state.realize(graph_storage),
                        ..self
                    };

                    // returns early if we can't even find a solution for the component,
                    // otherwise take the remaining k and proceed to the next component.
                    let (comp_success, comp_instance) = comp_instance.find_cluster_editing();
                    self.k = comp_instance.k;
                    self.edits = comp_instance.edits;
                    self.path_log = comp_instance.path_log;
                    self.r = comp_instance.r;
                    self.imap = comp_instance.imap;

                    graph_storage = comp_instance.g.into_parts().0;

                    if !comp_success {
                        dbg_trace_indent!(
                            self,
                            _k_start,
                            "Finished component {} with 'no solution found', returning.",
                            _i
                        );
                        self.g = full_state.realize(graph_storage);
                        return (false, self);
                    }

                    dbg_trace_indent!(
                        self,
                        _k_start,
                        "Finished component {}, remaining k is {}",
                        _i,
                        self.k
                    );
                    append_path_log!(
                        self,
                        "Finished component {}, remaining k is {}\n",
                        _i,
                        self.k
                    );
                }

                self.g = full_state.realize(graph_storage);

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

                // TODO: I believe this is currently *correct* even though it does not properly
                // create oplog entries. This is because the actual graph is not modified at all,
                // we only modify the components and create new edits. Thus no oplog-rollback is
                // necessary for the original graph.
                // *However*, this seems pretty fragile and easy to mess up in the future. If we
                // actually keep component splitting in some way, it should do this better.

                if self.k >= 0.0 {
                    return (true, self);
                } else {
                    return (false, self);
                }
            } else {
                self.g = full_state.realize(graph_storage);
            }

            dbg_trace_indent!(self, self.k, "Performing reduction");

            let k_before_param_dep_reduction = self.k;
            reduction::param_dependent_reduction(&mut self);

            if self.params.stats_dir.is_some() {
                self.params
                    .stats
                    .borrow_mut()
                    .param_dep_reduction
                    .entry(self.k_max as usize)
                    .or_default()
                    .insert(
                        FloatKey(k_before_param_dep_reduction),
                        k_before_param_dep_reduction - self.k,
                    );
            }

            let k_before_indep_reduction = self.k;
            if self.full_reduction_counter == 0 {
                reduction::full_param_independent_reduction(&mut self);
                self.full_reduction_counter = self.params.full_reduction_interval;

                if self.params.stats_dir.is_some() {
                    self.params
                        .stats
                        .borrow_mut()
                        .full_param_indep_reduction
                        .entry(self.k_max as usize) // k_max is always integer-valued, unlike k
                        .or_default()
                        .insert(FloatKey(_k_start), k_before_indep_reduction - self.k);
                }
            } else {
                self.full_reduction_counter -= 1;

                if self.fast_reduction_counter == 0 {
                    reduction::fast_param_independent_reduction(&mut self);
                    self.fast_reduction_counter = self.params.fast_reduction_interval;

                    if self.params.stats_dir.is_some() {
                        self.params
                            .stats
                            .borrow_mut()
                            .fast_param_indep_reduction
                            .entry(self.k_max as usize) // k_max is always integer-valued, unlike k
                            .or_default()
                            .insert(FloatKey(_k_start), k_before_indep_reduction - self.k);
                    }
                } else {
                    self.fast_reduction_counter -= 1;
                }
            }

            dbg_trace_indent!(
                self,
                _k_start,
                "Reduced from k={} to k={}",
                _k_start,
                self.k
            );
            append_path_log!(self, "Reduced from k={} to k={}\n", _k_start, self.k);
        }

        if self.k < 0.0 {
            return (false, self);
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

                append_path_log!(
                    self,
                    "Found no triple, realized {} zero-edges.\n",
                    zero_count
                );

                if self.k < 0.0 {
                    // not enough cost left over to actually realize those zero-edges.
                    return (false, self);
                }

                return (true, self);
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

        let edit_len = self.edits.len();
        let oplog_len = self.g.oplog_len();
        let path_len = self.path_log.len();
        let prev_imap = self.imap.clone();
        let prev_k = self.k;
        let prev_full_counter = self.full_reduction_counter;
        let prev_fast_counter = self.fast_reduction_counter;

        // Found a conflict triple, now branch into 2 cases:
        // 1. Set uv to forbidden
        let (solution_found, mut this) = {
            let uv = self.g.get(u, v);
            // TODO: Might not need this check after edge merging is in? Maybe?
            if uv.is_finite() {
                self.k = self.k - uv as f32;

                if self.k >= 0.0 {
                    dbg_trace_indent!(
                        self,
                        _k_start,
                        "Branch: Set {:?}-{:?} forbidden, k after edit: {} !",
                        self.imap[u],
                        self.imap[v],
                        self.k
                    );
                    append_path_log!(
                        self,
                        "Branch: Set {:?}-{:?} forbidden, k after edit: {} !\n",
                        self.imap[u],
                        self.imap[v],
                        self.k
                    );

                    Edit::delete(&mut self.edits, &self.imap, u, v);
                    self.g.set(u, v, InfiniteNum::NEG_INFINITY);

                    self.find_cluster_editing()
                } else {
                    dbg_trace_indent!(
                        self,
                        _k_start,
                        "Skipping Branch: Setting {:?}-{:?} to forbidden reduces k past 0!",
                        self.imap[u],
                        self.imap[v]
                    );
                    (false, self)
                }
            } else {
                (false, self)
            }
        };

        if solution_found {
            return (true, this);
        }

        this.edits.truncate(edit_len);
        this.g.rollback_to(oplog_len);
        this.path_log.truncate(path_len);
        this.imap = prev_imap;
        this.k = prev_k;
        this.full_reduction_counter = prev_full_counter;
        this.fast_reduction_counter = prev_fast_counter;

        // 2. Merge uv
        let res2 = {
            let uv = this.g.get(u, v);
            // TODO: Might not need this check after edge merging is in? Maybe?
            if uv.is_finite() {
                append_path_log!(
                    this,
                    "Branch: Merge {:?}-{:?}, k after merging: {} !\n",
                    this.imap[u],
                    this.imap[v],
                    this.k
                );

                let _imap_u = this.imap[u].clone();
                let _imap_v = this.imap[v].clone();
                this.merge(u, v);

                if this.k >= 0.0 {
                    dbg_trace_indent!(
                        this,
                        _k_start,
                        "Branch: Merge {:?}-{:?}, k after merging: {} !",
                        _imap_u,
                        _imap_v,
                        this.k
                    );
                    this.find_cluster_editing()
                } else {
                    dbg_trace_indent!(
                        this,
                        _k_start,
                        "Skipping Branch: Merging {:?}-{:?} reduces k past 0!",
                        _imap_u,
                        _imap_v
                    );
                    (false, this)
                }
            } else {
                (false, this)
            }
        };

        res2
    }

    /// Merge u and v. The merged vertex becomes the new vertex at index u in the graph, while v is
    /// marked as not present anymore.
    pub fn merge(&mut self, u: usize, v: usize) {
        assert!(self.g.is_present(u));
        assert!(self.g.is_present(v));
        assert!(self.g.get(u, v) >= Weight::ZERO);

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

fn write_stat_initial(stats_dir: &PathBuf, stats: &ComponentStatistics, component_index: usize) {
    std::fs::create_dir_all(stats_dir).unwrap();

    let info_path = stats_dir.join(format!("comp_{}_info.stats", component_index));
    std::fs::write(
        &info_path,
        format!(
            "initial_reduction: {}\n\
            node_count: {}\n\
            edge_count: {}",
            stats.initial_reduction, stats.component_node_count, stats.component_edge_count
        )
        .as_bytes(),
    )
    .unwrap();
}

fn write_stat_block(
    stats_dir: &PathBuf,
    stats: &ComponentStatistics,
    component_index: usize,
    k_max: usize,
) {
    if let Some(reductions) = stats.fast_param_indep_reduction.get(&k_max) {
        let path = stats_dir.join(format!("comp_{}_k{}_fast.stats", component_index, k_max));
        let file = File::create(&path).unwrap();
        let mut writer = BufWriter::new(file);
        writeln!(writer, "k | red").unwrap();
        for (k, v) in reductions {
            writeln!(writer, "{} | {}", k.0, v).unwrap();
        }
        writer.flush().unwrap();
    }

    if let Some(reductions) = stats.full_param_indep_reduction.get(&k_max) {
        let path = stats_dir.join(format!("comp_{}_k{}_full.stats", component_index, k_max));
        let file = File::create(&path).unwrap();
        let mut writer = BufWriter::new(file);
        writeln!(writer, "k | red").unwrap();
        for (k, v) in reductions {
            writeln!(writer, "{} | {}", k.0, v).unwrap();
        }
        writer.flush().unwrap();
    }

    if let Some(reductions) = stats.param_dep_reduction.get(&k_max) {
        let path = stats_dir.join(format!(
            "comp_{}_k{}_paramdep.stats",
            component_index, k_max
        ));
        let file = File::create(&path).unwrap();
        let mut writer = BufWriter::new(file);
        writeln!(writer, "k | red").unwrap();
        for (k, v) in reductions {
            writeln!(writer, "{} | {}", k.0, v).unwrap();
        }
        writer.flush().unwrap();
    }
}
