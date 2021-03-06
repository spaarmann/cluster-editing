use crate::{
    conflicts::ConflictStore,
    graph::{GraphWeight, IndexMap},
    induced_costs::InducedCosts,
    reduction,
    util::{FloatKey, InfiniteNum},
    Graph, PetGraph, Weight,
};

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{prelude::*, BufWriter};
use std::path::{Path, PathBuf};

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

#[derive(Default, Debug, Clone)]
pub struct ComponentStatistics {
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

    pub k_red_from_branching: f32,
    pub k_red_from_rules123: f32,
    pub k_red_from_rule4: f32,
    pub k_red_from_rule5: f32,
    pub k_red_from_ind_cost: f32,
    pub k_red_from_early_exit: f32,
    pub k_red_from_zeroes: f32,
}

#[derive(Debug, Default, Clone)]
pub struct Parameters {
    pub full_reduction_interval: i32,
    pub fast_reduction_interval: i32,
    pub debug_opts: HashMap<String, String>,
    pub stats_dir: Option<PathBuf>,

    // This doesn't really belong here, but it's a convenient place
    // to access from everywhere, throughout the whole algorithm.
    pub stats: RefCell<ComponentStatistics>,
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

// This is only used for debug state validation... it'd still be nicer to not put it into a static
// mut, but oh well.
static mut ORIGINAL_INPUT_GRAPH: Option<PetGraph> = None;

pub fn execute_algorithm(graph: &PetGraph, params: &mut Parameters) -> (PetGraph, Vec<Edit>) {
    let mut result = graph.clone();
    let (g, imap) = Graph::<Weight>::new_from_petgraph(&graph);
    let (components, _) = g.split_into_components(&imap);

    unsafe {
        ORIGINAL_INPUT_GRAPH = Some(graph.clone());
    }

    info!(
        "Decomposed input graph into {} components",
        components.len()
    );

    for (i, c) in components.into_iter().enumerate() {
        let (cg, imap) = c;

        if let Some(only_comp) = params.debug_opts.get("only_comp") {
            if i != only_comp.parse().unwrap() {
                continue;
            }
        }

        if params.stats_dir.is_some() {
            let comp_statistics = ComponentStatistics {
                component_node_count: cg.present_node_count(),
                component_edge_count: cg.edge_count(),
                ..Default::default()
            };
            params.stats = RefCell::new(comp_statistics);
        }

        if let Some(ref stats_dir) = params.stats_dir {
            write_stat_initial(&stats_dir, &params.stats.borrow(), i);
        }

        info!("Solving component {}...", i);
        let (k, edits) = find_optimal_cluster_editing(&cg, &imap, &params, i);

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

    info!("Starting sanity check.");

    // Do a sanity check: There should be no conflict triple in the resulting graph at the end.
    for u in result.node_indices() {
        for v in result.neighbors(u) {
            for w in result.neighbors(u) {
                if v == w {
                    continue;
                }

                if result.find_edge(v, w).is_none() {
                    // `vuw` is a conflict triple!
                    panic!(
                        "Result graph still has a conflict triple! v-u-w: {}-{}-{}",
                        v.index(),
                        u.index(),
                        w.index()
                    );
                }
            }
        }
    }

    info!("Computing de-duplicated set of edits.");

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

    (result, edits)
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
    comp_index: usize,
) -> (i32, Vec<Edit>) {
    // TODO: Not sure if executing the algo once with k = 0 is the best
    // way of handling already-disjoint-clique-components.

    let original_node_count = g.size();
    info!(
        "Computing optimal solution for graph with {} nodes.",
        original_node_count
    );

    let mut _path_debugs = String::new();
    let mut instance = ProblemInstance::new(params, g.clone(), imap.clone());
    let k_start = reduction::initial_param_independent_reduction(&mut instance);

    // `initial_param_independent_reduction` currently actually replaces the graph with
    // a completely new one. This isn't a great state of affair, it incurs a good amount
    // of unnecessary overhead, but it only happens once per component, so, eh.
    instance.conflicts = ConflictStore::new_for_graph(&instance.g);
    instance.induced_costs = InducedCosts::new_for_graph(&instance.g);

    info!(
        "Reduced graph from {} nodes to {} nodes using parameter-independent reduction.",
        original_node_count,
        instance.g.size()
    );

    params.stats.borrow_mut().initial_reduction = original_node_count - instance.g.size();

    if let Some(only_k) = params.debug_opts.get("only_k") {
        let only_k = only_k.parse::<usize>().unwrap();
        info!("[driver] Doing a single search with k={}...", only_k);

        let mut instance = instance.fork_new_branch();
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

    let min_cost = instance
        .conflicts
        .min_cost_to_resolve_edge_disjoint_conflicts(&instance.g);

    let mut k = f32::max(min_cost.floor(), k_start);
    loop {
        info!("[driver] Starting search with k={}...", k);

        let mut instance = instance.fork_new_branch();
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

#[derive(Clone)]
pub struct ProblemInstance<'a> {
    pub params: &'a Parameters,
    pub g: Graph<Weight>,
    pub conflicts: ConflictStore,
    pub induced_costs: InducedCosts,
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

impl<'a> ProblemInstance<'a> {
    pub fn new(params: &'a Parameters, g: Graph<Weight>, imap: IndexMap) -> Self {
        let conflicts = ConflictStore::new_for_graph(&g);
        let induced_costs = InducedCosts::new_for_graph(&g);
        Self {
            params,
            g,
            imap,
            conflicts,
            induced_costs,
            k: 0.0,
            k_max: 0.0,
            full_reduction_counter: 0,
            fast_reduction_counter: 0,
            edits: Vec::new(),
            path_log: String::new(),
            r: Some(Default::default()),
        }
    }

    fn fork_new_branch(&self) -> Self {
        self.clone()
    }

    pub fn set(&mut self, u: usize, v: usize, weight: Weight) {
        let prev = self.g.set(u, v, weight);
        self.induced_costs.update(&self.g, u, v, prev, weight);
    }

    /// Tries to find a solution of size <= k for this problem instance.
    // `k` is stored as float because it needs to be compared with and changed by values from
    // the WeightMap a lot, which are floats.
    fn find_cluster_editing(mut self) -> (bool, Self) {
        let _k_start = self.k;

        //self.validate_current_state();

        let min_cost = self
            .conflicts
            .min_cost_to_resolve_edge_disjoint_conflicts(&self.g);
        if self.k < min_cost {
            trace_and_path_log!(
                self,
                _k_start,
                "Need {} to resolve {} edge-disjoint conflicts, can't afford with {}",
                min_cost,
                self.conflicts.edge_disjoint_conflict_count(),
                self.k
            );

            self.params.stats.borrow_mut().k_red_from_early_exit += self.k.max(0.0);
            return (false, self);
        }

        // If k is already 0, we can only succeed if we currently have a solution; there is no point in trying
        // to do further reductions or splitting as we can't afford any edits anyway.
        if self.k > 0.0 {
            /*
            TODO: The component splitting code was not updated for conflict tracking yet.
            let (components, component_map) = self.g.split_into_components(&self.imap);
            if components.len() > 1 {
                // If a connected component decomposes into two components, we calculate
                // the optimum solution for these components separately.

                let _k_start = self.k;
                for (_i, (comp, comp_imap)) in components.into_iter().enumerate() {
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
                        g: comp,
                        imap: comp_imap,
                        ..self
                    };

                    // returns early if we can't even find a solution for the component,
                    // otherwise take the remaining k and proceed to the next component.
                    let (comp_success, comp_instance) = comp_instance.find_cluster_editing();
                    self.k = comp_instance.k;
                    self.edits = comp_instance.edits;
                    self.path_log = comp_instance.path_log;
                    self.r = comp_instance.r;

                    if !comp_success {
                        dbg_trace_indent!(
                            self,
                            _k_start,
                            "Finished component {} with 'no solution found', returning.",
                            _i
                        );
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
            }*/

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
                reduction::full_param_independent_reduction(&mut self, false);
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

            trace_and_path_log!(
                self,
                _k_start,
                "Reduced from k={} to k={}",
                _k_start,
                self.k
            );

            let min_cost = self
                .conflicts
                .min_cost_to_resolve_edge_disjoint_conflicts(&self.g);
            if self.k < min_cost {
                trace_and_path_log!(
                    self,
                    _k_start,
                    "Need {} to resolve {} edge-disjoint conflicts, can't afford with {} after reduction",
                    min_cost,
                    self.conflicts.edge_disjoint_conflict_count(),
                    self.k
                );

                self.params.stats.borrow_mut().k_red_from_early_exit += self.k.max(0.0);
                return (false, self);
            }
        }

        if self.k < 0.0 {
            return (false, self);
        }

        dbg_trace_indent!(self, _k_start, "Searching edge to branch on.");

        // If there are still conflicts left, get the edge with minimum branching number to branch
        // on.
        let branch_edge = if self.conflicts.conflict_count() > 0 {
            self.induced_costs
                .get_edge_with_min_branching_number(&self.g)
                .or_else(|| self.conflicts.get_next_conflict().map(|(u, v, _)| (u, v)))
        } else {
            None
        };

        // TODO: In the presence of zero-edges, it is currently possible that no edge with
        // non-infinite branching number can be found but we still have a conflict, so we fall back
        // on that.
        // This is (I think) because the induced cost calculations don't account for zero-edges.
        // With this workaround it should not be a correctness issue, but for performance it might
        // be nice to calculate more accurate induced costs; it could potentially lead to the
        // induced cost reduction being applicable more often too.

        //let conflict_edge = self.conflicts.get_next_conflict().map(|(u, v, _)| (u, v));
        /*if branch_edge.is_none() && conflict_edge.is_some() {
            let (u, v) = conflict_edge.unwrap();
            let costs = self.induced_costs.get_costs(u, v);
            let uv = self.g.get(u, v);
            let branching_num = InducedCosts::get_branching_number(costs, uv);
            log::error!("Found conflict but no branch_edge. Conflict edge {}-{} ({:?}-{:?}) with uv {}, costs {:?} and branching num {}", u, v, self.imap[u], self.imap[v], uv, costs, branching_num);

            crate::graphviz::print_debug_graph(
                "sfdp",
                "debug.png",
                &self.g.to_petgraph(Some(&self.imap), false),
            );
            crate::graphviz::print_debug_graph(
                "sfdp",
                "debug_inv.png",
                &self.g.to_petgraph(Some(&self.imap), true),
            );
        }
        assert_eq!(branch_edge.is_some(), conflict_edge.is_some());*/

        let (u, v) = match branch_edge {
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
                self.params.stats.borrow_mut().k_red_from_zeroes += zero_count / 2.0;

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
            "Found edge ({},{}) = ({:?}-{:?}) with weight {}, branching",
            u,
            v,
            self.imap[u],
            self.imap[v],
            self.g.get(u, v),
        );

        let edit_len = self.edits.len();
        let oplog_len = self.g.oplog_len();
        let path_len = self.path_log.len();
        let prev_imap = self.imap.clone();
        let conflicts_oplog_len = self.conflicts.oplog_len();
        let induced_costs_oplog_len = self.induced_costs.oplog_len();
        let prev_k = self.k;
        let prev_full_counter = self.full_reduction_counter;
        let prev_fast_counter = self.fast_reduction_counter;

        // Found a conflict triple, now branch into 2 cases:
        // 1. Set uv to forbidden
        let (solution_found, mut this) = {
            let uv = self.g.get(u, v);
            // TODO: Might not need this check after edge merging is in? Maybe?
            if uv.is_finite() {
                if uv > Weight::ZERO {
                    self.k -= uv;
                    self.make_delete_edit(u, v);
                    self.params.stats.borrow_mut().k_red_from_branching += uv;
                }
                if uv.abs() < 0.001 {
                    self.k -= 0.5;
                    self.params.stats.borrow_mut().k_red_from_zeroes += 0.5;
                }

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

                    self.set(u, v, InfiniteNum::NEG_INFINITY);

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
        this.conflicts.rollback_to(conflicts_oplog_len);
        this.induced_costs.rollback_to(induced_costs_oplog_len);
        this.k = prev_k;
        this.full_reduction_counter = prev_full_counter;
        this.fast_reduction_counter = prev_fast_counter;

        // 2. Merge uv
        let res2 = {
            let uv = this.g.get(u, v);
            // TODO: Might not need this check after edge merging is in? Maybe?
            if uv.is_finite() {
                let _imap_u = this.imap[u].clone();
                let _imap_v = this.imap[v].clone();
                this.merge(u, v);

                this.params.stats.borrow_mut().k_red_from_branching += prev_k - this.k;

                if this.k >= 0.0 {
                    trace_and_path_log!(
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

    // These two methods create appropriate `Edit` entries and update the conflict store,
    // anything that calls them is still responsible for updating `k` appropriately.
    // (And also for actually setting any values in the graph.)

    pub fn make_insert_edit(&mut self, u: usize, v: usize) {
        Edit::insert(&mut self.edits, &self.imap, u, v);
        self.conflicts.update_for_insert(&self.g, u, v);
    }

    pub fn make_delete_edit(&mut self, u: usize, v: usize) {
        Edit::delete(&mut self.edits, &self.imap, u, v);
        self.conflicts.update_for_delete(&self.g, u, v);
    }

    /// Merge u and v. The merged vertex becomes the new vertex at index u in the graph, while v is
    /// marked as not present anymore.
    #[allow(clippy::collapsible_else_if)]
    pub fn merge(&mut self, u: usize, v: usize) {
        assert!(self.g.is_present(u));
        assert!(self.g.is_present(v));

        let _start_k = self.k;
        let _start_edit_len = self.edits.len();

        let uv = self.g.get(u, v);
        if uv < Weight::ZERO {
            self.k += uv; // We essentially add the edge if it doesn't exist, which generates cost.
            self.make_insert_edit(u, v);
        }
        if uv.abs() < 0.0001 {
            self.k -= 0.5;
            self.make_insert_edit(u, v);
        }

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
                    self.set(u, w, new_weight);
                } else if vw > Weight::ZERO {
                    // (+, +)
                    self.set(u, w, uw + vw);
                } else {
                    // (+, 0)
                    self.k -= 0.5;
                    self.make_insert_edit(v, w);
                }
            } else if uw < Weight::ZERO {
                if vw < Weight::ZERO {
                    // (-, -)
                    self.set(u, w, uw + vw);
                } else if vw > Weight::ZERO {
                    // (-, +)
                    let new_weight = self.merge_nonmatching_nonzero(v, u, w);
                    self.set(u, w, new_weight);
                } else {
                    // (-, 0)
                    self.k -= 0.5;
                }
            } else {
                if vw < Weight::ZERO {
                    // (0, -)
                    self.k -= 0.5;
                    self.set(u, w, vw);
                } else if vw > Weight::ZERO {
                    // (0, +)
                    self.k -= 0.5;
                    self.set(u, w, vw);
                    self.make_insert_edit(u, w);
                } else {
                    // (0, 0)
                    self.k -= 0.5;
                }
            }
        }

        trace_and_path_log!(
            self,
            _start_k,
            "Merged {:?} and {:?}. k was {}, is now {}. New edits: {:?}",
            self.imap[u],
            self.imap[v],
            _start_k,
            self.k,
            &self.edits[_start_edit_len..]
        );

        self.induced_costs.update_for_not_present(&self.g, v);
        self.g.set_not_present(v);
        let mut imap_v = self.imap.take(v);
        self.imap[u].append(&mut imap_v);
        self.conflicts.update_for_not_present(&self.g, v);
    }

    // `merge` helper. Merge uw and vw, under the assumption that weight(uw) > 0 and weight(vw) < 0.
    // Adds appropriate edits to `edits` and modifies `k` and returns the correct new weight for the
    // merged edge.
    fn merge_nonmatching_nonzero(&mut self, u: usize, v: usize, w: usize) -> Weight {
        let uw = self.g.get(u, w);
        let vw = self.g.get(v, w);

        if (uw + vw).is_zero() {
            self.k -= uw as f32 - 0.5;
            self.make_delete_edit(u, w);
            Weight::ZERO
        } else {
            if uw > -vw {
                self.k -= -vw as f32;
                self.make_insert_edit(v, w);
            } else {
                self.k -= uw as f32;
                self.make_delete_edit(u, w);
            }
            uw + vw
        }
    }

    pub fn validate_current_state(&self) {
        let edits = self.calculate_current_deduplicated_edits();

        let mut zero_count = 0;
        for u in self.g.nodes() {
            for v in (u + 1)..self.g.size() {
                continue_if_not_present!(self.g, v);

                if self.g.get(u, v).abs() < 0.001 {
                    zero_count += 1;
                }
            }
        }

        let extra_k_for_zeroes = zero_count as f32 / 2.0;

        assert_eq!(self.k_max - self.k + extra_k_for_zeroes, edits.len() as f32);
    }

    pub fn calculate_current_deduplicated_edits(&self) -> Vec<Edit> {
        let mut result = unsafe { ORIGINAL_INPUT_GRAPH.as_ref().unwrap().clone() };

        for &edit in &self.edits {
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

        let mut edits = Vec::new();
        let graph = unsafe { ORIGINAL_INPUT_GRAPH.as_ref().unwrap() };
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

        edits
    }
}

fn write_stat_initial(stats_dir: &Path, stats: &ComponentStatistics, component_index: usize) {
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
    stats_dir: &Path,
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
