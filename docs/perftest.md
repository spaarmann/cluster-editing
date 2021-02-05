# Incremental Performance Testing/Diffing

## Preparation

Collect a list of instances that can be solved:
```
$ scripts/compute_solvable_set <timeout> <num_workers>
```

Note down where the run is stored.

## Testing

To get timing for the current revision:
```
$ scripts/perf_test <path to completed instance file from complete run> <timeout> <num_workers>
```
