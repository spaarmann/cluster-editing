# Incremental Correctness Testing/Diffing

## Preparation

First, compute solution with the old "knonw-good" version. There are two options, either
attempt to solve all instances in a short time, or use a predetermined list of instances.

### Attempt to solve all instances
```
$ cargo build --release
$ cargo run --release --bin runner -- --num-workers 12 --timeout 5 runner_out instances/exact*.gr
```
to find which instances are currently solvable within ~5 minutes.
This may take up to ~45 minutes. If the computer is used for something else at the same
time, `num-workers` should probably be reduced, with a corresponding effect on runtime.

To collect a list of instances that were solved, execute
```
$ grep -r "Output graph has" runner_out/ | cut -d ":" -f 1 | cut -d "/" -f 2 | sed 's/.out//g' | sed 's/^/instances\//' > tests/correctness_instances
```

And finally, collect the computed solutions for those instances:
```
$ rm tests/correctness_solutions/*
$ for f in (cat tests/correctness_instances); set out_file (echo $f | cut -d "/" -f 2 | sed 's/^/runner_out\//' | sed 's/$/\.out/'); grep 'Final set' $out_file | sed 's/\[.*\] //' > (echo $f | cut -d "/" -f 2 | sed 's/^/tests\/correctness_solutions\//' | sed 's/$/\.good/'); end
```

### Use a list of instances

Provide a `tests/correctness_instances` file, and the recompute and store solutions for them like this:

```
$ rm test/correctness_solutions/*
$ cargo build --release
$ cargo run --release --bin runner -- --timeout 5 runner_out (cat tests/correctness_instances)
$ for f in (cat tests/correctness_instances); set out_file (echo $f | cut -d "/" -f 2 | sed 's/^/runner_out\//' | sed 's/$/\.out/'); grep 'Final set' $out_file | sed 's/\[.*\] //' > (echo $f | cut -d "/" -f 2 | sed 's/^/tests\/correctness_solutions\//' | sed 's/$/\.good/'); end
```

## Testing Changes

Make the change that should be tested. Then:

```
$ rm test/correctness_solutions/*.test
$ cargo build --release
$ cargo run --release --bin runner -- --timeout 5 runner_out (cat tests/correctness_instances)
$ for f in (cat tests/correctness_instances); set out_file (echo $f | cut -d "/" -f 2 | sed 's/^/runner_out\//' | sed 's/$/\.out/'); grep 'Final set' $out_file | sed 's/\[.*\] //' > (echo $f | cut -d "/" -f 2 | sed 's/^/tests\/correctness_solutions\//' | sed 's/$/\.test/'); end
```

To compare the solutions, use e.g.
```
$ for f in tests/correctness_solutions/*.good; echo $f:; diff $f (echo $f | sed 's/good/test'); end
```
