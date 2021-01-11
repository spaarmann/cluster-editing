# Incremental Performance Testing/Diffing

## Preparation

Execute
```
$ cargo build --release
$ cargo run --release --bin runner -- --num-workers 12 --timeout 5 runner_out instances/exact*.gr
```
to find which instances are currently solvable within ~5 minutes.
This may take up to ~45 minutes. If the computer is used for something else at the same
time, `num-workers` should probably be reduced, with a corresponding effect on runtime.

To collect a list of instances that were solved, execute
```
$ grep -r "Output graph has" runner_out/ | cut -d ":" -f 1 | cut -d "/" -f 2 | sed 's/.out//g' | sed 's/^/instances\//' > perftestinstances
```

## Testing

Before the change under inspection, execute
```
$ cargo build --release
$ time cargo run --release --bin runner -- --num-workers 1 --timeout 5 perftest_out (cat perftestinstances)
```
and note the resulting timing.

Make the change that should be tested and then execute the above command(s) again, noting
down the time again. The difference is the time difference among instances that were
already solvable before, but this does obviously not check if further instances became solvable.
