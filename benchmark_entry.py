from benchopt.benchmark import Benchmark
from benchopt import run_benchmark

bench_ws = Benchmark('./benchmark_folder')

run_benchmark(bench_ws, max_runs=25, n_jobs=1,
              solver_names=['celer', 'skglm'], n_repetitions=1)
